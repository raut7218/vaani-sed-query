"""Fast (seconds) precondition check for a real Kaggle run.

Four full-scale "Save & Run All" attempts on this pipeline each burned close
to an hour of GPU time before failing on something knowable up front: a
missing HF_TOKEN, or disk exhaustion from downloading/generating more audio
than the session's disk quota can hold. This script checks both - reachable
credentials and a real disk budget against the corpus's actual size on the
server and the run's own configured targets - without downloading a single
audio file, so a bad configuration fails in seconds instead of an hour.

    python scripts/preflight.py --work /kaggle/working --max-shards 0 \
        --n-synthetic 20000 --n-pretrain 60000

Exits non-zero (and prints exactly what to change) on a fatal problem: no
route to the corpus, or corpus+synthetic alone would exceed free disk even
before splice-pretrain generation gets a chance to self-limit. A budget
that's tight only for splice-pretrain is reported, not failed, since
make_splice_pretrain.py already stops cleanly instead of crashing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_data import REPO, human, list_remote, resolve_token  # noqa: E402

# Empirically measured from this pipeline's own real runs (16 kHz mono FLAC):
# the full 90,635-clip / 154.588 h corpus, 20,000 synthetic clips, and a
# partial splice-pretrain run together used ~19 GB of a ~20 GB Kaggle
# /kaggle/working quota right before the disk guard fired - about 65 MB per
# hour of audio, all-in (manifest/VAD/misc overhead included).
MB_PER_HOUR = 65.0
CHECKPOINT_OVERHEAD_GB = 1.0     # ATST-frame + BEATs encoder checkpoints
SAFETY_MARGIN_GB = 5.0           # headroom checkpoint I/O needs - see make_*.py


def hours_for(need_shards: int, total_shards: int, total_hours: float) -> float:
    return total_hours if not need_shards else total_hours * need_shards / total_shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="the run's working directory")
    ap.add_argument("--data-from", default="", help="skip the corpus-size check if set")
    ap.add_argument("--token", default="")
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--n-synthetic", type=int, default=0)
    ap.add_argument("--n-pretrain", type=int, default=0)
    ap.add_argument("--clip-len", type=float, default=8.0)
    args = ap.parse_args()

    problems, notes = [], []
    corpus_hours = 0.0

    if not args.data_from:
        token = resolve_token(args.token)
        if not token:
            problems.append(
                "No HF_TOKEN found (checked --token, Kaggle/Colab secret, env, cached "
                "login) and --data-from is empty - no route to the audio corpus.")
        else:
            try:
                remote = list_remote(REPO, token)
                shards = [f for f in remote if f["path"].endswith(".parquet")]
                # Challenge-page figure, same constant download_data.py uses.
                corpus_hours = hours_for(args.max_shards, len(shards), 154.6)
            except Exception as e:  # noqa: BLE001
                problems.append("Could not reach the dataset repo to verify access: %s" % e)

    synth_hours = args.n_synthetic * args.clip_len / 3600
    pretrain_hours = args.n_pretrain * args.clip_len / 3600

    base_gb = (corpus_hours + synth_hours) * MB_PER_HOUR / 1024 + CHECKPOINT_OVERHEAD_GB
    pretrain_gb = pretrain_hours * MB_PER_HOUR / 1024

    Path(args.work).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.work).free / 1024 ** 3

    print("[preflight] %.1f GB free at %s" % (free_gb, args.work))
    print("[preflight] projected: corpus %.1fh + synthetic %.1fh -> %.1f GB "
          "(+ %.1f GB checkpoint-I/O margin)"
          % (corpus_hours, synth_hours, base_gb, SAFETY_MARGIN_GB))
    if args.n_pretrain:
        print("[preflight] splice-pretrain target %.1fh -> %.1f GB "
              "(self-limits if it doesn't fit, see make_splice_pretrain.py)"
              % (pretrain_hours, pretrain_gb))

    if base_gb + SAFETY_MARGIN_GB > free_gb:
        problems.append(
            "Corpus + synthetic data alone (%.1f GB) plus the %.1f GB checkpoint-I/O "
            "safety margin exceeds free space (%.1f GB). Lower --max-shards or "
            "--n-synthetic, or attach a prepared corpus via DATA_FROM."
            % (base_gb, SAFETY_MARGIN_GB, free_gb))
    elif args.n_pretrain and base_gb + pretrain_gb + SAFETY_MARGIN_GB > free_gb:
        fit_hours = max(0.0, free_gb - base_gb - SAFETY_MARGIN_GB) * 1024 / MB_PER_HOUR
        notes.append(
            "splice-pretrain will likely stop around ~%d clips (~%.1fh), short of "
            "the %d requested - expected, not a failure."
            % (int(fit_hours * 3600 / args.clip_len), fit_hours, args.n_pretrain))

    for n in notes:
        print("[preflight] note: %s" % n)
    if problems:
        raise SystemExit(
            "PREFLIGHT FAILED - fix before spending GPU time:\n"
            + "\n".join("  - %s" % p for p in problems))
    print("[preflight] OK")


if __name__ == "__main__":
    main()
