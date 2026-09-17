"""Full-corpus splice-boundary self-supervision for the query model.

Same mechanism as make_synthetic.py - cut a segment out of one clip and paste
it into another at a known position, so the boundary is exact by
construction - but donors are drawn from the *whole* corpus (any tier, not
just single-tag bronze) since the pretext task only needs "is there a splice
here", not the donor's real class. That turns 154.6 h of audio into an
effectively unlimited supply of boundary supervision - every draw combines a
different host, donor, placement, SNR and crossfade width - at a scale no
amount of the 20 h gold tier can match, before the model ever sees a human
timestamp. `--ramp-ms` is swept wide (2-80 ms) rather than fixed, so the head
learns to localise both hard edges and soft ones instead of overfitting to
one splice signature.

python scripts/make_splice_pretrain.py --data data/vaani --out data/vaani_pretrain -n 60000
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_synthetic import energetic_window, read_audio               # noqa: E402
from src.data.labels import LabelEncoder                              # noqa: E402


def load_manifest(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--num", type=int, default=60000)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--clip-len", type=float, default=8.0)
    ap.add_argument("--events-per-clip", type=int, default=2,
                    help="max synthetic splices pasted into one host")
    ap.add_argument("--snr-db", type=float, nargs=2, default=(-3.0, 18.0))
    ap.add_argument("--ramp-ms", type=float, nargs=2, default=(2.0, 80.0),
                    help="crossfade width range - varied so the model does not "
                         "overfit to one splice signature")
    ap.add_argument("--class-name", default="",
                    help="pseudo-class fed to the frame/clip aux heads; default "
                         "is the encoder's first class - those heads are not "
                         "the point of this pretext task, only the boundary is")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    data, out = Path(args.data), Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    le = LabelEncoder(expand_vehicle=True)
    pseudo_cls = args.class_name or le.classes[0]

    recs = load_manifest(data / "manifest.jsonl")
    # Any clip long enough to donate a slice - not just single-tag bronze, so
    # the whole corpus (gold + silver + bronze) is usable donor material.
    donors = [r for r in recs if r.get("duration", 0) >= 0.3]
    # Hosts still need to be event-free, so a pasted splice never overlaps a
    # real annotation it does not know about.
    hosts = [r for r in recs if not r.get("events")]
    if not hosts:
        hosts = recs
    print("[splice] %d donors (full corpus), %d hosts" % (len(donors), len(hosts)))
    if not donors or not hosts:
        raise SystemExit("empty manifest - is --data correct?")

    n_samp = int(args.clip_len * args.sr)
    written, man = 0, (out / "manifest.jsonl").open("w", encoding="utf-8")
    MIN_FREE_BYTES = 2 * 1024**3

    for i in range(args.num):
        if written % 500 == 0 and shutil.disk_usage(out).free < MIN_FREE_BYTES:
            print("[splice] low disk space - stopping at %d clips (target was %d)"
                  % (written, args.num), flush=True)
            break
        host = rng.choice(hosts)
        try:
            hy = read_audio(data / host["path"], args.sr)
        except Exception:                                             # noqa: BLE001
            continue
        buf = np.zeros((n_samp,), "float32")
        m = min(len(hy), n_samp)
        buf[:m] = hy[:m]
        host_rms = float(np.sqrt(np.mean(buf[:m] ** 2) + 1e-12))

        events, occupied = [], []
        for _ in range(rng.randint(1, args.events_per_clip)):
            drec = rng.choice(donors)
            try:
                dy = read_audio(data / drec["path"], args.sr)
            except Exception:                                         # noqa: BLE001
                continue
            a, b = energetic_window(dy, args.sr, 0.15, 3.0, rng)
            seg = dy[a:b]
            if len(seg) < int(0.05 * args.sr):
                continue
            for _try in range(8):
                t0 = rng.uniform(0.0, max(0.0, args.clip_len - len(seg) / args.sr))
                t1 = t0 + len(seg) / args.sr
                if all(t1 <= o0 or t0 >= o1 for o0, o1 in occupied):
                    break
            else:
                continue
            s0 = int(t0 * args.sr)
            seg_rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
            snr = float(np_rng.uniform(*args.snr_db))
            gain = host_rms / max(seg_rms, 1e-9) * (10.0 ** (snr / 20.0))
            ramp_ms = float(np_rng.uniform(*args.ramp_ms))
            ramp = min(max(2, int(ramp_ms / 1000 * args.sr)), len(seg) // 2)
            w = np.ones(len(seg), "float32")
            w[:ramp] = np.hanning(2 * ramp)[:ramp]
            w[-ramp:] = np.hanning(2 * ramp)[ramp:]
            buf[s0:s0 + len(seg)] += (seg * w * gain).astype("float32")
            occupied.append((t0, t1))
            events.append({"cls": pseudo_cls, "start": round(t0, 4), "end": round(t1, 4)})

        if not events:
            continue
        peak = float(np.abs(buf).max())
        if peak > 1.0:
            buf = buf / peak * 0.98

        uid = "splice_%06d" % i
        sf.write(str(out / "audio" / (uid + ".flac")), buf, args.sr)
        man.write(json.dumps({
            "uid": uid, "path": "audio/%s.flac" % uid, "duration": round(args.clip_len, 4),
            # Exact by construction, same as make_synthetic.py's bronze splices.
            "tier": "gold", "state": "SPLICE", "district": "SPLICE",
            "events": sorted(events, key=lambda e: e["start"]),
            "clip_labels": [pseudo_cls],
        }) + "\n")
        written += 1
        if written % 2000 == 0:
            print("[splice] %d clips" % written, flush=True)

    man.close()
    print("[splice] wrote %d clips to %s" % (written, out))
    print("[splice] pretrain:  python -m src.train.train_query --config configs/query.yaml "
          "--data %s --out runs/pretrain --epochs 3" % out)


if __name__ == "__main__":
    main()
