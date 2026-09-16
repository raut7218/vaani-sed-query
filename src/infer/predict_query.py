"""Inference -> submission.zip, for the query (set-prediction) model.

Same submission contract as predict.py (v2's span model): every evaluation
clip appears exactly once, `[]` when nothing is detected, milliseconds
rounded, non-negative, non-decreasing. Per-district calibration is not wired
in yet for this model - `finalize_candidates`' `score_scale`/`pp_override`
hooks are there for it, but SoftNMS-safety-net decoding needs its own
validation first (see the design spec). Multiple checkpoints still fuse with
1D WBF, reusing runner.py's `fuse_candidates` unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.labels import LabelEncoder                                   # noqa: E402
from src.infer.predict import AudioDirDataset, collate, write_submission   # noqa: E402
from src.infer.runner import fuse_candidates                               # noqa: E402
from src.infer.runner_query import (DEFAULT_POSTPROC, finalize_candidates,  # noqa: E402
                                    run_loader_query)
from src.models.encoders import build_encoder                              # noqa: E402
from src.models.span_model import build_query_model                        # noqa: E402

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def load_checkpoint(path: Path, device):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    le = LabelEncoder(expand_vehicle=bool(cfg["data"].get("expand_vehicle", True)))
    enc = build_encoder(cfg["model"], ckpt_dir=cfg["model"].get("beats_dir", "checkpoints"))
    model = build_query_model(cfg, len(le), enc)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print("[predict] %s: %d missing keys (first %s)"
              % (path.name, len(missing), missing[:3]))
    return model.to(device).eval(), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one or more checkpoints; several are fused with 1D WBF")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--postproc", default="", help="JSON file of post-proc overrides")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pp = dict(DEFAULT_POSTPROC)
    if args.postproc:
        pp.update(json.loads(Path(args.postproc).read_text(encoding="utf-8")))

    files = sorted(p for p in Path(args.audio_dir).rglob("*")
                   if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit("no audio found under %s" % args.audio_dir)
    print("[predict] %d clips" % len(files))

    per_model: List[Dict[str, dict]] = []
    for ck in args.ckpt:
        model, cfg = load_checkpoint(Path(ck), device)
        d = cfg["data"]
        ds = AudioDirDataset(files, sr=int(d["sr"]), clip_len=float(d["clip_len"]),
                             fps=float(d["fps"]))
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=device.type == "cuda")
        cands = run_loader_query(model, ld, device, float(d["fps"]), pp)
        per_model.append(cands)
        print("[predict] %s done" % Path(ck).name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    uids = sorted(per_model[0])
    if len(per_model) == 1:
        fused = per_model[0]
    else:
        fused = {u: fuse_candidates([m[u] for m in per_model if u in m], pp)
                 for u in uids}

    preds = {u: finalize_candidates(fused[u], pp) for u in uids}
    write_submission(preds, Path(args.out))

    n = [len(v) for v in preds.values()]
    print("[predict] events/clip %.2f | empty %.1f%%"
          % (float(np.mean(n)), 100.0 * float(np.mean([x == 0 for x in n]))))


if __name__ == "__main__":
    main()
