"""Model output -> events, for the query (set-prediction) model.

Mirrors runner.py's split between raw candidates (so an ensemble can fuse
across checkpoints before any thresholding) and the final threshold step -
but there is no count head and no SoftNMS-as-selection here: the query
decoder already answers "which spans" via Hungarian-matched training.
SoftNMS below is only a safety net against near-duplicate queries converging
on the same event, not a selection mechanism.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from src.infer.decode import finalise, soft_nms_1d
from src.models.query_head import decode_queries

DEFAULT_POSTPROC = {
    "nms_sigma": 0.15,
    "score_floor": 0.05,
    "min_dur": 0.03,
    "max_out": 16,
    "score_scale": 1.0,     # per-district calibration multiplies this
}


@torch.no_grad()
def query_candidates(model, wav: torch.Tensor, frame_valid: torch.Tensor, fps: float,
                     durations: np.ndarray, pp: dict) -> List[dict]:
    outs = model(wav, frame_valid)
    spans, scores = decode_queries(outs)
    spans, scores = spans.float().cpu().numpy(), scores.float().cpu().numpy()
    res = []
    for i in range(spans.shape[0]):
        dur = float(durations[i])
        s, c = spans[i], scores[i]
        ok = (c > 1e-4) & (s[:, 1] > s[:, 0])
        s, c = soft_nms_1d(s[ok], c[ok], sigma=float(pp["nms_sigma"]),
                           max_out=int(pp["max_out"]))
        # `count` is carried only so this candidate dict has the same shape
        # runner.py's `fuse_candidates` expects; nothing here reads it back.
        res.append({"spans": s, "scores": c, "duration": dur, "count": np.zeros(1)})
    return res


def finalize_candidates(cand: dict, pp: dict | None = None) -> List[List[float]]:
    pp = {**DEFAULT_POSTPROC, **(pp or {}), **cand.get("pp_override", {})}
    s, c = cand["spans"], cand["scores"] * float(pp["score_scale"])
    keep = c > float(pp["score_floor"])
    return finalise(s[keep], c[keep], cand["duration"], min_dur=float(pp["min_dur"]))


@torch.no_grad()
def run_loader_query(model, loader, device, fps: float, pp: dict | None = None,
                     amp: bool = True) -> Dict[str, dict]:
    pp = {**DEFAULT_POSTPROC, **(pp or {})}
    model.eval()
    out_all: Dict[str, dict] = {}
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        fv = batch["frame_valid"].to(device, non_blocking=True)
        durations = batch["frame_valid"].sum(1).numpy() / fps
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            cands = query_candidates(model, wav, fv, fps, durations, pp)
        for uid, c in zip(batch["uid"], cands):
            out_all[uid] = c
    return out_all
