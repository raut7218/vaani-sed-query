"""Proves the query head is time-aligned to the audio. `python tests/test_query_overfit.py`

Same purpose as test_overfit.py, for the set-prediction head: overfit a
handful of clips with tones at known, deliberately off-grid times and assert
the decoded events land on them. If this fails, the bug is in query_head.py,
set_losses.py, or the Hungarian assignment - not in the shared frontend/FPN,
which test_overfit.py already covers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import MAX_EVENTS                                # noqa: E402
from src.evaluation.metrics import evaluate                            # noqa: E402
from src.infer.decode import finalise, soft_nms_1d                     # noqa: E402
from src.models.query_head import decode_queries                       # noqa: E402
from src.models.span_model import build_query_model                    # noqa: E402
from src.train.set_losses import SetSpanLoss                           # noqa: E402

SR, FPS, CLIP = 16000, 25.0, 6.0
N_FRAMES = int(CLIP * FPS)


def make_batch(n=8, seed=0):
    rng = np.random.default_rng(seed)
    wav = rng.normal(0, 0.005, (n, int(CLIP * SR))).astype("float32")
    spans = np.full((n, MAX_EVENTS, 2), -1.0, "float32")
    cls = np.full((n, MAX_EVENTS), -1, "int64")
    frame = np.zeros((n, N_FRAMES, 4), "float32")
    refs = {}
    for i in range(n):
        t0 = 0.4 + 0.31 * i
        ln = 0.7 + 0.03 * i
        a, b = int(t0 * SR), int((t0 + ln) * SR)
        t = np.arange(b - a) / SR
        wav[i, a:b] += (0.5 * np.sin(2 * np.pi * 900 * t) * np.hanning(len(t))).astype("float32")
        spans[i, 0] = (t0 * FPS, (t0 + ln) * FPS)
        cls[i, 0] = 0
        frame[i, int(t0 * FPS):int((t0 + ln) * FPS), 0] = 1.0
        refs["c%d" % i] = [[t0, t0 + ln]]
    return {
        "wav": torch.from_numpy(wav),
        "spans": torch.from_numpy(spans),
        "span_cls": torch.from_numpy(cls),
        "frame_target": torch.from_numpy(frame),
        "clip_target": torch.from_numpy(frame.max(1)),
        "frame_valid": torch.ones(n, N_FRAMES),
        "speech_target": torch.zeros(n, N_FRAMES),
        "has_vad": torch.zeros(n),
        "tier": torch.zeros(n, dtype=torch.long),
        "uid": ["c%d" % i for i in range(n)],
    }, refs


def main() -> None:
    torch.manual_seed(0)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs"
                          / "query.yaml").read_text(encoding="utf-8"))
    cfg["model"].update(encoders=[], d_model=128, n_levels=2, n_base_layers=1,
                        dropout=0.0, specaug=False)
    cfg["model"]["query"].update(n_queries=2, n_stages=2, coarse_delta=3.0, refine_delta=0.3)
    cfg["data"]["clip_len"] = CLIP
    model = build_query_model(cfg, 4, None)
    crit = SetSpanLoss(cfg, int(cfg["model"]["n_bins"]))
    batch, refs = make_batch()

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.train()
    for step in range(1200):
        outs = model(batch["wav"], batch["frame_valid"])
        loss, logs = crit(outs, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 200 == 0:
            print("step %3d  loss %.4f  presence %.4f  reg %.4f  dfl %.4f"
                  % (step, float(logs["loss"]), float(logs["presence"]),
                     float(logs["reg"]), float(logs["dfl"])))

    model.eval()
    with torch.no_grad():
        outs = model(batch["wav"], batch["frame_valid"])
    spans, scores = decode_queries(outs)
    spans, scores = spans.numpy(), scores.numpy()

    preds = {}
    for i, uid in enumerate(batch["uid"]):
        s, c = spans[i], scores[i]
        ok = (c > 1e-4) & (s[:, 1] > s[:, 0])
        s, c = soft_nms_1d(s[ok], c[ok], sigma=0.15, max_out=16)
        keep = c > 0.5
        preds[uid] = finalise(s[keep], c[keep], CLIP, min_dur=0.03)

    r = evaluate(preds, refs)
    print("\nrefs :", {k: [[round(x, 3) for x in e] for e in v] for k, v in list(refs.items())[:3]})
    print("preds:", {k: preds[k] for k in list(preds)[:3]})
    print("\nF1 %.4f  Dice %.4f  score %.4f" % (r["event_f1"], r["segment_dice"], r["score"]))

    errs = []
    for u, ref in refs.items():
        if preds[u]:
            errs.append(abs(preds[u][0][0] - ref[0][0]))
            errs.append(abs(preds[u][0][1] - ref[0][1]))
    if errs:
        print("mean absolute boundary error: %.1f ms" % (1000 * float(np.mean(errs))))

    ok = r["event_f1"] >= 0.99 and r["segment_dice"] >= 0.9
    print("\n%s: overfit F1 %.3f, Dice %.3f" % ("PASS" if ok else "FAIL",
                                                r["event_f1"], r["segment_dice"]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
