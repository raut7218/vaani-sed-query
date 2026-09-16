"""Fast checks for the query decoder and its loss. `python tests/test_query_components.py`

CPU, seconds, no audio download - the query-model analogue of
test_components.py's shape/finiteness checks. test_query_overfit.py is the
slower, more meaningful proof that boundaries are actually learnable; this
file exists so a broken shape or a NaN is caught before spending minutes on
that one instead of the diagnosis it deserves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import MAX_EVENTS                                # noqa: E402
from src.models.query_head import SpanQueryDecoder, decode_queries     # noqa: E402
from src.models.span_model import build_query_model                    # noqa: E402
from src.train.set_losses import SetSpanLoss, hungarian_match           # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("query decoder forward shapes")
def _():
    torch.manual_seed(0)
    dec = SpanQueryDecoder(d_model=32, n_queries=6, n_bins=8, n_stages=3, n_head=4,
                           clip_len=8.0, coarse_delta=4.0, refine_delta=0.4)
    mem = torch.randn(2, 20, 32)
    mask = torch.ones(2, 20)
    mask[1, 15:] = 0.0
    outs = dec(mem, mask)
    assert len(outs) == 3
    for o in outs:
        assert o["span"].shape == (2, 6, 2)
        assert o["presence"].shape == (2, 6)
        assert o["start_logits"].shape == (2, 6, 8)
        assert torch.isfinite(o["span"]).all()
    # Refinement narrows: later stages should sit closer to the coarse guess
    # than the coarse delta allows a first guess to be wrong by.
    assert outs[-1]["delta_max"] < outs[0]["delta_max"]


@check("decode_queries returns thresholdable (span, prob)")
def _():
    torch.manual_seed(1)
    dec = SpanQueryDecoder(d_model=16, n_queries=4, n_bins=6, n_stages=2, n_head=2,
                           clip_len=4.0)
    outs = dec(torch.randn(1, 10, 16), torch.ones(1, 10))
    spans, probs = decode_queries(outs)
    assert spans.shape == (1, 4, 2)
    assert probs.shape == (1, 4)
    assert (probs >= 0).all() and (probs <= 1).all()


@check("hungarian_match assigns the geometrically closest query")
def _():
    # Two queries, two targets, one clip: query 0 is near target 1 and query 1
    # is near target 0, so the optimal (min-cost) assignment must swap them -
    # a bug that assigned by index instead of by cost would fail this.
    spans = torch.tensor([[[5.0, 5.5], [1.0, 1.5]]])          # (1, 2, 2)
    presence = torch.zeros(1, 2)
    tgt = torch.tensor([[[0.9, 1.4], [4.9, 5.4]]])            # (1, 2, 2)
    valid = torch.tensor([[True, True]])
    matches = hungarian_match(spans, presence, tgt, valid)
    qi, ti = matches[0]
    order = {int(q): int(t) for q, t in zip(qi.tolist(), ti.tolist())}
    assert order[0] == 1 and order[1] == 0


@check("hungarian_match handles a clip with zero events")
def _():
    spans = torch.randn(1, 3, 2)
    presence = torch.zeros(1, 3)
    tgt = torch.zeros(1, 2, 2)
    valid = torch.zeros(1, 2, dtype=torch.bool)
    qi, ti = hungarian_match(spans, presence, tgt, valid)[0]
    assert qi.numel() == 0 and ti.numel() == 0


@check("SetSpanLoss is finite, including a zero-event clip in the batch")
def _():
    torch.manual_seed(2)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs"
                          / "query.yaml").read_text(encoding="utf-8"))
    cfg["model"].update(encoders=[], d_model=32, n_levels=2, n_base_layers=1,
                        dropout=0.0, specaug=False)
    cfg["model"]["query"].update(n_queries=4, n_stages=2)
    cfg["data"]["clip_len"] = 4.0
    n_frames = int(4.0 * cfg["data"]["fps"])
    model = build_query_model(cfg, 3, None)
    crit = SetSpanLoss(cfg, int(cfg["model"]["n_bins"]))

    B = 3
    spans = torch.full((B, MAX_EVENTS, 2), -1.0)
    cls = torch.full((B, MAX_EVENTS), -1, dtype=torch.long)
    spans[0, 0] = torch.tensor([10.0, 30.0])       # base frames, clip 0 has an event
    cls[0, 0] = 0
    # clip 1 and 2 have none - exercises the n_pos==0-in-a-microbatch path.
    batch = {
        "wav": torch.randn(B, int(4.0 * 16000)) * 0.01,
        "spans": spans, "span_cls": cls,
        "frame_target": torch.zeros(B, n_frames, 3),
        "clip_target": torch.zeros(B, 3),
        "frame_valid": torch.ones(B, n_frames),
        "speech_target": torch.zeros(B, n_frames),
        "has_vad": torch.zeros(B),
        "tier": torch.zeros(B, dtype=torch.long),
    }
    outs = model(batch["wav"], batch["frame_valid"])
    loss, logs = crit(outs, batch)
    assert torch.isfinite(loss)
    for k, v in logs.items():
        assert torch.isfinite(v), k
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def main() -> None:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:                                       # noqa: BLE001
            failed += 1
            print("FAIL  %s: %r" % (name, e))
    print("\n%d/%d checks passed" % (len(CHECKS) - failed, len(CHECKS)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
