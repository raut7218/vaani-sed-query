# Vaani Track 1 — a set-prediction decoder, on the v2 frontend

**Competition:** [Datathon@IndoML 2026 — Track 1](https://www.codabench.org/competitions/17825/).
Scored on **Event F1 + Segment Dice** (max 2.0) over an 11 h withheld test set
(3,437 natural clips + 2,080 synthetic). Leaderboard top is currently **1.6**
(F1 0.78 / Dice 0.82); this repo's prior architecture (the fused
v2-span + TRACE pipeline in [`vaani-sed-v2`](https://github.com/raut7218/vaani-sed-v2))
scored **1.19**.

## Why a new head, not another tuning pass

The v2 README's own measurements, over one fixed candidate pool:

| | score |
|---|---|
| as decoded (SoftNMS + count head) | 1.1753 |
| oracle **count** selection, same pool | 1.2123 (+0.037) |
| oracle **subset** selection, same pool | **1.4569 (+0.28)** |

The span proposals are already good. Picking the *right ones* — not the right
*number* — is where 90% of the reachable headroom sits, and SoftNMS + a count
head + (in the later TRACE variant) a semi-Markov decoder fused by rerank are
all post-hoc approximations of that question, never trained against it
directly.

## What changed

**Kept, unmodified:** the ATST-Frame + BEATs dual-SSL frontend, spectral-flux
channels, `TemporalFPN`, staged encoder unfreezing, tier-weighted losses, the
20 ms boundary branch, per-district calibration hooks. None of this was ever
diagnosed as the bottleneck.

**Replaced:** `TridentHead`'s dense per-point regression + SoftNMS + count
head + the separate TRACE model + its HSMM decoder + the fuse/rerank glue —
all of it — with a single **DETR/TadTR-style sparse set-prediction decoder**
(`src/models/query_head.py`). A fixed 24 learned queries cross-attend to the
FPN's finest level; each query directly predicts *(presence, onset
distribution, offset distribution)*. Training uses Hungarian bipartite
matching (`src/train/set_losses.py`), so the loss *is* "did you output the
right subset" — not a proxy for it. No NMS-as-selection, no count head:
thresholding presence on the queries above threshold is the whole decode
step (`src/infer/runner_query.py`).

Two refinements on top of vanilla DETR, both standard and low-risk relative
to a full diffusion-style detector:

* **Iterative span refinement** (Deformable DETR / Sparse R-CNN): 3 decoder
  stages, each correcting the previous stage's (onset, offset) rather than
  predicting from scratch — the mechanism that buys the last mile of
  boundary precision.
* **Spread, learned per-query anchors** (DAB-DETR / Anchor-DETR): each
  query's initial reference span is centred at a distinct, evenly-spaced
  point across the clip at initialisation, not derived identically for every
  query through a shared projection. Without this, `tests/test_query_overfit.py`
  fails with every clip decoding to the same "average" span regardless of
  its actual content — a textbook DETR symmetry-collapse, confirmed and
  fixed during development, not a hypothetical.
* **DFL boundary distributions carried over unchanged** from `TridentHead`
  (16-bin distribution per boundary, expectation = the value) — proven to
  place boundaries to ~5 ms on off-grid targets; only what the bins express
  (a delta from the current stage's estimate, not a distance from a grid
  point) is new.

**New pretraining pillar**, targeting the *other* measured gap — the head
places boundaries to 4.2 ms when overfit to a handful of clips and carries
~190 ms of jitter once trained for real; that is a generalisation gap, not a
capacity one. `scripts/make_splice_pretrain.py` extends the existing
cut-paste synthetic-labelling trick (`make_synthetic.py`, previously limited
to ~30 h of single-tag bronze donors) to the **entire 154.6 h corpus**: any
two clips, spliced at a known point with a randomised crossfade width, are
free, exact boundary supervision at a scale no amount of the 20 h gold tier
can match. `src/train/train_query.py --init-from` warm-starts fine-tuning
from a splice-pretrained checkpoint.

## Layout (new/changed files only — everything else is `vaani-sed-v2` unchanged)

```
src/models/query_head.py        SpanQueryDecoder: the set-prediction head
src/models/span_model.py        + VaaniQueryModel / build_query_model
src/train/set_losses.py         Hungarian matching + SetSpanLoss
src/train/train_query.py        training loop (reuses train.py's DDP/EMA/resume)
src/infer/runner_query.py       decode: presence threshold + SoftNMS safety net
src/infer/predict_query.py      -> submission.zip
scripts/make_splice_pretrain.py full-corpus splice-boundary pretext data
configs/query.yaml               query-model hyperparameters
tests/test_query_components.py  shape/finiteness checks (CPU, seconds)
tests/test_query_overfit.py     boundary-precision proof (CPU, ~1 min)
notebooks/build_query_notebook.py   generates the Kaggle notebook (editable source)
notebooks/Vaani_Track1_Query_Kaggle.ipynb   the notebook itself
```

## Status

Local CPU smoke tests (`tests/test_query_components.py`,
`tests/test_query_overfit.py`) verified before any GPU time was spent — see
commit history. The Kaggle notebook runs a small-scale `SMOKE_TEST = True`
pass through the *entire* pipeline (data, splice-pretrain, fine-tune,
predict, submission validation) before the real run.
