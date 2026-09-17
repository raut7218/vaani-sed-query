"""Generates notebooks/Vaani_Track1_Query_Kaggle.ipynb from the cell source below.

Run locally (`python notebooks/build_query_notebook.py`) whenever a cell needs
to change - the notebook itself is checked in as the built artifact, this
script is the editable source, so a diff review sees the actual cell text
instead of raw .ipynb JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "Vaani_Track1_Query_Kaggle.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
           "outputs": [], "source": src.splitlines(keepends=True)}


CELLS = [
md("""# Vaani Track 1 - query (set-prediction) model, on Kaggle

Replaces the fused v2-span + TRACE pipeline's SoftNMS + count head + semi-Markov
decoder with a single end-to-end DETR/TadTR-style set-prediction decoder on the
same dual-SSL frontend, plus label-free splice-boundary pretraining on the whole
154.6 h corpus. See `docs/superpowers/specs/` for the design rationale.

**`SMOKE_TEST = True`** (the default below) runs every stage at toy scale - a
few hundred clips, a handful of steps, one epoch - so the *whole* pipeline
(data, pretrain, fine-tune, predict, submission validation) is proven to run
start-to-finish on this Kaggle image before any real GPU-hours are spent. Flip
it to `False` only after a smoke run has finished clean.
"""),

code("""# ============================ HF TOKEN ================================
# Leave "" to use the HF_TOKEN Kaggle secret (Add-ons -> Secrets) instead -
# never paste a real token into a notebook that gets pushed to a repo.
HF_TOKEN = \"\"
"""),

code('''# ============================== CONFIG ==============================
# The only cell you need to edit.

REPO = "raut7218/vaani-sed-query"
REPO_IS_PRIVATE = False

SMOKE_TEST = True     # see the markdown above - flip to False for the real run

# --- data ---------------------------------------------------------------
DATA_FROM  = ""               # e.g. "/kaggle/input/vaani-prepared" to skip the download
MAX_SHARDS = 2 if SMOKE_TEST else 0     # 0 = all 182 shards (~16.5 GB)
DATA_LIMIT = 600 if SMOKE_TEST else 0   # debug: stop the download after N clips

# --- self-supervised pretraining (splice-boundary, full corpus) ---------
USE_PRETRAIN   = True
N_PRETRAIN     = 300 if SMOKE_TEST else 60000
PRETRAIN_STEPS = 20 if SMOKE_TEST else 0     # 0 = a full epoch (--max-steps 0)
PRETRAIN_EPOCHS = 1 if SMOKE_TEST else 3
PRETRAIN_TIME_LIMIT_H = 0.0 if SMOKE_TEST else 2.5   # stop cleanly, leave state.pt, well inside Kaggle's ~12h cap

# --- fine-tune ------------------------------------------------------------
FOLD        = 0
EPOCHS      = 1 if SMOKE_TEST else 20
BATCH_SIZE  = 4 if SMOKE_TEST else 16       # PER GPU
MAX_STEPS   = 10 if SMOKE_TEST else 0       # 0 = a full epoch
TIME_LIMIT_H = 0.0 if SMOKE_TEST else 8.5    # same idea; re-run with RESUME_FROM set to continue
RESUME_FROM = ""

USE_VAD       = True
USE_SYNTHETIC = True
N_SYNTHETIC   = 300 if SMOKE_TEST else 20000

# --- inference ------------------------------------------------------------
TEST_AUDIO_DIR = ""           # run the "find eval audio" cell, paste the path here
ENSEMBLE_CKPTS = []

# =====================================================================
from pathlib import Path

WORK  = "/kaggle/working"

if not DATA_FROM:
    for pattern in ("*/manifest.jsonl", "*/data/manifest.jsonl"):
        hits = sorted(Path("/kaggle/input").glob(pattern))
        if hits:
            DATA_FROM = str(hits[0].parent)
            print("[data] auto-detected a prepared corpus at %s - skipping the download"
                  % DATA_FROM)
            break

DATA     = DATA_FROM or (WORK + "/data")
SYNTH    = WORK + "/synth"
PRETRAIN = WORK + "/pretrain_data"
RUN      = WORK + "/runs/f%d" % FOLD
PRETRAIN_RUN = WORK + "/runs/pretrain"

if SMOKE_TEST and not TEST_AUDIO_DIR:
    # Smoke-only: no real held-out test set is attached to this kernel, so
    # reuse the just-downloaded training audio to exercise predict_query.py
    # and the submission-schema check end-to-end. Never do this for the real
    # run - TEST_AUDIO_DIR must point at the actual competition test set.
    TEST_AUDIO_DIR = DATA + "/audio"

print("data:", DATA, "| run:", RUN, "| smoke test:", SMOKE_TEST)
'''),

md("## 1. Fetch the repo, and the credentials"),

code('''import os, subprocess, shutil
from pathlib import Path


def kaggle_secret(label):
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret(label) or None
    except Exception as e:                # noqa: BLE001
        print("[secrets] %s lookup failed: %s: %s" % (label, type(e).__name__, str(e)[:200]))
        return None


SRC = WORK + "/query"
shutil.rmtree(SRC, ignore_errors=True)

url = "https://github.com/%s.git" % REPO
if REPO_IS_PRIVATE:
    tok = kaggle_secret("GITHUB_TOKEN")
    if not tok:
        raise RuntimeError("REPO_IS_PRIVATE is True but no GITHUB_TOKEN secret is attached.")
    url = "https://x-access-token:%s@github.com/%s.git" % (tok, REPO)

r = subprocess.run(["git", "clone", "--depth", "1", url, SRC], capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit("clone failed. Check: Internet is On, and the repo/token are right.")
subprocess.run(["git", "-C", SRC, "remote", "set-url", "origin",
                "https://github.com/%s.git" % REPO], check=True)

os.chdir(SRC)
print("cloned to", SRC)

_HF_PASTED = bool(HF_TOKEN)
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN") or kaggle_secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF_TOKEN: found (%s)" % ("pasted" if _HF_PASTED else "Kaggle secret"))
elif DATA_FROM:
    print("HF_TOKEN: missing, but DATA_FROM is set - only encoder downloads are rate-limited.")
else:
    raise RuntimeError(
        "No HF_TOKEN and DATA_FROM is empty, so this session has no route to the audio.\\n"
        "Add-ons -> Secrets -> add HF_TOKEN, tick this notebook, re-run.\\n"
        "Already downloaded the corpus earlier? Set DATA_FROM in CONFIG instead.")
'''),

code('!pip -q install -r requirements.txt 2>&1 | tail -2\n'),

md("## 2. Verify the wiring before spending GPU hours\n\n"
   "CPU, seconds: shape/finiteness checks, then the overfit proof that the "
   "query head is time-aligned to the audio (mirrors `test_overfit.py` for "
   "the trident head). Then a disk/credentials preflight - four separate "
   "real runs of this pipeline each burned close to an hour before failing "
   "on a missing HF_TOKEN or disk exhaustion; this catches both in seconds, "
   "before anything downloads."),

code('!python tests/test_query_components.py\n'
    '!python tests/test_query_overfit.py 2>&1 | tail -6\n'),

code('get_ipython().system(\n'
    '    "python scripts/preflight.py --work %s --data-from \'%s\' "\n'
    '    "--max-shards %d --n-synthetic %d --n-pretrain %d --clip-len %g"\n'
    '    % (WORK, DATA_FROM, MAX_SHARDS, N_SYNTHETIC if USE_SYNTHETIC else 0,\n'
    '       N_PRETRAIN if USE_PRETRAIN else 0, 8.0))\n'
    'rc = int(get_ipython().user_ns.get("_exit_code", 0) or 0)\n'
    'if rc:\n'
    '    raise RuntimeError("preflight failed - see the message above; fix it "\n'
    '                       "before this burns another GPU session.")\n'),

md("## 3. Encoders"),

code('!python scripts/fetch_encoders.py --all\n'
    '!ls -la checkpoints/ 2>/dev/null || echo "no checkpoints dir"\n'),

md("## 4. Config"),

code('''import yaml, pathlib, torch
cfg = yaml.safe_load(open("configs/query.yaml"))
cfg["data"]["vad_dir"]      = (DATA + "/vad") if USE_VAD else ""
cfg["model"]["beats_dir"]   = SRC + "/checkpoints"
cfg["train"]["num_workers"] = 2
pathlib.Path("configs/kaggle_query.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(yaml.safe_dump({k: cfg[k] for k in ("data", "model")}, sort_keys=False))
'''),

md("## 5. Data"),

code('''import json, collections

if DATA_FROM:
    print("using prepared corpus at", DATA_FROM)
else:
    man_path = Path(DATA) / "manifest.jsonl"
    if man_path.exists():
        print("[data] %s already has a manifest from earlier this session - skipping" % DATA)
    else:
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("Vaani corpus is gated and no HF_TOKEN is available - see cell 1.")
        flags = "--max-shards %d" % MAX_SHARDS if MAX_SHARDS else ""
        if DATA_LIMIT:
            flags += " --limit %d" % DATA_LIMIT
        get_ipython().system("python scripts/download_data.py --out %s %s" % (DATA, flags))

man = Path(DATA) / "manifest.jsonl"
if not man.exists():
    raise RuntimeError("no manifest at %s - the download above did not finish." % man)
recs = [json.loads(l) for l in man.open(encoding="utf-8") if l.strip()]
print(len(recs), "clips |", collections.Counter(r["tier"] for r in recs))
'''),

md("## 6. Free supervision: VAD, bronze splicing, and full-corpus splice pretraining\n\n"
   "`make_splice_pretrain.py` is the new one: any two clips in the corpus, cut "
   "and pasted at a known point, teach boundary precision at a scale the 20 h "
   "gold tier alone cannot reach - see the design spec for why."),

code('''if USE_VAD and not DATA_FROM:
    get_ipython().system("python scripts/make_vad.py --data %s" % DATA)
if USE_SYNTHETIC:
    get_ipython().system("python scripts/make_synthetic.py --data %s --out %s -n %d"
                         % (DATA, SYNTH, N_SYNTHETIC))
if USE_PRETRAIN:
    get_ipython().system("python scripts/make_splice_pretrain.py --data %s --out %s -n %d"
                         % (DATA, PRETRAIN, N_PRETRAIN))
'''),

md("## 7. Pretrain (splice-boundary self-supervision)\n\n"
   "Trains the shared frontend + FPN + query head on nothing but manufactured "
   "splices, so real fine-tuning (next cell) starts from weights that already "
   "know how to localise a boundary precisely."),

code('''if USE_PRETRAIN:
    max_steps = "--max-steps %d" % PRETRAIN_STEPS if PRETRAIN_STEPS else ""
    time_limit = "--time-limit-h %g" % PRETRAIN_TIME_LIMIT_H if PRETRAIN_TIME_LIMIT_H else ""
    get_ipython().system(
        "torchrun --standalone --nproc_per_node=%d -m src.train.train_query "
        "--config configs/kaggle_query.yaml --data %s --out %s "
        "--epochs %d --batch-size %d %s %s"
        % (max(1, torch.cuda.device_count()), PRETRAIN, PRETRAIN_RUN,
           PRETRAIN_EPOCHS, BATCH_SIZE, max_steps, time_limit))
    rc = int(get_ipython().user_ns.get("_exit_code", 0) or 0)
    if rc:
        raise RuntimeError("splice pretraining exited with code %d - see the traceback above." % rc)
'''),

md("## 8. Fine-tune"),

code('''extra = ("--extra-data " + SYNTH) if USE_SYNTHETIC else ""
init_from = ""
resume = ""
if RESUME_FROM:
    Path(RUN).mkdir(parents=True, exist_ok=True)
    for f in ("state.pt", "best.pt", "history.json"):
        s = Path(RESUME_FROM) / f
        if s.exists():
            shutil.copy(s, Path(RUN) / f)
    resume = "--resume auto"
elif USE_PRETRAIN and Path(PRETRAIN_RUN, "state.pt").exists():
    init_from = "--init-from %s/state.pt" % PRETRAIN_RUN

max_steps = "--max-steps %d" % MAX_STEPS if MAX_STEPS else ""
time_limit = "--time-limit-h %g" % TIME_LIMIT_H if TIME_LIMIT_H else ""
get_ipython().system(
    "torchrun --standalone --nproc_per_node=%d -m src.train.train_query "
    "--config configs/kaggle_query.yaml --data %s %s --out %s --fold %d "
    "--epochs %d --batch-size %d %s %s %s %s"
    % (max(1, torch.cuda.device_count()), DATA, extra, RUN, FOLD,
       EPOCHS, BATCH_SIZE, resume, init_from, max_steps, time_limit))

rc = int(get_ipython().user_ns.get("_exit_code", 0) or 0)
if rc:
    raise RuntimeError(
        "training exited with code %d. Read the traceback above - under torchrun "
        "every rank prints its own copy, so read the *first* traceback block." % rc)
'''),

md("### If the session times out anyway\n\n"
   "Kaggle sessions cap at ~12 h. Both training cells above already pass "
   "`--time-limit-h` (`PRETRAIN_TIME_LIMIT_H` / `TIME_LIMIT_H` in CONFIG) for "
   "the real run, so they stop cleanly after the last epoch that fits and "
   "leave `state.pt` behind. If a session still gets killed mid-epoch (OOM, "
   "quota, manual stop), set `RESUME_FROM` to that run's output directory "
   "and re-run this cell - it copies the checkpoint in and passes "
   "`--resume auto`, the same as the span model's notebook."),

md("## 9. Find the evaluation audio"),

code('''for p in sorted(Path("/kaggle/input").glob("*")):
    audio = list(p.rglob("*.wav")) + list(p.rglob("*.flac"))
    print("%7d audio files   %s" % (len(audio), p))
'''),

md("## 10. Submit"),

code('''ckpts = [c for c in (ENSEMBLE_CKPTS or [RUN + "/best.pt"]) if Path(c).exists()]
if not TEST_AUDIO_DIR:
    print("SKIPPED: set TEST_AUDIO_DIR in CONFIG (see the listing above) and re-run.")
elif not ckpts:
    print("SKIPPED: no checkpoint at %s - train first." % (RUN + "/best.pt"))
else:
    get_ipython().system(
        "python -m src.infer.predict_query --ckpt %s --audio-dir %s --out %s/submission.zip "
        "--batch-size 16 --num-workers 2" % (" ".join(ckpts), TEST_AUDIO_DIR, WORK))
'''),

code('''import zipfile, json, numpy as np
if not Path(WORK + "/submission.zip").exists():
    print("no submission.zip - the cell above was skipped.")
else:
    with zipfile.ZipFile(WORK + "/submission.zip") as z:
        assert z.namelist() == ["predictions.jsonl"], z.namelist()
        rows = [json.loads(l) for l in
                z.read("predictions.jsonl").decode().splitlines() if l.strip()]
    ids = [r["clip_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate clip_id"
    for r in rows:
        for e in r["events"]:
            assert e["onset"] >= 0 and e["offset"] >= e["onset"], r["clip_id"]
    print("%d clips | %d events | %.2f events/clip | %.1f%% empty"
          % (len(rows), sum(len(r["events"]) for r in rows),
             np.mean([len(r["events"]) for r in rows]),
             100 * np.mean([not r["events"] for r in rows])))
    print(rows[0])
    print("\\nSMOKE_TEST =", SMOKE_TEST,
          "- if True and this printed cleanly, flip it to False and Save & Run All for real.")
'''),
]

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT, "(%d cells)" % len(CELLS))
