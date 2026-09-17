"""Generates a tiny CPU-only Kaggle notebook that does nothing but clone the
repo and run scripts/preflight.py with the real run's numbers - a pass/fail
in well under a minute, instead of finding out an hour into a GPU session.
"""
import json
from pathlib import Path

REPO = "raut7218/vaani-sed-query"


def cell(src, kind="code"):
    return {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {})}


CELLS = [
    cell("# Preflight-only check: does the real run's config actually fit "
         "this session's disk, and is the HF token reachable? CPU, no GPU, "
         "done in well under a minute.", "markdown"),
    cell('import subprocess\n'
         'r = subprocess.run(["git", "clone", "--depth", "1",\n'
         '                    "https://github.com/%s.git", "/kaggle/working/query"],\n'
         '                   capture_output=True, text=True)\n'
         'print(r.stdout[-2000:], r.stderr[-2000:])\n'
         'assert r.returncode == 0, "clone failed"\n' % REPO),
    cell('!pip -q install huggingface_hub 2>&1 | tail -3\n'),
    cell('import subprocess, os\n'
         'os.chdir("/kaggle/working/query")\n'
         'r = subprocess.run(["python", "scripts/preflight.py", "--work", "/kaggle/working",\n'
         '                    "--max-shards", "0", "--n-synthetic", "20000",\n'
         '                    "--n-pretrain", "60000", "--clip-len", "8"],\n'
         '                   capture_output=True, text=True)\n'
         'print(r.stdout)\n'
         'print(r.stderr)\n'
         'print("EXIT CODE:", r.returncode)\n'),

    cell("## Disk-speed check\n\n"
         "The actual bug wasn't disk exhaustion itself - it was that a "
         "checkpoint write got *slow* (not fast, not failing) once free "
         "space got close to the guard's margin, and that slowness silently "
         "blocked a DDP barrier for 48+ minutes with zero output. This "
         "times a real `save_atomic()` write of a realistically-sized "
         "checkpoint with exactly the guard's margin free, to confirm the "
         "margin is actually enough - not just enough by arithmetic.",
         "markdown"),
    cell('import shutil, sys, time, torch\n'
         'sys.path.insert(0, "/kaggle/working/query")\n'
         'from src.train.train import save_atomic\n'
         '\n'
         'WORK = "/kaggle/working"\n'
         'MARGIN_GB = 5.0\n'
         'free = shutil.disk_usage(WORK).free / 1024**3\n'
         'filler_gb = max(0.0, free - MARGIN_GB)\n'
         'filler = os.path.join(WORK, "_filler.bin")\n'
         'print("[disk-speed] %.1f GB free -> filling %.1f GB to leave the %.1f GB margin"\n'
         '      % (free, filler_gb, MARGIN_GB))\n'
         'with open(filler, "wb") as f:\n'
         '    f.truncate(int(filler_gb * 1024**3))\n'
         '\n'
         '# ~475 MB: matches state.pt+opt+scaler+ema for the real 23.7M-param model.\n'
         'dummy = {"model": {"w%d" % i: torch.zeros(2_000_000) for i in range(60)}}\n'
         't0 = time.time()\n'
         'save_atomic(dummy, os.path.join(WORK, "_dummy_ckpt.pt"))\n'
         'elapsed = time.time() - t0\n'
         'print("[disk-speed] wrote a ~475 MB checkpoint in %.1fs with %.1f GB free"\n'
         '      % (elapsed, shutil.disk_usage(WORK).free / 1024**3))\n'
         'if elapsed > 10:\n'
         '    print("[disk-speed] SLOW - this is the exact mechanism that hung a real run; "\n'
         '          "raise the guard margin further before trusting a real run.")\n'
         'else:\n'
         '    print("[disk-speed] OK - fast enough that this will not stall a DDP barrier.")\n'
         '\n'
         'os.remove(filler)\n'
         'os.remove(os.path.join(WORK, "_dummy_ckpt.pt"))\n'),
]

nb = {
    "cells": CELLS,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path(__file__).parent / "preflight_check.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
