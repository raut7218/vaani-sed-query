"""Training loop for the query (set-prediction) model.

Everything generic - DDP setup, EMA, staged unfreeze, checkpoint save/resume,
the cosine schedule, GPU augmentation, param groups - is imported straight
from `train.py` unchanged: none of it is specific to which head sits on top
of the shared frontend/FPN. Only the model builder, the loss, and the decode
path at validation time differ, because the query decoder answers "which
spans" directly and needs neither SoftNMS-driven count selection nor a count
head to get there - `runner_query.py` (shared with predict_query.py) is the
whole decode step.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (TierBatchSampler, VaaniSpanDataset, collate,  # noqa: E402
                              load_manifest, split_manifest)
from src.data.labels import LabelEncoder                                    # noqa: E402
from src.evaluation.metrics import evaluate                                 # noqa: E402
from src.infer.runner_query import finalize_candidates, run_loader_query    # noqa: E402
from src.models.encoders import build_encoder                               # noqa: E402
from src.models.span_model import build_query_model                        # noqa: E402
from src.train.set_losses import SetSpanLoss                               # noqa: E402
from src.train.train import (AsyncSaver, EMA, build_refs, cosine_lr,        # noqa: E402
                             gpu_augment, is_main, log, make_param_groups,
                             setup_ddp, should_eval, split_batch, to_host,
                             _unwrap)


def validate_query(net, loader, refs, device, fps, pp, amp, ddp) -> dict | None:
    cands = run_loader_query(net, loader, device, fps, pp, amp=amp)
    preds = {u: finalize_candidates(c, pp) for u, c in cands.items()}
    if ddp:
        parts = [None] * dist.get_world_size() if is_main() else None
        dist.gather_object(preds, parts, dst=0)
        if not is_main():
            return None
        preds = {}
        for p in parts:
            preds.update(p)
    return evaluate(preds, refs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/query.yaml")
    ap.add_argument("--data", default="data/vaani")
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--out", default="runs/query")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vad-dir", default="")
    ap.add_argument("--time-limit-h", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-from", default="",
                    help="warm-start weights (e.g. a splice-pretrain checkpoint); "
                         "unlike --resume, does not restore optimiser/step/epoch")
    ap.add_argument("--no-encoders", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.seed:
        cfg["seed"] = args.seed
    if args.no_encoders:
        cfg["model"]["encoders"] = []

    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    device, ddp, rank, world = setup_ddp(int(t.get("ddp_timeout_min", 60)))
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp = bool(t.get("amp", True)) and device.type == "cuda"

    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_run.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    le = LabelEncoder(expand_vehicle=bool(d.get("expand_vehicle", True)))
    recs = load_manifest(Path(args.data) / "manifest.jsonl")
    tr_recs, va_recs = split_manifest(recs, fold=args.fold,
                                      n_folds=int(d.get("n_folds", 5)), seed=int(cfg["seed"]))
    for extra in args.extra_data:
        er = load_manifest(Path(extra) / "manifest.jsonl")
        for r in er:
            r["_root"] = str(extra)
            r["pool"] = "synth"
        tr_recs += er
        log("[data] + %d extra clips from %s" % (len(er), extra))
    log("[data] %d train / %d val  (fold %d of %d)"
        % (len(tr_recs), len(va_recs), args.fold, int(d.get("n_folds", 5))))

    vad = args.vad_dir or d.get("vad_dir") or None
    ds_kw = dict(root=args.data, le=le, clip_len=float(d["clip_len"]),
                sr=int(d["sr"]), fps=float(d["fps"]), vad_dir=vad)
    tr_ds = VaaniSpanDataset(tr_recs, train=True, **ds_kw)
    va_ds = VaaniSpanDataset(va_recs[rank::world], train=False, **ds_kw)

    bs = int(t["batch_size"])
    ebs = int(t.get("eval_batch_size", 0)) or 2 * bs
    import os
    nw = max(0, min(int(t.get("num_workers", 4)), (os.cpu_count() or 2) // world))
    sampler = TierBatchSampler(tr_recs, bs, t.get("tier_quotas"), seed=cfg["seed"],
                               rank=rank, world_size=world,
                               steps_per_epoch=int(t.get("steps_per_epoch", 0)))
    gen = torch.Generator()
    gen.manual_seed(int(cfg["seed"]) + 1000 * rank)
    tr_ld = DataLoader(tr_ds, batch_sampler=sampler, num_workers=nw, collate_fn=collate,
                       pin_memory=True, generator=gen, persistent_workers=nw > 0,
                       prefetch_factor=int(t.get("prefetch_factor", 4)) if nw else None)
    va_ld = DataLoader(va_ds, batch_size=ebs, shuffle=False, num_workers=nw,
                       collate_fn=collate, pin_memory=True, persistent_workers=nw > 0,
                       prefetch_factor=int(t.get("prefetch_factor", 4)) if nw else None)

    enc = build_encoder(m, ckpt_dir=m.get("beats_dir", "checkpoints"))
    model = build_query_model(cfg, len(le), enc).to(device)

    def wrap_ddp(net):
        if not ddp:
            return net
        ddp_net = torch.nn.parallel.DistributedDataParallel(
            _unwrap(net), device_ids=[device.index],
            find_unused_parameters=bool(t.get("find_unused_parameters", False)),
            gradient_as_bucket_view=True,
            # The unused-parameter set (the VAD/speech head, see
            # find_unused_parameters above) never changes between
            # iterations, so DDP only needs to search for it once instead
            # of on every step - static_graph tells it that's safe.
            static_graph=bool(t.get("static_graph", True)))
        if bool(t.get("fp16_allreduce", True)):
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            ddp_net.register_comm_hook(None, default_hooks.fp16_compress_hook)
        return ddp_net

    if args.init_from:
        st = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(st["model"], strict=False)
        log("[init-from] %s (missing %d, unexpected %d)"
            % (args.init_from, len(missing), len(unexpected)))
        del st
        if is_main():
            # Once loaded, the pretrain run's checkpoints (~475MB: state.pt
            # + best.pt) serve no further purpose - fine-tune has its own
            # checkpoints from here. A real run measured only ~5GB free
            # after data generation; this run's own disk footprint growing
            # for hours (two checkpoint dirs, notebook output) ate that
            # margin and hit ENOSPC mid-run. Reclaiming this now, right as
            # the long fine-tune phase starts, is free and direct.
            pretrain_dir = Path(args.init_from).parent
            for f in ("state.pt", "best.pt"):
                (pretrain_dir / f).unlink(missing_ok=True)
            log("[init-from] freed %s" % pretrain_dir)

    crit = SetSpanLoss(cfg, int(m.get("n_bins", 16)))
    lr = float(t["lr"])
    enc_scale = float(t.get("encoder_lr_scale", 0.05))
    wd = float(t["weight_decay"])
    ema = EMA(model, float(t.get("ema_decay", 0.999)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs = int(t["epochs"])
    total_steps = epochs * len(sampler)
    warmup = int(t.get("warmup_steps", 500))
    step = 0
    history, best, start_epoch = [], -1.0, 1
    unfreeze_at = int(t.get("unfreeze_epoch", 999))
    n_unfreeze = int(t.get("unfreeze_blocks", 0))
    ckpt_blocks = bool(t.get("checkpoint_unfrozen", False))

    def unfreeze():
        got = _unwrap(model).encoder.unfreeze_last(n_unfreeze, ckpt_blocks)
        ema.refresh(model)
        return got

    st = None
    if args.resume:
        p = (out_dir / "state.pt") if args.resume == "auto" else Path(args.resume)
        if p.exists():
            st = torch.load(str(p), map_location="cpu", weights_only=False)
            log("[resume] %s" % p)
        elif args.resume != "auto":
            raise SystemExit("--resume %s does not exist" % args.resume)
    if st is not None and int(st.get("unfroze_at_epoch", 0)):
        unfreeze()
    model = wrap_ddp(model)
    opt = torch.optim.AdamW(make_param_groups(model, lr, enc_scale, wd))
    if st is not None:
        _unwrap(model).load_state_dict(st["model"])
        base_lrs = [g["base_lr"] for g in opt.param_groups]
        opt.load_state_dict(st["opt"])
        for g, b in zip(opt.param_groups, base_lrs):
            g["base_lr"] = b
        scaler.load_state_dict(st["scaler"])
        ema.shadow.load_state_dict(st["ema"])
        step, best = int(st["step"]), float(st["best"])
        old_total = int(st.get("total_steps", 0))
        if old_total and old_total != total_steps:
            step = int(round(step * total_steps / old_total))
        history = st.get("history", [])
        start_epoch = int(st["epoch"]) + 1
        log("[resume] -> epoch %d, step %d, best %.4f" % (start_epoch, step, best))
        del st

    refs = {}
    if is_main():
        ref_ld = DataLoader(
            VaaniSpanDataset(va_recs, train=False, labels_only=True, **ds_kw),
            batch_size=ebs, shuffle=False, num_workers=nw, collate_fn=collate)
        refs = build_refs(ref_ld, float(d["fps"]))

    clip_params = [p for g in opt.param_groups for p in g["params"]]
    accum = max(1, int(t.get("unfreeze_grad_accum", 1)))
    log_every = int(t.get("log_every", 100))
    saver = AsyncSaver()
    n_params = sum(p.numel() for p in clip_params)
    log("[train] %d steps/epoch/rank x %d rank(s), batch %d/GPU, %.1fM trainable params"
        % (len(sampler), world, bs, n_params / 1e6))

    t_start, recent = time.time(), []
    for epoch in range(start_epoch, epochs + 1):
        if args.time_limit_h and epoch > start_epoch:
            left = args.time_limit_h * 3600 - (time.time() - t_start)
            epoch_s = max(recent[-3:])
            if left < 1.1 * epoch_s:
                log("[time] %.0f min left - stopping at epoch %d; --resume auto continues"
                    % (left / 60, epoch - 1))
                break
        t_epoch = time.time()
        is_unfrozen = epoch > unfreeze_at and n_unfreeze > 0
        if epoch == unfreeze_at + 1 and n_unfreeze > 0:
            got = unfreeze()
            log("[train] unfroze top blocks per encoder: %s" % got)
            model = wrap_ddp(model)
            for g in make_param_groups(model, lr, enc_scale, wd)[2:]:
                opt.add_param_group(g)
            clip_params = [p for g in opt.param_groups for p in g["params"]]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

        model.train()
        acc, nb, t0 = {}, 0, time.time()
        t_wait, t_mark = 0.0, time.time()
        for batch in tr_ld:
            t_wait += time.time() - t_mark
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            batch["wav"] = gpu_augment(batch["wav"])
            f = cosine_lr(step, total_steps, warmup)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f

            opt.zero_grad(set_to_none=True)
            chunks = split_batch(batch, accum) if (is_unfrozen and accum > 1) else [batch]
            logs = {}
            for i, chunk in enumerate(chunks):
                w = chunk["wav"].size(0) / batch["wav"].size(0)
                sync = contextlib.nullcontext() if (not ddp or i == len(chunks) - 1) \
                    else model.no_sync()
                with sync:
                    with torch.autocast(device_type=device.type, enabled=amp):
                        outs = model(chunk["wav"], chunk["frame_valid"])
                        c_loss, c_logs = crit(outs, chunk)
                    scaler.scale(c_loss * w).backward()
                for k, v in c_logs.items():
                    logs[k] = logs.get(k, 0.0) + v.detach() * w
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(clip_params, float(t.get("grad_clip", 5.0)))
            scaler.step(opt)
            scaler.update()
            ema.update()
            step += 1
            nb += 1
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v
            if log_every and nb % log_every == 0:
                el = time.time() - t0
                log("  [%d/%d] step %d  loss %.4f  %.2f it/s  %.0f clips/s  mem %.1f GB"
                    % (epoch, epochs, nb, float(acc["loss"]) / nb, nb / el,
                       nb * bs * world / el,
                       torch.cuda.max_memory_allocated() / 2**30
                       if device.type == "cuda" else 0.0))
            if args.max_steps and nb >= args.max_steps:
                break
            t_mark = time.time()

        train_s = time.time() - t0
        msg = "  ".join("%s %.4f" % (k, float(v) / max(nb, 1)) for k, v in acc.items())
        log("[epoch %d/%d] %s" % (epoch, epochs, msg))
        log("[epoch %d/%d] train %.0fs (%.0f clips/s)"
            % (epoch, epochs, train_s, nb * bs * world / max(train_s, 1e-6)))

        jobs = []
        if should_eval(epoch, epochs, t):
            t1 = time.time()
            nets = [("ema", ema.shadow)]
            if bool(t.get("eval_raw", False)):
                nets.append(("raw", _unwrap(model)))
            for name, net in nets:
                r = validate_query(net, va_ld, refs, device, float(d["fps"]),
                                   cfg.get("postproc", {}), amp, ddp)
                if r is None:
                    continue
                r.update(epoch=epoch, which=name)
                history.append(r)
                log("   [%s] F1 %.4f  Dice %.4f  score %.4f  (tp %d fp %d fn %d)"
                    % (name, r["event_f1"], r["segment_dice"], r["score"],
                       r["tp"], r["fp"], r["fn"]))
                if r["score"] > best:
                    best = r["score"]
                    jobs.append((to_host(
                        {"model": net.state_dict(), "cfg": cfg, "classes": le.classes,
                         "score": best, "which": name, "epoch": epoch}),
                       out_dir / "best.pt"))
            log("   eval %.0fs" % (time.time() - t1))

        if is_main():
            (out_dir / "history.json").write_text(json.dumps(history, indent=1),
                                                  encoding="utf-8")
            jobs.append((to_host(
                {"model": _unwrap(model).state_dict(), "opt": opt.state_dict(),
                 "scaler": scaler.state_dict(), "ema": ema.shadow.state_dict(),
                 "step": step, "best": best, "total_steps": total_steps,
                 "epoch": epoch, "history": history, "cfg": cfg,
                 "unfroze_at_epoch": is_unfrozen}), out_dir / "state.pt"))
            saver.submit(jobs)

        if ddp:
            dist.barrier()
        epoch_s = time.time() - t_epoch
        if ddp:
            tt = torch.tensor([epoch_s], device=device)
            dist.broadcast(tt, 0)
            epoch_s = float(tt.item())
        recent.append(epoch_s)

    if is_main():
        saver.wait()
    log("[done] best val score %.4f -> %s" % (best, out_dir / "best.pt"))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
