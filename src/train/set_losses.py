"""Hungarian-matched losses for the sparse span decoder (query_head.py).

Every boundary/quality term below is the *same* function TridentHead's loss
uses (losses.py): 1D DIoU, Distribution Focal Loss, sigmoid focal, soft Dice.
What is different is the assignment - `assign_targets` puts one match per
pyramid *point* by geometry; `hungarian_match` puts at most one match per
*query* by an end-to-end cost (presence + L1 + 1D GIoU), which is what set
prediction is trained on and what removes the ranking heuristics entirely.

The auxiliary terms (frame BCE, soft-Dice on the agnostic mask, clip BCE,
speech BCE, the 20 ms boundary map) are computed from the same base-level
heads `VaaniQueryModel` keeps unchanged from `VaaniSpanModel` - deliberately
duplicated from `SpanLoss` rather than shared, so this file has no import-time
dependency on the trident-specific loss class and the two can be edited
independently.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.train.losses import (boundary_targets, diou_1d, distribution_focal,
                              sigmoid_focal, soft_dice)

TIER_GOLD = 0
TIER_BRONZE = 2


@torch.no_grad()
def hungarian_match(spans: torch.Tensor, presence: torch.Tensor,
                    tgt_spans: torch.Tensor, tgt_valid: torch.Tensor,
                    cost_presence: float = 1.0, cost_l1: float = 2.0,
                    cost_iou: float = 2.0) -> List[tuple]:
    """Per-clip optimal query<->event assignment.

    spans: (B, N, 2) sec. presence: (B, N) logits. tgt_spans: (B, M, 2) sec,
    padded. tgt_valid: (B, M) bool. Returns one (query_idx, tgt_idx) pair of
    int64 tensors per clip; `tgt_idx` indexes only the valid (masked) events.
    """
    B, N, _ = spans.shape
    prob = presence.sigmoid()
    out: List[tuple] = []
    for b in range(B):
        m = tgt_valid[b]
        M = int(m.sum().item())
        if M == 0:
            out.append((spans.new_zeros(0, dtype=torch.long),
                       spans.new_zeros(0, dtype=torch.long)))
            continue
        ts = tgt_spans[b, m]                                      # (M, 2)
        ps = spans[b]                                             # (N, 2)
        l1 = (ps[:, None, :] - ts[None, :, :]).abs().sum(-1)       # (N, M)
        s1, e1 = ps[:, None, 0], ps[:, None, 1]
        s2, e2 = ts[None, :, 0], ts[None, :, 1]
        inter = (torch.minimum(e1, e2) - torch.maximum(s1, s2)).clamp(min=0)
        union = (e1 - s1) + (e2 - s2) - inter
        iou = inter / union.clamp(min=1e-6)
        enclose = (torch.maximum(e1, e2) - torch.minimum(s1, s2)).clamp(min=1e-6)
        giou = iou - (enclose - union) / enclose                  # (N, M)
        cost = (cost_l1 * l1 - cost_iou * giou
               - cost_presence * prob[b][:, None].clamp(min=1e-6).log())
        ri, ci = linear_sum_assignment(cost.float().cpu().numpy())
        out.append((torch.as_tensor(ri, dtype=torch.long, device=spans.device),
                   torch.as_tensor(ci, dtype=torch.long, device=spans.device)))
    return out


class SetSpanLoss:
    """Assembles every term for the query model. Returns (total, logs)."""

    def __init__(self, cfg: dict, n_bins: int):
        w = cfg.get("loss", {})
        self.n_bins = n_bins
        self.w_presence = float(w.get("presence", 1.0))
        self.w_reg = float(w.get("reg", 2.0))
        self.w_dfl = float(w.get("dfl", 0.5))
        self.w_dice = float(w.get("dice", 1.0))
        self.w_frame = float(w.get("frame", 0.5))
        self.w_clip = float(w.get("clip", 0.5))
        self.w_speech = float(w.get("speech", 0.2))
        self.w_bmap = float(w.get("bmap", 1.0))
        self.silver_bmap = float(w.get("silver_bmap_weight", 0.15))
        self.silver_frame = float(w.get("silver_frame_weight", 0.5))
        # Negative (unmatched) queries are the vast majority with n_queries=24
        # and ~1.4 events/clip; down-weighting them the way DETR's `eos_coef`
        # does keeps them from drowning the few positives in the focal sum -
        # sigmoid_focal's own alpha/gamma already help, eos_coef is the second
        # knob if that alone is not enough.
        self.eos_coef = float(w.get("eos_coef", 0.1))
        self.fps = float(cfg["data"]["fps"])

    def _stage_loss(self, out: dict, spans_sec: torch.Tensor,
                    valid: torch.Tensor, ev_w: torch.Tensor) -> tuple:
        presence = out["presence"]
        B, N = presence.shape
        matches = hungarian_match(out["span"], presence, spans_sec, valid)

        pos = presence.new_zeros(B, N)
        tgt_s = presence.new_zeros(B, N)
        tgt_e = presence.new_zeros(B, N)
        bw = presence.new_zeros(B, N)
        for b, (qi, ti) in enumerate(matches):
            if qi.numel() == 0:
                continue
            valid_spans = spans_sec[b][valid[b]]
            pos[b, qi] = 1.0
            tgt_s[b, qi] = valid_spans[ti, 0]
            tgt_e[b, qi] = valid_spans[ti, 1]
            bw[b, qi] = ev_w[b]

        n_pos = pos.sum().clamp(min=1.0)
        pres_w = torch.where(pos > 0.5, torch.ones_like(pos),
                             torch.full_like(pos, self.eos_coef))
        l_presence = sigmoid_focal(presence, pos, pres_w) / n_pos

        one = torch.ones_like(pos)
        ps = torch.where(pos > 0.5, out["span"][..., 0], one)
        pe = torch.where(pos > 0.5, out["span"][..., 1], one)
        ta = torch.where(pos > 0.5, tgt_s, one)
        tb = torch.where(pos > 0.5, tgt_e, one)
        n_bw = (pos * bw).sum().clamp(min=1e-3)
        l_reg = (diou_1d(ps, pe, ta, tb) * pos * bw).sum() / n_bw

        delta = out["delta_max"]
        d_s = ((tgt_s - out["ref"][..., 0]) / delta).clamp(-1, 1)
        d_e = ((tgt_e - out["ref"][..., 1]) / delta).clamp(-1, 1)
        bin_s = (d_s + 1) / 2 * (self.n_bins - 1)
        bin_e = (d_e + 1) / 2 * (self.n_bins - 1)
        nb = out["start_logits"].size(-1)
        dfl = (distribution_focal(out["start_logits"].reshape(B * N, nb),
                                  bin_s.reshape(-1))
              + distribution_focal(out["end_logits"].reshape(B * N, nb),
                                   bin_e.reshape(-1)))
        l_dfl = (dfl.view(B, N) * pos * bw).sum() / n_bw / 2

        return l_presence, l_reg, l_dfl

    def __call__(self, outs: List[dict], batch: dict) -> tuple:
        tier = batch["tier"]
        valid = batch["span_cls"] >= 0
        spans_sec = batch["spans"] / self.fps                    # (B, M, 2)
        ev_w = torch.where(tier == TIER_GOLD, torch.ones_like(tier, dtype=spans_sec.dtype),
                           torch.full_like(tier, 0.25, dtype=spans_sec.dtype))

        l_presence = l_reg = l_dfl = outs[0]["presence"].new_zeros(())
        for out in outs:
            p, r, d = self._stage_loss(out, spans_sec, valid, ev_w)
            l_presence, l_reg, l_dfl = l_presence + p, l_reg + r, l_dfl + d
        n_stages = len(outs)
        l_presence, l_reg, l_dfl = l_presence / n_stages, l_reg / n_stages, l_dfl / n_stages

        # --- auxiliary frame-level terms (base-level heads, unchanged from
        # VaaniSpanModel) - deliberately mirrors SpanLoss's own block. ---
        vmask = batch["frame_valid"]
        strong = (tier != TIER_BRONZE).float()
        fw = torch.where(tier == TIER_GOLD, torch.ones_like(strong),
                         torch.full_like(strong, self.silver_frame)) * strong

        frame_bce = F.binary_cross_entropy_with_logits(
            outs[-1]["frame_logits"], batch["frame_target"], reduction="none")
        frame_bce = (frame_bce.mean(-1) * vmask).sum(1) / vmask.sum(1).clamp(min=1)
        l_frame = (frame_bce * fw).sum() / fw.sum().clamp(min=1)

        agn_t = batch["frame_target"].amax(-1)
        l_dice = soft_dice(outs[-1]["agn_logits"], agn_t, vmask, fw)

        with torch.autocast(device_type=outs[-1]["clip_probs"].device.type, enabled=False):
            l_clip = F.binary_cross_entropy(
                outs[-1]["clip_probs"].float(), batch["clip_target"].float())

        has_vad = batch.get("has_vad")
        if has_vad is None:
            l_speech = outs[-1]["presence"].new_zeros(())
        else:
            sb = F.binary_cross_entropy_with_logits(
                outs[-1]["speech_logits"], batch["speech_target"], reduction="none")
            sb = (sb * vmask).sum(1) / vmask.sum(1).clamp(min=1)
            l_speech = (sb * has_vad).sum() / has_vad.sum().clamp(min=1)

        if "onset_logits" in outs[-1] and self.w_bmap > 0:
            n_hi = outs[-1]["onset_logits"].size(1)
            n_frames = vmask.size(1)
            mult = n_hi / max(1, n_frames)
            bt, bwt = boundary_targets(batch["spans"], vmask, n_hi, mult, tier,
                                       self.silver_bmap)
            blog = torch.stack([outs[-1]["onset_logits"], outs[-1]["offset_logits"]], 1)
            hm = outs[-1]["hi_mask"].unsqueeze(1)
            bl = sigmoid_focal(blog, bt, bwt * hm, alpha=0.5, gamma=2.0)
            l_bmap = bl / (bt > 0.05).float().mul(hm).sum().clamp(min=1.0)
        else:
            l_bmap = outs[-1]["presence"].new_zeros(())

        total = (self.w_presence * l_presence + self.w_reg * l_reg + self.w_dfl * l_dfl
                + self.w_frame * l_frame + self.w_dice * l_dice + self.w_clip * l_clip
                + self.w_speech * l_speech + self.w_bmap * l_bmap)

        logs = {"loss": total.detach(), "presence": l_presence.detach(),
               "reg": l_reg.detach(), "dfl": l_dfl.detach(), "frame": l_frame.detach(),
               "dice": l_dice.detach(), "clip": l_clip.detach(),
               "speech": l_speech.detach(), "bmap": l_bmap.detach()}
        return total, logs
