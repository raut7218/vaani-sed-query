"""Sparse set-prediction span decoder (query head).

TridentHead reads boundaries off a dense per-point regression grid and needs
SoftNMS plus a separate count head to turn that into a final set - both are
post-hoc approximations of "pick the right subset", which this repo's own
diagnosis (losses.py, README) measured at +0.28 of the score against +0.037
for picking the right *count* over the same candidate pool. A DETR/TadTR-style
decoder is trained end-to-end, via Hungarian matching, on exactly that
question: N learned queries cross-attend to the FPN's finest level and each
one emits presence + a span directly, with no threshold or NMS needed to reach
the final answer - the count head and SoftNMS both disappear.

Iterative refinement (Deformable DETR / Sparse R-CNN): each decoder stage
corrects the *previous* stage's span rather than predicting from scratch, so
the first stage only has to be roughly right and later stages do the boundary
precision work. The DFL bin-expectation trick from trident.py is reused
unchanged for the correction itself - only what the bins express (a delta from
the current estimate, not a distance from a fixed grid point) is new.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn


def sinusoidal_pe(n: int, d: int, device) -> torch.Tensor:
    """(n, d) fixed sinusoidal position embedding, added to decoder memory."""
    pos = torch.arange(n, device=device).float().unsqueeze(1)
    i = torch.arange(d, device=device).float().unsqueeze(0)
    freq = torch.exp(-math.log(10000.0) * (2 * (i // 2)) / d)
    ang = pos * freq
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(ang[:, 0::2])
    pe[:, 1::2] = torch.cos(ang[:, 1::2])
    return pe


class RefineStage(nn.Module):
    """One decoder stage's prediction: presence + a DFL delta on (start, end).

    `delta_max` is in seconds and the bins span [-delta_max, +delta_max]: a
    coarse first stage wants a wide range (the initial guess can be anywhere in
    the window) and later stages want a narrow one (they only correct a stage
    that is already close) - that asymmetry is what turns N independent
    regressions into a refinement cascade.
    """

    def __init__(self, d_model: int, n_bins: int, delta_max: float):
        super().__init__()
        self.n_bins = n_bins
        self.delta_max = float(delta_max)
        self.presence = nn.Linear(d_model, 1)
        self.start_out = nn.Linear(d_model, n_bins)
        self.end_out = nn.Linear(d_model, n_bins)
        nn.init.constant_(self.presence.bias, -math.log((1 - 0.02) / 0.02))
        self.register_buffer("bins", torch.linspace(-1.0, 1.0, n_bins), persistent=False)

    def forward(self, q: torch.Tensor, ref: torch.Tensor) -> dict:
        """q: (B, N, D) query states. ref: (B, N, 2) current (start, end) sec."""
        s_logits = self.start_out(q)                    # (B, N, bins)
        e_logits = self.end_out(q)
        s_p = s_logits.softmax(-1)
        e_p = e_logits.softmax(-1)
        d_s = (s_p * self.bins).sum(-1) * self.delta_max
        d_e = (e_p * self.bins).sum(-1) * self.delta_max
        start = ref[..., 0] + d_s
        end = torch.maximum(ref[..., 1] + d_e, start + 0.02)
        return {"presence": self.presence(q).squeeze(-1),
                "span": torch.stack([start, end], -1),
                "start_logits": s_logits, "end_logits": e_logits,
                "ref": ref, "delta_max": self.delta_max}


class SpanQueryDecoder(nn.Module):
    def __init__(self, d_model: int = 384, n_queries: int = 24, n_bins: int = 16,
                 n_stages: int = 3, n_head: int = 8, dropout: float = 0.1,
                 clip_len: float = 8.0, coarse_delta: float = 4.0,
                 refine_delta: float = 0.4):
        super().__init__()
        self.n_queries = n_queries
        self.clip_len = float(clip_len)
        self.query_embed = nn.Embedding(n_queries, d_model)
        # Learned, but initialised to evenly-spaced (centre, width) anchors
        # across the clip - not derived from `query_embed` through a shared
        # linear layer. That derived form puts every query at the *same*
        # sigmoid(0) = clip_len/2 centre at step 0 regardless of index, so
        # breaking symmetry is left entirely to gradient descent on the small
        # random differences between embeddings - slow, and with few queries
        # it can fail to break at all (every clip decodes to one shared
        # "average" span). Spreading the anchors deterministically at init is
        # the DAB-DETR / Anchor-DETR fix for exactly this collapse.
        centers = (torch.arange(n_queries).float() + 0.5) / n_queries * clip_len
        widths = torch.full((n_queries,), clip_len / max(2, n_queries))
        self.ref_points = nn.Parameter(torch.stack([centers, widths], dim=-1))
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            for _ in range(n_stages)])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            for _ in range(n_stages)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4),
                          nn.GELU(), nn.Linear(d_model * 4, d_model))
            for _ in range(n_stages)])
        self.norm_sa = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_stages)])
        self.norm_ca = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_stages)])
        deltas = [coarse_delta] + [refine_delta] * (n_stages - 1)
        self.stages = nn.ModuleList([RefineStage(d_model, n_bins, dm) for dm in deltas])

    def forward(self, memory: torch.Tensor, mem_mask: torch.Tensor) -> List[dict]:
        """memory: (B, T, D) base-level FPN features. mem_mask: (B, T), 1=valid.

        Returns one dict per stage (deep supervision trains against all of
        them; inference decodes from the last).
        """
        B = memory.size(0)
        pe = sinusoidal_pe(memory.size(1), memory.size(2), memory.device)
        mem = memory + pe.unsqueeze(0)
        # An all-masked row would make softmax over an entirely -inf key row
        # produce NaN; clips are never fully padding here, but guard anyway,
        # the same way ConvTransformerBlock does.
        key_pad = mem_mask < 0.5
        allpad = key_pad.all(dim=1)
        key_pad = key_pad.clone()
        key_pad[:, 0] = key_pad[:, 0] & ~allpad

        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        center = self.ref_points[:, 0].clamp(0.0, self.clip_len)
        width = self.ref_points[:, 1].clamp(min=0.05)
        start = (center - width / 2).clamp(min=0.0)
        end = torch.maximum((center + width / 2).clamp(max=self.clip_len), start + 0.02)
        ref = torch.stack([start, end], -1).unsqueeze(0).expand(B, -1, -1)

        outs = []
        for i in range(len(self.stages)):
            a, _ = self.self_attn[i](q, q, q, need_weights=False)
            q = self.norm_sa[i](q + a)
            c, _ = self.cross_attn[i](q, mem, mem, key_padding_mask=key_pad,
                                      need_weights=False)
            q = self.norm_ca[i](q + c)
            q = q + self.ffn[i](q)
            out = self.stages[i](q, ref)
            ref = out["span"].detach()          # next stage refines from here
            outs.append(out)
        return outs


@torch.no_grad()
def decode_queries(outs: List[dict], score_thr: float = 0.5
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Last stage's spans and presence probabilities. No NMS, no count head:

    the model is trained (Hungarian matching) to answer "which subset"
    directly, so thresholding presence *is* the selection. `runner.py` may
    still run a light SoftNMS pass as a safety net against near-duplicate
    queries; that is a decode-time concern, not this function's.
    """
    last = outs[-1]
    return last["span"], last["presence"].sigmoid()
