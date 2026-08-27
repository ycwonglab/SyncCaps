"""TR-Caps: CTM-style temporal routing head and per-tick loss.

Design: docs/plans/2026-07-05-ctm-temporal-routing-design.md
Loss ported from the CTM repo (arXiv:2505.05522), utils/losses.py
image_classification_loss; entropy from models/utils.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_entropy(logits, eps=1e-12):
    """Entropy of softmax(logits) over the last dim, normalized to [0, 1]."""
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return ent / (math.log(logits.shape[-1]) + eps)


def ctm_tick_loss(logits, certainties, targets, use_most_certain=True):
    """Mean of CE at the per-sample min-loss tick and the most-certain tick.

    logits: [B, C, T]; certainties: [B, 2, T] with [:, 1] = 1 - normalized
    entropy; targets: [B]. Returns (loss, most_certain_tick_index[B]).
    With use_most_certain=False the second tick is always the final one.
    """
    T = logits.size(-1)
    targets_expanded = targets.unsqueeze(-1).expand(-1, T)
    losses = nn.CrossEntropyLoss(reduction='none')(logits, targets_expanded)  # [B, T]

    idx_min = losses.argmin(dim=1)
    idx_sel = certainties[:, 1].argmax(-1)
    if not use_most_certain:
        idx_sel = torch.full_like(idx_sel, T - 1)

    b = torch.arange(logits.size(0), device=logits.device)
    loss = (losses[b, idx_min].mean() + losses[b, idx_sel].mean()) / 2
    return loss, idx_sel


def hybrid_readout(logits, theta=0.5):
    """Accumulate-then-exit readout (causal, deployable).

    Prefix-means the per-tick logits, then exits at the first tick whose
    certainty (1 - normalized entropy of the ACCUMULATED logits) reaches
    theta, falling back to the final tick. Averaging cancels per-tick logit
    noise, so this dominates both the instantaneous most-certain readout and
    full-clip accumulation (see docs/plans/2026-07-05-ctm-temporal-routing-
    design.md, readout-policy study).

    logits: [B, C, T]. Returns (pred [B], exit_tick [B] 0-indexed,
    accumulated logits [B, C, T]).
    """
    B, C, T = logits.shape
    denom = torch.arange(1, T + 1, device=logits.device, dtype=logits.dtype)
    acc = logits.cumsum(-1) / denom
    certainty = 1 - normalized_entropy(acc.transpose(1, 2))     # [B, T]
    ticks = torch.arange(T, device=logits.device).expand(B, T)
    idx = torch.where(certainty >= theta, ticks, T - 1).min(dim=1).values
    b = torch.arange(B, device=logits.device)
    pred = acc[b, :, idx].argmax(1)
    return pred, idx, acc


class TemporalRoutingCaps(nn.Module):
    """Recurrent routing head: one tick per frame.

    Per tick t: votes u_hat from frame-t primary capsules; coupling from
    agreement with the carried consensus v_{t-1}; leaky evidence accumulation
    with learned per-class decay r = exp(-rho); CTM-style count normalization
    v_t = squash(s_t / sqrt(beta_t)); pose-sync statistic alpha_t/sqrt(beta_t);
    per-tick logits tau*||v_t|| + lam*sync_t and certainty 1 - norm. entropy.
    """

    def __init__(self, in_capsules, in_dim=8, num_classes=11, out_dim=16,
                 w_init=0.01, w_init_scale=1.0, tau=10.0, use_sync=True,
                 lambda_init=0.1):
        super().__init__()
        self.num_classes, self.out_dim, self.tau = num_classes, out_dim, tau
        self.W = nn.Parameter(w_init * w_init_scale *
                              torch.randn(1, in_capsules, num_classes,
                                          out_dim, in_dim))
        self.rho = nn.Parameter(torch.zeros(num_classes))
        bound = math.sqrt(1.0 / out_dim)
        self.v0 = nn.Parameter(
            torch.zeros(num_classes, out_dim).uniform_(-bound, bound))
        if use_sync:
            self.lam = nn.Parameter(torch.tensor(float(lambda_init)))
        else:
            self.register_buffer('lam', torch.tensor(0.0))

    @staticmethod
    def squash(v, eps=1e-8):
        n = v.norm(dim=-1, keepdim=True)
        return (n ** 2 / (1 + n ** 2)) * v / (n + eps)

    def forward(self, primary_seq):
        B, T, N, _ = primary_seq.shape
        C, D = self.num_classes, self.out_dim
        dev = primary_seq.device

        self.rho.data.clamp_(0, 15)          # CTM decay clamp (kuviki fix)
        r = torch.exp(-self.rho)             # [C]
        rC = r.view(1, C)
        rCD = r.view(1, C, 1)

        v = self.v0.unsqueeze(0).expand(B, -1, -1)          # [B, C, D]
        s = torch.zeros(B, C, D, device=dev)
        alpha = torch.zeros(B, C, device=dev)
        beta = torch.zeros(B, C, device=dev)
        W = self.W.expand(B, -1, -1, -1, -1)                # [B, N, C, D, in]

        logits, certainties, syncs, v_states = [], [], [], []
        for t in range(T):
            u = primary_seq[:, t]                            # [B, N, in]
            u_hat = torch.matmul(
                W, u.unsqueeze(2).unsqueeze(-1)).squeeze(-1)  # [B, N, C, D]
            b = (u_hat * v.unsqueeze(1)).sum(-1, keepdim=True)  # [B, N, C, 1]
            c = F.softmax(b, dim=2)
            s = rCD * s + (c * u_hat).sum(dim=1)             # [B, C, D]
            beta = rC * beta + 1
            v = self.squash(s / beta.sqrt().unsqueeze(-1))
            a = (c.squeeze(-1) *
                 (u_hat * v.unsqueeze(1)).sum(-1)).mean(dim=1)  # [B, C]
            alpha = rC * alpha + a
            sync = alpha / beta.sqrt()
            logit = self.tau * v.norm(dim=-1) + self.lam * sync
            ne = normalized_entropy(logit)
            logits.append(logit)
            certainties.append(torch.stack((ne, 1 - ne), dim=1))
            syncs.append(sync)
            v_states.append(v)

        return dict(logits=torch.stack(logits, dim=-1),
                    certainties=torch.stack(certainties, dim=-1),
                    sync=torch.stack(syncs, dim=-1),
                    v_states=torch.stack(v_states, dim=-1),
                    v_last=v)


from torchvision.models import resnet18

from .capsule_layers import PrimaryCaps


class TRCapsNet(nn.Module):
    """V1+ TR-Caps: ResNet-18 body (random init, = V2's architecture) ->
    per-frame S3 grid -> PrimaryCaps -> TemporalRoutingCaps (tick = frame)."""

    def __init__(self, num_classes=11, caps_grid=3, w_init_scale=20.0,
                 use_sync=True, tau=10.0, shuffle_frames=False):
        super().__init__()
        self.caps_grid = caps_grid
        self.shuffle_frames = shuffle_frames
        m = resnet18(weights=None)
        self.conv = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                  m.layer1, m.layer2, m.layer3)  # 256 ch
        self.pre_caps_norm = nn.LayerNorm([256, caps_grid, caps_grid])
        self.primary = PrimaryCaps(256, 32, 8)
        self.routing = TemporalRoutingCaps(
            in_capsules=32 * caps_grid * caps_grid, in_dim=8,
            num_classes=num_classes, out_dim=16,
            w_init_scale=w_init_scale, tau=tau, use_sync=use_sync)

    def forward(self, x):
        B, T, C, H, W = x.shape
        if self.shuffle_frames:
            x = x[:, torch.randperm(T, device=x.device)]
        f = self.conv(x.reshape(B * T, C, H, W))
        f = F.adaptive_avg_pool2d(f, self.caps_grid)
        f = self.pre_caps_norm(f)
        u = self.primary(f)                       # [B*T, N, 8]
        u = u.view(B, T, -1, 8)
        return self.routing(u)
