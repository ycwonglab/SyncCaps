"""Sync-Caps: CTM synchronization-as-representation over capsule states.

Design: docs/plans/2026-07-06-sync-as-representation-design.md
Sync recurrence ported from the CTM repo (arXiv:2505.05522), models/ctm.py
compute_synchronisation / set_synchronisation_parameters.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from .temporal_routing import normalized_entropy, TemporalRoutingCaps
from .capsule_layers import PrimaryCaps


def assert_frozen_bn_eval(model):
    """Acceptance check for the frozen-stem protocol (v5 repair plan section 1).

    The 2026-08-21 defect was invisible to every ordinary freeze check: the stem
    had requires_grad=False throughout, so only `training` on its BatchNorm
    modules distinguished a correctly frozen run from a broken one -- and nothing
    inspected that. This asserts it directly, on every call to `.train()`, so the
    class of bug cannot recur silently.

    Cost is one walk over the stem's submodules per epoch, which is nothing
    beside a training epoch. Skipped when SYNC_FROZEN_BN_TRAIN=1 deliberately
    restores the legacy behaviour, since there the training-mode BN IS the
    requested configuration.
    """
    if os.environ.get('SYNC_FROZEN_BN_TRAIN'):
        return
    bad = [n for n, m in model.conv.named_modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
    if bad:
        raise AssertionError(
            'frozen stem has {} BatchNorm module(s) in TRAINING mode after '
            '.train(): {}. They would normalise with per-batch statistics and '
            'mutate their running stats, which is the 2026-08-21 defect.'
            .format(len(bad), ', '.join(bad[:5])))


class PairwiseSync(nn.Module):
    """Decay-weighted pairwise products of activation traces; sync = a/sqrt(b).

    mode='random-pairing': n_synch fixed random pairs over d_model neurons,
    the first n_self of them self-pairs (i == j) so activation magnitudes stay
    recoverable. mode='full': all upper-triangular pairs (n_synch ignored).

    pose_coupling (per-tick pair statistic; 'dot'/'cosine' tested 2026-07,
    both lose to 'scalar' on UCF-11 — see docs/paper §5.5/§6.2):
      scalar   z[i]*z[j] over flat neurons (baseline B1).
      dot      <u_i,u_j> over capsules — trace of u_i u_j^T, a fixed isotropic
               projection of the 64-entry pose-interaction matrix.
      cosine   direction-only dot on cross pairs (presence kept on self-pairs).
      bilinear (u_i . a_k)(u_j . b_k), learned rank-1 metric per pair; a_k/b_k
               init one-hot so at init this IS a cross-capsule scalar pair,
               and training can rotate the projections into pose space.
      aligned  <W_{type(i)} u_i, W_{type(j)} u_j> with one learned 8x8 frame
               per capsule TYPE (index % n_caps_types), init identity — votes
               without routing; agreement measured in a learned common frame.
      outer    vec(u_i u_j^T), the full 64-D interaction, traced per entry
               (decay shared per pair) — pose covariance is accumulated over
               ticks BEFORE any collapse, so rigid co-rotation is visible.
               out_dim = 64*n_synch: use fewer pairs to match head budgets.
    """

    POSE_MODES = ('scalar', 'dot', 'cosine', 'bilinear', 'aligned', 'outer')

    def __init__(self, d_model, n_synch=1024, mode='random-pairing',
                 n_self=64, seed=0, pose_coupling='scalar', caps_dim=8,
                 n_caps_types=32, sync_norm=False, exclude_self=False):
        super().__init__()
        if pose_coupling not in self.POSE_MODES:
            raise ValueError(f'unknown pose_coupling: {pose_coupling}')
        # sync_norm: signed-sqrt + L2 on the emitted statistic (improved-B-CNN
        # normalisation). At r->1 sync IS a subsampled Gram matrix, so the raw
        # products are heavy-tailed and a few pairs dominate the head's input
        # scale; /sqrt(beta) normalises the COUNT, not the distribution.
        # Off by default so every pre-2026-08 arm stays bit-identical.
        self.sync_norm = sync_norm
        self.mode = mode
        self.pose_coupling = pose_coupling
        self.caps_dim = caps_dim
        # scalar: pair over d_model flat neurons. dot/cosine: pair over the
        # N = d_model // caps_dim capsules (couple their pose vectors).
        if pose_coupling == 'scalar':
            bound = d_model
        else:
            if d_model % caps_dim != 0:
                raise ValueError(
                    f'caps_dim ({caps_dim}) must divide d_model ({d_model})')
            bound = d_model // caps_dim
        if mode == 'full':
            left, right = torch.triu_indices(bound, bound)
        elif mode == 'random-pairing':
            if n_self > n_synch:
                raise ValueError(
                    f'n_self ({n_self}) must be <= n_synch ({n_synch})')
            g = torch.Generator().manual_seed(seed)
            left = torch.randint(0, bound, (n_synch,), generator=g)
            right = torch.cat(
                (left[:n_self],
                 torch.randint(0, bound, (n_synch - n_self,), generator=g)))
            if exclude_self and n_synch > n_self:
                # The cross block draws both endpoints INDEPENDENTLY, so about
                # n_cross/bound of its pairs land on i == j by chance (0.89 of
                # 2048 at bound=2304). That is invisible in accuracy but fatal
                # for a cross-ONLY arm, whose entire claim is that no squared
                # activation reaches the head. Resample the collisions until
                # none remain. OFF by default and gated on the flag, not on
                # n_self == 0, so every pair dictionary committed before
                # 2026-08-21 is reproduced bit-identically: the loop is the only
                # thing that would draw extra values from `g`.
                if bound < 2:
                    raise ValueError('exclude_self needs bound >= 2, '
                                     f'got {bound}')
                idx = torch.arange(n_self, n_synch)
                for _ in range(64):
                    clash = idx[left[idx] == right[idx]]
                    if clash.numel() == 0:
                        break
                    right[clash] = torch.randint(
                        0, bound, (clash.numel(),), generator=g)
                else:
                    raise RuntimeError('exclude_self failed to converge')
        else:
            raise ValueError(f'unknown mode: {mode}')
        self.n_synch = left.numel()
        self.register_buffer('left', left)
        self.register_buffer('right', right)
        self.register_buffer('is_self', left == right, persistent=False)
        self.rho = nn.Parameter(torch.zeros(self.n_synch))
        # feature width seen by the head (== n_synch except 'outer')
        self.out_dim = self.n_synch * (caps_dim ** 2
                                       if pose_coupling == 'outer' else 1)
        if pose_coupling == 'bilinear':
            # one-hot init: pair k starts as the plain scalar product of one
            # random coordinate of u_i with one of u_j (== a scalar cross-
            # capsule pair), so 'bilinear' can only move away from the scalar
            # baseline if the learned pose metric earns it.
            g = torch.Generator().manual_seed(seed + 1)
            a = F.one_hot(torch.randint(0, caps_dim, (self.n_synch,),
                                        generator=g), caps_dim).float()
            b = F.one_hot(torch.randint(0, caps_dim, (self.n_synch,),
                                        generator=g), caps_dim).float()
            self.proj_a = nn.Parameter(a)
            self.proj_b = nn.Parameter(b)
        elif pose_coupling == 'aligned':
            self.frames = nn.Parameter(
                torch.eye(caps_dim).repeat(n_caps_types, 1, 1))
            self.n_caps_types = n_caps_types

    def forward(self, z, alpha=None, beta=None):
        """z: [B, d_model] fp32. Returns (sync [B, out_dim], alpha, beta)."""
        self.rho.data.clamp_(0, 15)          # CTM decay clamp (kuviki fix)
        r = torch.exp(-self.rho).unsqueeze(0)
        prod = self._couple(z)
        while r.dim() < prod.dim():          # broadcast over 'outer' entries
            r = r.unsqueeze(-1)
        if alpha is None:
            alpha, beta = prod, torch.ones_like(prod)
        else:
            alpha = r * alpha + prod
            beta = r * beta + 1
        sync = (alpha / beta.sqrt()).flatten(1)
        if self.sync_norm:
            sync = torch.sign(sync) * (sync.abs() + 1e-8).sqrt()
            sync = F.normalize(sync, dim=1)
        return sync, alpha, beta

    def _couple(self, z, eps=1e-8):
        """Per-tick co-activation for each pair. [B, d_model] -> [B, n_synch]
        ([B, n_synch, caps_dim, caps_dim] for 'outer')."""
        if self.pose_coupling == 'scalar':
            return z[:, self.left] * z[:, self.right]
        u = z.view(z.shape[0], -1, self.caps_dim)   # [B, N, caps_dim]
        if self.pose_coupling == 'aligned':
            # rotate each capsule into its type's learned frame first
            types = torch.arange(u.shape[1], device=u.device) % self.n_caps_types
            u = torch.einsum('nde,bnd->bne', self.frames[types], u)
        ui = u[:, self.left]                          # [B, n_synch, caps_dim]
        uj = u[:, self.right]
        if self.pose_coupling == 'bilinear':
            return (ui * self.proj_a).sum(-1) * (uj * self.proj_b).sum(-1)
        if self.pose_coupling == 'outer':
            return ui.unsqueeze(-1) * uj.unsqueeze(-2)  # [B, n_synch, d, d]
        dot = (ui * uj).sum(-1)                        # [B, n_synch]
        if self.pose_coupling in ('dot', 'aligned'):
            return dot
        # cosine: normalize cross-pairs to pure direction agreement; keep raw
        # ||u||^2 on self-pairs so presence survives (cos of a self-pair == 1).
        cos = dot / (ui.norm(dim=-1) * uj.norm(dim=-1) + eps)
        return torch.where(self.is_self.unsqueeze(0), dot, cos)


class _SecondOrderPool(nn.Module):
    """Shared tick accumulator for the second-order BASELINE readouts.

    PairwiseSync's accumulation is alpha_t = r*alpha_{t-1} + prod_t,
    beta_t = r*beta_{t-1} + 1, feat = alpha/sqrt(beta). These baselines pin
    r = 1, which is exactly the B4_gram operating point (frozen rho = 0) and
    the order-agnostic limit the paper's Section 5.4 controls land on. Pinning
    it is what makes the comparison single-variable: stem, primaries, tick
    loop, normalisation and head width are all shared with B4_gram, so the ONLY
    thing that differs between these arms is HOW the second-order statistic is
    reduced to `out_dim` features.

    Note the r = 1 accumulation divides by sqrt(count), not count. With
    sync_norm on that distinction is vacuous: signed-sqrt turns a uniform scale
    c into sqrt(c) and the following L2 normalisation removes it entirely, so
    "count-normalised mean" and "sum over sqrt(count)" are the same vector.

    Subclasses implement _prod(z) -> [B, out_dim] and set self.out_dim.
    The forward/attribute contract mirrors PairwiseSync (returns
    (feat, alpha, beta), carries .out_dim and .rho) so SyncCapsNet.forward, the
    per-tick CTM loss, the hybrid early exit and run_experiment_sync's decay
    histogram all work against these arms with no branching.
    """

    def _register_rho(self):
        # A non-trainable zero BUFFER, not a Parameter. These baselines have no
        # learned temporal decay by construction (r = exp(0) = 1); keeping the
        # attribute lets run_experiment_sync log an honest all-zero decay
        # histogram instead of raising on a missing field.
        self.register_buffer('rho', torch.zeros(self.out_dim), persistent=False)

    def forward(self, z, alpha=None, beta=None):
        """z: [B, d_model] fp32. Returns (feat [B, out_dim], alpha, beta)."""
        prod = self._prod(z)
        if alpha is None:
            alpha, beta = prod, torch.ones_like(prod)
        else:
            alpha = alpha + prod          # r = 1, pinned (see class docstring)
            beta = beta + 1
        feat = alpha / beta.sqrt()
        if self.sync_norm:
            feat = torch.sign(feat) * (feat.abs() + 1e-8).sqrt()
            feat = F.normalize(feat, dim=1)
        return feat, alpha, beta


class CompactBilinear(_SecondOrderPool):
    """Compact bilinear pooling via Tensor Sketch (Gao et al., CVPR 2016).

    WHY THIS ARM EXISTS (2026-08-20). Section 2.4 of the paper concedes that at
    its learned operating point PairwiseSync IS a subsampled Gram statistic
    under improved-B-CNN normalisation, and names compact bilinear pooling as
    the established way to escape the quadratic width -- then never runs it.
    This is that run. PairwiseSync samples n_synch Gram ENTRIES; Tensor Sketch
    applies a Count-Sketch random projection to the whole outer product and
    keeps all of it in out_dim dimensions. Same width, same normalisation, same
    everything else, so the contrast is the projection scheme alone.

    Sketching identity (Pham & Pagh, KDD 2013): the Count Sketch of x (x) y is
    the circular convolution of their individual sketches, computed here as an
    elementwise product in the Fourier domain. Two INDEPENDENT sketches of the
    same z are required -- reusing one would sketch z (x) z with correlated
    hashes and bias the estimate.
    """

    def __init__(self, d_model, out_dim=2048, seed=0, sync_norm=True):
        super().__init__()
        self.out_dim = out_dim
        self.sync_norm = sync_norm
        g = torch.Generator().manual_seed(seed)
        for k in (1, 2):
            self.register_buffer(
                f'h{k}', torch.randint(0, out_dim, (d_model,), generator=g))
            self.register_buffer(
                f's{k}',
                torch.randint(0, 2, (d_model,), generator=g).float() * 2 - 1)
        self._register_rho()

    def _sketch(self, z, h, s):
        # Count Sketch: bucket each of the d_model coordinates by h, signed by
        # s. index_add_ is the scatter-add form; it is differentiable in z.
        return z.new_zeros(z.shape[0], self.out_dim).index_add_(1, h, z * s)

    def _prod(self, z):
        f1 = torch.fft.rfft(self._sketch(z, self.h1, self.s1), dim=1)
        f2 = torch.fft.rfft(self._sketch(z, self.h2, self.s2), dim=1)
        return torch.fft.irfft(f1 * f2, n=self.out_dim, dim=1)


class LowRankBilinear(_SecondOrderPool):
    """Dense second-order reference: the FULL Gram of a learned low-rank
    projection (low-rank bilinear CNN).

    The third way to fit a quadratic statistic into a fixed width, and the
    "widest tractable pair dictionary" the experiment plan asks for as a
    ceiling estimate. PairwiseSync samples entries, CompactBilinear projects
    the product; this projects the FEATURES first and then keeps every pairwise
    product of the survivors. At rank 64 the upper triangle is 2080 entries,
    within 2% of n_synch = 2048, so the head width is matched.

    The projection is LEARNED, which is the strong form of the baseline. It
    costs d_model x rank = 147k trainable parameters that the sampled-pair arms
    do not spend, so this arm is width-matched but NOT parameter-matched, and
    both counts are reported rather than one of them quietly chosen.
    """

    def __init__(self, d_model, rank=64, sync_norm=True):
        super().__init__()
        self.proj = nn.Linear(d_model, rank, bias=False)
        iu, ju = torch.triu_indices(rank, rank)
        self.register_buffer('iu', iu)
        self.register_buffer('ju', ju)
        self.out_dim = int(iu.numel())
        self.sync_norm = sync_norm
        self._register_rho()

    def _prod(self, z):
        p = self.proj(z)
        return p[:, self.iu] * p[:, self.ju]

def _conv4_stem(dropout_rate):
    """Original 4-conv stem (verbatim from EnhancedCapsNet, incl. dropout —
    the 78.8% norm-readout baseline). 3 -> 256 channels, /16 spatial."""
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Dropout2d(p=dropout_rate * 0.5),
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=dropout_rate),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=dropout_rate),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_rate * 0.5),
    )


# Deeper ResNet stems (2026-08-17). `resnet` truncates at layer3 (256 ch); these
# keep the FULL body through layer4 and project back to 256, so the capsule stack
# and readout stay identical. Same family and same pretraining data as `resnet`,
# so a rung added here isolates CAPACITY/DEPTH -- unlike the CLIP rung, which
# changes architecture and pretraining corpus at the same time.
# (torchvision ctor, weights enum name, layer4 output channels).
_RESNET_STEMS = {
    'r18_full': ('resnet18', 'ResNet18_Weights', 512),
    'r50_full': ('resnet50', 'ResNet50_Weights', 2048),
}


def _resnet_full_stem(key, pretrained=False, out_ch=256):
    """Full ResNet body (through layer4) + TRAINABLE 1x1 projection to out_ch."""
    import torchvision.models as tvm
    name, wname, ch = _RESNET_STEMS[key]
    w = getattr(tvm, wname).IMAGENET1K_V1 if pretrained else None
    m = getattr(tvm, name)(weights=w)
    body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                         m.layer1, m.layer2, m.layer3, m.layer4)
    return _with_trainable_tail(nn.Sequential(body, nn.Conv2d(ch, out_ch, 1)))


def _resnet_stem(pretrained=False):
    """ResNet-18 body to layer3 (256 ch). `pretrained=True` loads ImageNet
    weights — the direction-B stem for testing sync vs linear readout on a
    strong SHARED representation. Default random init is the from-scratch
    depth-stacking probe (identical to TRCapsNet's stem)."""
    w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=w)
    return nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                         m.layer1, m.layer2, m.layer3)


# MobileNet stems (2026-08-16). Efficiency probe against DVFL-Net's 22M distilled
# student. Each entry is (torchvision ctor name, truncate_at, out_channels).
# truncate_at=None keeps the FULL feature stack. At the pipeline's real 224x224
# input the truncated variants emit 14x14 (matching ResNet-18@layer3) and the
# full stacks emit 7x7 — both >= the caps_grid=3 adaptive pool, so all are valid.
_MOBILENET_STEMS = {
    'mnv3s':      ('mobilenet_v3_small',    9,   48),
    'mnv3s_full': ('mobilenet_v3_small', None,  576),
    'mnv3l':      ('mobilenet_v3_large',   13,  112),
    'mnv3l_full': ('mobilenet_v3_large', None,  960),
    'mnv2':       ('mobilenet_v2',         14,   96),
    'mnv2_full':  ('mobilenet_v2',       None, 1280),
}


def _with_trainable_tail(stem):
    """Tag a stem whose LAST module is our own (non-pretrained) projection.

    `freeze_stem` must never freeze that projection. Tagging the stem at
    construction — rather than testing membership of a stem-name dict at the
    freeze site — is the rule-level fix: any stem that appends its own layer
    declares it here and cannot be silently frozen later. The instance-level
    version of this check is exactly what invalidated the 2026-08-16 `_full`
    MobileNet triage runs.
    """
    stem.trainable_tail = True
    return stem


def _mobilenet_stem(key, pretrained=False, out_ch=256):
    """MobileNet body + 1x1 projection to `out_ch`.

    The projection keeps pre_caps_norm/PrimaryCaps identical across stems, so a
    stem swap is a single-variable change and the capsule/readout config stays
    comparable to the ResNet and conv4 arms.
    """
    import torchvision.models as tvm
    name, cut, ch = _MOBILENET_STEMS[key]
    body = getattr(tvm, name)(weights='DEFAULT' if pretrained else None).features
    body = nn.Sequential(*list(body)[:cut]) if cut is not None else body
    return _with_trainable_tail(nn.Sequential(body, nn.Conv2d(ch, out_ch, 1)))


# ViT stems (2026-08-17). The next rung up the feature-quality ladder from
# ResNet-18: the sync readout's advantage over the linear probe is monotone in
# stem quality (conv4 -1.47 -> mnv3s_full -0.02 -> ResNet +2.89 on UCF101), so a
# language-supervised ViT tests whether that dose-response keeps climbing.
# (timm ctor, embed_dim). The `quickgelu` variant is the faithful OpenAI CLIP
# graph -- the plain `vit_base_patch32_clip_224.openai` entry runs OpenAI weights
# through nn.GELU, which is not the activation they were trained with.
_VIT_STEMS = {
    'clip_b32': ('vit_base_patch32_clip_quickgelu_224.openai', 768),
}

# What exp_base.UCF11VideoDataset bakes into its cached frames (ImageNet stats).
_PIPELINE_MEAN = (0.485, 0.456, 0.406)
_PIPELINE_STD  = (0.229, 0.224, 0.225)


class _ViTGrid(nn.Module):
    """timm ViT -> [B, C, h, w] feature grid, with input stats corrected.

    Two jobs, both silent-corruption hazards if skipped:

    1. **Re-normalisation.** The data pipeline normalises with ImageNet stats and
       caches the result as fp16 .npy, so the frames CLIP would otherwise see are
       wrong by a per-channel affine. Undoing one affine and applying the other
       is itself affine, so it folds into a single fixed scale+shift held in
       buffers: exact, free, and it leaves the multi-GB frame cache valid.
       Feeding CLIP ImageNet-normalised pixels instead is a quiet tax that would
       present as "CLIP underperforms".
    2. **Tokens -> grid.** `forward_features` returns [B, 1+hw, D] with the CLS
       token first. We drop the `num_prefix_tokens` prefix and fold the patch
       tokens back onto their spatial grid (7x7 at 224/32), because everything
       downstream (adaptive_avg_pool2d -> pre_caps_norm -> PrimaryCaps) is
       spatial. Keeping CLS would pool a non-spatial token into the grid.

    No BatchNorm anywhere in a ViT, so unlike the frozen ResNet stem this one is
    genuinely frozen under model.train(): no running stats drift on UCF frames.
    """

    def __init__(self, ctor, pretrained=True):
        super().__init__()
        import timm
        self.vit = timm.create_model(ctor, pretrained=pretrained, num_classes=0)
        self.n_prefix = self.vit.num_prefix_tokens
        cfg = self.vit.pretrained_cfg
        m_in = torch.tensor(_PIPELINE_MEAN).view(1, 3, 1, 1)
        s_in = torch.tensor(_PIPELINE_STD).view(1, 3, 1, 1)
        m_to = torch.tensor(cfg['mean']).view(1, 3, 1, 1)
        s_to = torch.tensor(cfg['std']).view(1, 3, 1, 1)
        # x_pipeline = (p - m_in)/s_in  ->  x_vit = (p - m_to)/s_to
        self.register_buffer('renorm_scale', s_in / s_to)
        self.register_buffer('renorm_shift', (m_in - m_to) / s_to)

    def forward(self, x):
        x = x * self.renorm_scale + self.renorm_shift
        tok = self.vit.forward_features(x)[:, self.n_prefix:]     # [B, hw, D]
        B, hw, D = tok.shape
        g = int(hw ** 0.5)
        return tok.transpose(1, 2).reshape(B, D, g, g)


def _vit_stem(key, pretrained=True, out_ch=256):
    """Frozen ViT body + TRAINABLE 1x1 projection to `out_ch`.

    Same contract as `_mobilenet_stem`: the projection normalises the channel
    count so pre_caps_norm/PrimaryCaps/readout stay identical across stems and a
    stem swap remains a single-variable change.
    """
    ctor, ch = _VIT_STEMS[key]
    return _with_trainable_tail(
        nn.Sequential(_ViTGrid(ctor, pretrained=pretrained),
                      nn.Conv2d(ch, out_ch, 1)))


class SyncCapsNet(nn.Module):
    """Option B: routing-free sync readout.

    stem -> per-frame S3 grid -> PrimaryCaps -> flattened per-tick state
    z_t [B, 2304] -> PairwiseSync -> Linear. Tick = frame. No vote tensor,
    no routing iterations, no class capsules. `stem='conv4'` (default) is the
    original 4-conv backboneless stem; `stem='resnet'` swaps in the V1
    ResNet-18 body to test whether sync-as-representation stacks with depth.

    `readout='sync'` (default) is the pairwise-sync readout. `readout='linear'`
    is the B0 control: same stem/primaries/tick machinery, but each tick's
    logit is a plain `Linear(z_t)` on the flattened primaries — no pairwise
    products, no temporal decay. B1(sync) - B0(linear) isolates exactly what
    the sync statistic adds over a linear read of the same features.
    """

    def __init__(self, num_classes=11, caps_grid=3, n_synch=1024, n_self=64,
                 dropout_rate=0.3, pair_seed=0, shuffle_frames=False,
                 stem='conv4', readout='sync', pose_coupling='scalar',
                 pretrained=False, freeze_stem=False, sync_norm=False,
                 feat_cache=False, out_ch=256, lr_rank=64,
                 exclude_self=False):
        super().__init__()
        self.caps_grid = caps_grid
        self.shuffle_frames = shuffle_frames
        self.stem = stem
        self.readout = readout
        self.feat_cache = feat_cache
        _known = ({'conv4', 'resnet'} | set(_MOBILENET_STEMS) | set(_VIT_STEMS)
                  | set(_RESNET_STEMS))
        if stem not in _known:
            # Validated BEFORE the feat_cache branch: that branch's fallback is
            # nn.Identity(), so a typo'd stem name would otherwise build a
            # silently headless model instead of raising.
            raise ValueError(f'unknown stem: {stem} (expected one of '
                             f'{sorted(_known)})')
        if feat_cache:
            # The frozen body already ran offline (see
            # exp_base.FrozenFeatureDataset): inputs are POOLED [C, 3, 3]
            # features, so all that remains of the stem is OUR trainable 1x1
            # projection. Pooling and a 1x1 conv commute, so this is exact --
            # and adaptive_avg_pool2d(., 3) in forward() becomes the identity on
            # an already-3x3 input, which is why forward needs no branch.
            # Stems without a projection tail (conv4/resnet emit 256 directly)
            # have nothing trainable left in the stem at all.
            if (stem in _VIT_STEMS or stem in _MOBILENET_STEMS
                    or stem in _RESNET_STEMS):
                ch = (_VIT_STEMS[stem][1] if stem in _VIT_STEMS
                      else _RESNET_STEMS[stem][2] if stem in _RESNET_STEMS
                      else _MOBILENET_STEMS[stem][2])
                self.conv = _with_trainable_tail(
                    nn.Sequential(nn.Conv2d(ch, out_ch, 1)))
            else:
                self.conv = nn.Identity()
        elif stem == 'conv4':
            self.conv = _conv4_stem(dropout_rate)
        elif stem == 'resnet':
            self.conv = _resnet_stem(pretrained=pretrained)
        elif stem in _MOBILENET_STEMS:
            self.conv = _mobilenet_stem(stem, pretrained=pretrained, out_ch=out_ch)
        elif stem in _RESNET_STEMS:
            self.conv = _resnet_full_stem(stem, pretrained=pretrained, out_ch=out_ch)
        else:
            self.conv = _vit_stem(stem, pretrained=pretrained, out_ch=out_ch)
        # freeze_stem: direction-B "linear-probe" mode — identical frozen
        # ImageNet features into both readouts so the sync-vs-linear contrast
        # is purely the head (opt already filters requires_grad params).
        self.freeze_stem = freeze_stem
        if freeze_stem:
            for p in self.conv.parameters():
                p.requires_grad_(False)
            # The 1x1 channel projection is OURS, not pretrained — it must stay
            # trainable. `_full` MobileNet stacks emit 576-1280 channels, so a
            # FROZEN RANDOM 1x1 into 256 is a fixed random COMPRESSION that
            # discards most of the signal before the readout sees it. That bug
            # invalidated the 2026-08-16 `_full` triage runs (mnv2_full 68.99 vs
            # truncated mnv2 78.60 despite being 2.5x larger, with a 14.7-point
            # val/test gap). Truncated stems escaped it only because their
            # projections EXPAND (48/96/112 -> 256), which is rank-preserving.
            # The stem TAGS itself (see `_with_trainable_tail`) instead of this
            # site testing a stem-name dict, so a newly added stem carrying its
            # own projection is covered automatically rather than re-introducing
            # the bug the day someone forgets to extend a condition here.
            if getattr(self.conv, 'trainable_tail', False):
                for p in self.conv[-1].parameters():
                    p.requires_grad_(True)
        # `out_ch` is the PROJECTION WIDTH, i.e. how hard the stem's channels are
        # compressed before the readout. It does NOT change d_model: PrimaryCaps
        # emits 32 types x 8 dims per grid cell whatever its input width, so
        # n_synch, the sync output width and the head are all untouched. Sweeping
        # it is therefore a genuine single-variable test of whether compression
        # before the pairwise products is what suppresses the sync statistic
        # (2048->256 on r50_full is 8x; the truncated ResNet needs none).
        self.pre_caps_norm = nn.LayerNorm([out_ch, caps_grid, caps_grid])
        self.primary = PrimaryCaps(out_ch, 32, 8)
        d_model = 32 * caps_grid * caps_grid * 8          # 2304 at S3
        if readout == 'sync':
            self.sync = PairwiseSync(d_model, n_synch, n_self=n_self,
                                     seed=pair_seed, pose_coupling=pose_coupling,
                                     caps_dim=8, sync_norm=sync_norm,
                                     exclude_self=exclude_self)
            self.head = nn.Linear(self.sync.out_dim, num_classes)
        elif readout == 'cbp':
            # Compact bilinear baseline. n_synch is reused as the head WIDTH so
            # a sweep that moves n_synch moves this arm with it -- the two must
            # never drift apart, because a width-mismatched baseline is not a
            # baseline. sync_norm is forced on: the arm exists to be compared
            # against B4_gram, which is normalised, and Gao et al. apply the
            # same improved-B-CNN correction.
            self.sync = CompactBilinear(d_model, out_dim=n_synch,
                                        seed=pair_seed, sync_norm=True)
            self.head = nn.Linear(self.sync.out_dim, num_classes)
        elif readout == 'lrbp':
            # Dense low-rank bilinear reference; out_dim = rank(rank+1)/2 is
            # DERIVED, not chosen, so it is reported rather than tuned to
            # flatter the arm (rank 64 -> 2080 vs n_synch 2048).
            self.sync = LowRankBilinear(d_model, rank=lr_rank, sync_norm=True)
            self.head = nn.Linear(self.sync.out_dim, num_classes)
        elif readout == 'linear':
            self.sync = None
            self.head = nn.Linear(d_model, num_classes)
        elif readout == 'linear_ln':
            # B0 + LayerNorm control (2026-08-15). B5_concat feeds the head
            # LayerNorm(z) while B0_linear feeds raw z, so B5's deficit could be
            # the fusion OR the damaged linear branch. This arm isolates the
            # LayerNorm alone: B0_linear_ln - B0_linear is that confound's size.
            self.sync = None
            self.norm_z = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)
        elif readout == 'sync_linear':
            # First+second order fusion. Post group-aware-fix B0_linear (34.51)
            # ~ B1_sync (33.57) on UCF101/conv4, i.e. the two readouts are at
            # parity; this arm nests BOTH so the head can weight them. If it
            # beats max(B0, B1) the orders are complementary; if it matches the
            # better one, sync adds nothing over a linear read.
            # Each branch is LayerNorm'd before concat: with sync_norm on, sync
            # is L2-normalised (||.||=1) while z_t carries the raw primary-caps
            # magnitude, so an unnormalised concat would let z dominate the head
            # by scale alone and mask any sync contribution.
            self.sync = PairwiseSync(d_model, n_synch, n_self=n_self,
                                     seed=pair_seed, pose_coupling=pose_coupling,
                                     caps_dim=8, sync_norm=sync_norm,
                                     exclude_self=exclude_self)
            self.norm_sync = nn.LayerNorm(self.sync.out_dim)
            self.norm_z = nn.LayerNorm(d_model)
            self.head = nn.Linear(self.sync.out_dim + d_model, num_classes)
        else:
            raise ValueError(f'unknown readout: {readout}')

    def train(self, mode=True):
        """Keep a FROZEN stem in eval mode, so its BatchNorm uses the pretrained
        running statistics instead of per-batch ones.

        THE BUG THIS FIXES (found 2026-08-21). `freeze_stem` set
        requires_grad_(False) on the stem but never called .eval(), and
        nn.Module.train() recurses into every child -- so each epoch put the
        "frozen" ResNet's 15 BatchNorm layers back into training mode. They then
        normalised with PER-BATCH statistics at batch size 4 and mutated their
        running stats, which means the frozen features were neither frozen nor
        the pretrained ones. Because requires_grad was correctly False, every
        obvious check of "is the stem frozen?" passed.

        Measured cost on UCF101 official split-1, frozen ResNet-18 layer3:
        B0_linear 63.27 -> 65.72 and B4_syncnorm 69.78 -> 71.57 once the stem is
        held in eval mode. It also silently split the backbone ladder in two,
        since feature-cached stems were precomputed under body.eval() and so
        always used running statistics -- the cached and uncached paths were
        computing different things while being reported in one table.

        LayerNorm/ViT stems are unaffected (no batch statistics), as are
        from-scratch stems, which are not frozen and SHOULD train their BN.

        SYNC_FROZEN_BN_TRAIN=1 restores the legacy behaviour, because every
        pre-2026-08-21 uncached frozen-ResNet number in the ledger was produced
        that way and must stay reproducible.
        """
        super().train(mode)
        if (mode and self.freeze_stem
                and not os.environ.get('SYNC_FROZEN_BN_TRAIN')):
            self.conv.eval()
            # our own 1x1 projection is trainable and stays in train mode
            if getattr(self.conv, 'trainable_tail', False):
                self.conv[-1].train(mode)
            assert_frozen_bn_eval(self)
        return self

    def forward(self, x):
        B, T, C, H, W = x.shape
        if self.shuffle_frames:
            x = x[:, torch.randperm(T, device=x.device)]
        f = self.conv(x.reshape(B * T, C, H, W))
        f = F.adaptive_avg_pool2d(f, self.caps_grid)
        f = self.pre_caps_norm(f)
        u = self.primary(f)                                # [B*T, N, 8]
        z_seq = u.reshape(B, T, -1).float()                # fp32 for sync
        alpha = beta = None
        logits, certainties = [], []
        for t in range(T):
            if self.readout == 'sync_linear':
                sync, alpha, beta = self.sync(z_seq[:, t], alpha, beta)
                logit = self.head(torch.cat(
                    (self.norm_sync(sync), self.norm_z(z_seq[:, t])), dim=1))
            elif self.sync is not None:
                sync, alpha, beta = self.sync(z_seq[:, t], alpha, beta)
                logit = self.head(sync)
            elif self.readout == 'linear_ln':
                logit = self.head(self.norm_z(z_seq[:, t]))
            else:
                logit = self.head(z_seq[:, t])             # B0 linear control
            ne = normalized_entropy(logit)
            logits.append(logit)
            certainties.append(torch.stack((ne, 1 - ne), dim=1))
        return dict(logits=torch.stack(logits, dim=-1),
                    certainties=torch.stack(certainties, dim=-1))


class SyncTRCapsNet(nn.Module):
    """Option A: TR-Caps recurrence kept; logits from full pairwise sync over
    the flattened consensus states v_t [B, C*16] (replaces tau*||v|| + lam*sync).
    Same 4-conv stem as SyncCapsNet (backboneless comparability with Option B).
    """

    def __init__(self, num_classes=11, caps_grid=3, w_init_scale=20.0,
                 dropout_rate=0.3, pair_seed=0, shuffle_frames=False):
        super().__init__()
        self.caps_grid = caps_grid
        self.shuffle_frames = shuffle_frames
        proto = SyncCapsNet(num_classes=num_classes, caps_grid=caps_grid,
                            n_synch=1, n_self=0, dropout_rate=dropout_rate)
        self.conv = proto.conv
        self.pre_caps_norm = proto.pre_caps_norm
        self.primary = proto.primary
        self.routing = TemporalRoutingCaps(
            in_capsules=32 * caps_grid * caps_grid, in_dim=8,
            num_classes=num_classes, out_dim=16,
            w_init_scale=w_init_scale, use_sync=False)
        d = num_classes * 16
        self.sync = PairwiseSync(d, mode='full', seed=pair_seed)
        self.head = nn.Linear(self.sync.out_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        if self.shuffle_frames:
            x = x[:, torch.randperm(T, device=x.device)]
        f = self.conv(x.reshape(B * T, C, H, W))
        f = F.adaptive_avg_pool2d(f, self.caps_grid)
        f = self.pre_caps_norm(f)
        u = self.primary(f).view(B, T, -1, 8)
        v_states = self.routing(u)['v_states']            # [B, C, 16, T]
        z_seq = v_states.reshape(B, -1, T).float()
        alpha = beta = None
        logits, certainties = [], []
        for t in range(T):
            sync, alpha, beta = self.sync(z_seq[:, :, t], alpha, beta)
            logit = self.head(sync)
            ne = normalized_entropy(logit)
            logits.append(logit)
            certainties.append(torch.stack((ne, 1 - ne), dim=1))
        return dict(logits=torch.stack(logits, dim=-1),
                    certainties=torch.stack(certainties, dim=-1))


class DynamicRoutingCaps(nn.Module):
    """Standard dynamic routing (Sabour et al., NeurIPS 2017), r iterations.

    WHY THIS ARM EXISTS (2026-08-20). Section 6.2 of the paper prices routing
    ANALYTICALLY, from tensor shapes, and says so explicitly: "these routing
    figures are analytic estimates ... because the models contain no routing
    stage to profile". This class is that stage. Same stem, same primary
    capsules, same per-tick loss and certainty machinery as every other arm, so
    the difference measured against B4 is the readout and nothing else.

    Deliberately TEXTBOOK, not this project's variant: the log-priors b are
    reset to zero at the start of every tick, i.e. routing is solved
    independently per frame. TemporalRoutingCaps (the A1 arm) instead carries
    the consensus across ticks and adds a learned decay; that is our own
    contribution and would be a strawman standing in for "standard routing".

    Parameter cost is the point, not an accident: the vote tensor is
    in_capsules x num_classes x out_dim x in_dim, which at 288 x 101 x 16 x 8
    is 3.72 M trainable parameters against a 2048-wide sync head's 0.21 M.
    """

    def __init__(self, in_capsules, in_dim=8, num_classes=11, out_dim=16,
                 r=3, w_init=0.01, w_init_scale=20.0):
        super().__init__()
        self.num_classes, self.out_dim, self.r = num_classes, out_dim, r
        # w_init * w_init_scale is the std. The scale MUST be probed per
        # (grid, num_classes): at 101 classes an under-scaled init leaves every
        # class capsule at the same tiny norm, the softmax over classes is
        # uniform, and the loss parks at ln(101) = 4.615 with dead capsules.
        self.W = nn.Parameter(w_init * w_init_scale *
                              torch.randn(1, in_capsules, num_classes,
                                          out_dim, in_dim))

    @staticmethod
    def squash(v, eps=1e-8):
        n = v.norm(dim=-1, keepdim=True)
        return (n ** 2 / (1 + n ** 2)) * v / (n + eps)

    def forward(self, u):
        """u: [B, N, in_dim] primary capsules for ONE tick. Returns s [B, C, D],
        the pre-squash routed evidence, so the caller owns tick accumulation."""
        B, N, _ = u.shape
        u_hat = torch.matmul(self.W.expand(B, -1, -1, -1, -1),
                             u.unsqueeze(2).unsqueeze(-1)).squeeze(-1)  # [B,N,C,D]
        b = u.new_zeros(B, N, self.num_classes)      # log-priors, reset per tick
        for i in range(self.r):
            c = F.softmax(b, dim=2)                  # over CLASS capsules
            # einsum keeps both contractions fused: materialising c*u_hat would
            # retain a [B,N,C,D] tensor per iteration per tick for backward,
            # ~360 MB at B=4/T=16/r=3, which is the difference between this arm
            # coexisting with the others on an 8 GB card and an OOM.
            s = torch.einsum('bnc,bncd->bcd', c, u_hat)
            if i < self.r - 1:
                v = self.squash(s)
                b = b + torch.einsum('bncd,bcd->bnc', u_hat, v)
        return s


class RoutingCapsNet(nn.Module):
    """Same-stem dynamic-routing control for the SyncCaps readout matrix.

    Stem, pre-caps LayerNorm and PrimaryCaps are TAKEN FROM a SyncCapsNet
    prototype rather than rebuilt, so the features entering the routing head
    are constructed by exactly the same code path (including the frozen /
    feature-cached branches) as the features entering PairwiseSync. Rebuilding
    them here would be one refactor away from an unnoticed mismatch, and a
    baseline on subtly different features answers no question.

    Tick aggregation matches the second-order baselines: the routed evidence
    s_t is accumulated with r = 1 and count-normalised, s_acc/sqrt(t+1), before
    the squash. So routing gets the SAME temporal treatment as B4_gram, and the
    contrast is the readout mechanism alone rather than a temporal handicap.
    """

    def __init__(self, num_classes=11, caps_grid=3, r=3, tau=10.0,
                 w_init_scale=20.0, out_dim=16, dropout_rate=0.3, **stem_kw):
        super().__init__()
        proto = SyncCapsNet(num_classes=num_classes, caps_grid=caps_grid,
                            readout='linear', dropout_rate=dropout_rate,
                            **stem_kw)
        self.conv = proto.conv
        self.pre_caps_norm = proto.pre_caps_norm
        self.primary = proto.primary
        self.caps_grid = caps_grid
        self.tau = tau
        # Inherited along with the stem: this arm takes proto's frozen conv, so
        # it must re-pin it to eval exactly as SyncCapsNet.train does, or the
        # routing baseline would train against different features than the arms
        # it is being compared with.
        self.freeze_stem = proto.freeze_stem
        self.routing = DynamicRoutingCaps(
            32 * caps_grid * caps_grid, in_dim=8, num_classes=num_classes,
            out_dim=out_dim, r=r, w_init_scale=w_init_scale)

    def train(self, mode=True):
        """Same frozen-BatchNorm discipline as SyncCapsNet.train (see its
        docstring for the bug this fixes). Defined here rather than borrowed
        from SyncCapsNet, because that method's zero-argument super() binds to
        SyncCapsNet and raises on a RoutingCapsNet instance. This arm takes its
        stem from a SyncCapsNet prototype, so without this the routing baseline
        would train against batch-statistic features while the arms it is
        compared against used the pretrained running statistics.
        """
        super().train(mode)
        if (mode and self.freeze_stem
                and not os.environ.get('SYNC_FROZEN_BN_TRAIN')):
            self.conv.eval()
            if getattr(self.conv, 'trainable_tail', False):
                self.conv[-1].train(mode)
            assert_frozen_bn_eval(self)
        return self

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.conv(x.reshape(B * T, C, H, W))
        f = F.adaptive_avg_pool2d(f, self.caps_grid)
        f = self.pre_caps_norm(f)
        u_seq = self.primary(f).view(B, T, -1, 8).float()
        s_acc = None
        logits, certainties = [], []
        for t in range(T):
            s = self.routing(u_seq[:, t])
            s_acc = s if s_acc is None else s_acc + s
            v = DynamicRoutingCaps.squash(s_acc / math.sqrt(t + 1))
            logit = self.tau * v.norm(dim=-1)
            ne = normalized_entropy(logit)
            logits.append(logit)
            certainties.append(torch.stack((ne, 1 - ne), dim=1))
        return dict(logits=torch.stack(logits, dim=-1),
                    certainties=torch.stack(certainties, dim=-1))
