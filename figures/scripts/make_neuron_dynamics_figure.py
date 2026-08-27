"""CTM Fig-2a-style neural-dynamics panel for SyncCaps.

Runs the released B1 sync checkpoint (synccaps_ucf11_conv4_B1_sync_seed42.pt)
on a real UCF-11 clip and captures, per internal tick t = 1..16:
  * z_t[d]        raw primary-capsule neuron activations  (2304-dim)
  * sync_{k,t}    the synchronisation statistic a/sqrt(b) (1024-dim) that the
                  paper argues IS the representation (SyncCaps section 3.3)
Then draws a grid of random-colored single-unit traces, mirroring CTM Fig 2a.

Usage:  python figures/scripts/make_neuron_dynamics_figure.py [outdir]
"""
import glob, os, sys, cv2, numpy as np, torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, ".")
from src.models.sync_caps import SyncCapsNet

# learned-decay colormap: blue = order-agnostic (rho~0) -> red = recency-weighted
RHO_CMAP = plt.cm.coolwarm
CELL = 256          # neurons per 3x3 grid cell (d // 256 == cell, row-major)

# --- figure configuration ----------------------------------------------------
# 2026-08-18: the original figures were built on the UCF-11 / 4-conv / B1_sync
# checkpoint under the RETIRED clip-level split, whose numbers are invalid (see
# RUN-LEDGER, "The split boundary"). The paper's figures move to the corrected
# headline cell: UCF101 official split-1, FROZEN ImageNet ResNet-18 -> layer3,
# B4_syncnorm, seed 42. The legacy config is kept selectable so the superseded
# artefact can still be reproduced for the record rather than silently lost.
#
#   SYNCCAPS_FIG_CFG=legacy   python figures/scripts/make_neuron_dynamics_figure.py
#
# The capsule geometry is IDENTICAL across configs -- caps_grid=3 and
# PrimaryCaps(256,32,8) give d_model=2304 and CELL=256 for every stem -- so the
# 3x3 spatial arc layout of the web figures carries over unchanged.
UCF101_ROOT = "UCF101_full"

# 2026-08-26: the "headline" config below is PRE-CORRECTION. It loads the
# uncached frozen-ResNet checkpoint trained before the frozen-BatchNorm fix of
# 2026-08-21 (RUN-LEDGER Phase 14/15), whose stem ran on per-batch statistics at
# batch size 4. Its seed-42 accuracy is 70.10 against 71.82 for the corrected
# run, so every figure drawn from it is superseded. `headline_fc` is the
# corrected default.
#
# The corrected checkpoint was trained feature-cached, so its state_dict has NO
# conv.* keys at all: for the truncated `resnet` stem (256 channels native) the
# cached branch sets self.conv = nn.Identity(), leaving nothing trainable in the
# stem. Its head therefore transplants into an UNCACHED model whose ResNet body
# is plain ImageNet weights -- which is exactly what synccaps_precompute_stem.py
# ran (under body.eval()) to build the cache. Verified 2026-08-26 on 300 test
# clips: 100.00% prediction agreement between the cached and uncached forward,
# max |delta logit| 3.2e-3, i.e. the TF32 convolution floor (ledger trap #7).
# That is also the acceptance check the v5 repair plan's section 1 asks for.
CONFIGS = {
    "headline_fc": dict(
        ckpt="checkpoints/synccaps_ucf101_resnet_ptfz_official1_noval_fc_B4_syncnorm_seed42.pt",
        num_classes=101, n_synch=2048, n_self=64, stem="resnet",
        pretrained=True, freeze_stem=True, sync_norm=True,
        # official split-1 TEST groups are g01-g07: a figure must never be drawn
        # on a clip the model trained on.
        clip_glob=UCF101_ROOT + "/{cls}/v_{cls}_g0[1-7]_c*.avi",
        clips={"Diving":     UCF101_ROOT + "/Diving/v_Diving_g01_c01.avi",
               "Basketball": UCF101_ROOT + "/Basketball/v_Basketball_g01_c01.avi"},
        # same eleven activity concepts as the original UCF-11 panel, so the
        # corrected figure stays visually comparable to the superseded one
        web_classes=["Basketball", "Biking", "Diving", "GolfSwing", "HorseRiding",
                     "SoccerJuggling", "Swing", "TennisSwing", "TrampolineJumping",
                     "VolleyballSpiking", "WalkingWithDog"],
        subtitle="SyncCaps B4_syncnorm | UCF101 official split-1 (test groups g01-g07) | "
                 "frozen ResNet-18 stem | one clip per class",
        arm_label="B4_syncnorm", dset_label="UCF101 split-1",
    ),
    "headline": dict(
        ckpt="checkpoints/synccaps_ucf101_resnet_ptfz_official1_noval_B4_syncnorm_seed42.pt",
        num_classes=101, n_synch=2048, n_self=64, stem="resnet",
        pretrained=True, freeze_stem=True, sync_norm=True,
        clip_glob=UCF101_ROOT + "/{cls}/v_{cls}_g0[1-7]_c*.avi",
        clips={"Diving":     UCF101_ROOT + "/Diving/v_Diving_g01_c01.avi",
               "Basketball": UCF101_ROOT + "/Basketball/v_Basketball_g01_c01.avi"},
        web_classes=["Basketball", "Biking", "Diving", "GolfSwing", "HorseRiding",
                     "SoccerJuggling", "Swing", "TennisSwing", "TrampolineJumping",
                     "VolleyballSpiking", "WalkingWithDog"],
        subtitle="SyncCaps B4_syncnorm | UCF101 official split-1 | frozen ResNet-18 "
                 "| PRE-BATCHNORM-FIX -- superseded",
        arm_label="B4_syncnorm", dset_label="UCF101 split-1 (pre-fix)",
    ),
    "legacy": dict(
        ckpt="checkpoints/synccaps_ucf11_conv4_B1_sync_seed42.pt",
        num_classes=11, n_synch=1024, n_self=64, stem="conv4",
        pretrained=False, freeze_stem=False, sync_norm=False,
        clip_glob="UCF11_updated_mpg/{cls}/**/*.mpg",
        clips={"diving":     "UCF11_updated_mpg/diving/v_diving_08/v_diving_08_03.mpg",
               "basketball": "UCF11_updated_mpg/basketball/v_shooting_08/v_shooting_08_01.mpg"},
        web_classes=['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding',
                     'soccer_juggling', 'swing', 'tennis_swing', 'trampoline_jumping',
                     'volleyball_spiking', 'walking'],
        subtitle="SyncCaps B1 | UCF-11 | RETIRED clip-level split -- superseded",
        arm_label="B1", dset_label="UCF-11 (RETIRED split)",
    ),
}
CFG = CONFIGS[os.environ.get("SYNCCAPS_FIG_CFG", "headline_fc")]
CKPT = CFG["ckpt"]
CLIPS = CFG["clips"]
SEQ_LEN, FPS, SIZE = 16, 5.0, (224, 224)

_clean_tf = T.Compose([
    T.ToPILImage(), T.Resize(SIZE), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _sequential_fallback(path, last):
    """Seek-hostile container: on some UCF-11 .mpg files ANY POS_FRAMES seek makes
    every subsequent read() fail, while a plain sequential read returns every
    frame. Applied only when the seeking read came back empty -- seeking perturbs
    MPEG-1 decode slightly, so switching unconditionally would change every
    published figure. See exp_base._load_video for the same guard."""
    cap = cv2.VideoCapture(path)
    raw, cur = {}, 0
    while cur <= last:
        ok, fr = cap.read()
        if not ok:
            break
        raw[cur] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        cur += 1
    cap.release()
    return raw


def load_clip(path):
    """Replicates exp_base.UCF11VideoDataset clean-transform path."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / FPS)))
    anchors = [min(i * step, total - 1) for i in range(SEQ_LEN)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, anchors[0])
    raw, cur, last = {}, anchors[0], anchors[-1]
    while cur <= last:
        ok, fr = cap.read()
        if ok:
            raw[cur] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        cur += 1
    cap.release()
    raw = raw or _sequential_fallback(path, last)
    frames = [_clean_tf(raw[min(raw, key=lambda k: abs(k - a))]) for a in anchors]
    return torch.stack(frames)                      # [16,3,224,224]


def build_model():
    m = SyncCapsNet(num_classes=CFG["num_classes"], caps_grid=3,
                    n_synch=CFG["n_synch"], n_self=CFG["n_self"],
                    readout="sync", stem=CFG["stem"], pose_coupling="scalar",
                    pretrained=CFG["pretrained"], freeze_stem=CFG["freeze_stem"],
                    sync_norm=CFG["sync_norm"])
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict(sd, strict=False)             # is_self buffer is non-persistent
    m.eval()
    return m


@torch.no_grad()
def capture(model, clip):
    """Returns z_traces [T,2304], sync_traces [T,1024], logits [T,11]."""
    x = clip.unsqueeze(0)                           # [1,16,3,224,224]
    B, Tn, C, H, W = x.shape
    f = model.conv(x.reshape(B * Tn, C, H, W))
    f = F.adaptive_avg_pool2d(f, model.caps_grid)
    f = model.pre_caps_norm(f)
    u = model.primary(f)
    z_seq = u.reshape(B, Tn, -1).float()            # [1,16,2304]
    alpha = beta = None
    syncs, logits = [], []
    for t in range(Tn):
        sync, alpha, beta = model.sync(z_seq[:, t], alpha, beta)
        syncs.append(sync[0].clone())
        logits.append(model.head(sync)[0].clone())
    return (z_seq[0].numpy(),
            torch.stack(syncs).numpy(),
            torch.stack(logits).numpy())



def select_clip(model, cls, configured=None, max_tries=10):
    """Pick a CORRECTLY-CLASSIFIED clip for `cls`, reproducibly.

    Both figure captions claim a correctly-classified clip, so the choice must be
    a rule rather than a hand-pick: keep `configured` if the model gets it right,
    else scan the class's other test-group clips in sorted order and take the
    first correct one. Returns (path, correct); when nothing is correct the
    caller MUST NOT caption the panel as correct.
    """
    tgt = CLASS_NAMES.index(cls) if cls in CLASS_NAMES else None
    cands = ([configured] if configured else []) + [
        v for v in sorted(glob.glob(CFG["clip_glob"].format(cls=cls), recursive=True))
        if v != configured][:max_tries]
    for v in cands:
        _, _, lg = capture(model, load_clip(v))
        if tgt is None or int(lg.mean(0).argmax()) == tgt:
            return v, True
    return (cands[0] if cands else configured), False


def load_display_frames(path):
    """Same anchors as load_clip, but raw RGB uint8 224x224 for backdrops."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / FPS)))
    anchors = [min(i * step, total - 1) for i in range(SEQ_LEN)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, anchors[0])
    raw, cur, last = {}, anchors[0], anchors[-1]
    while cur <= last:
        ok, fr = cap.read()
        if ok:
            raw[cur] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        cur += 1
    cap.release()
    raw = raw or _sequential_fallback(path, last)
    return [cv2.resize(raw[min(raw, key=lambda k: abs(k - a))], SIZE)
            for a in anchors]


# Dataset class order is sorted(listdir), matching UCF11VideoDataset.
CLASS_NAMES = (sorted(os.listdir(UCF101_ROOT)) if CFG["num_classes"] == 101
               else ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding',
               'soccer_juggling', 'swing', 'tennis_swing', 'trampoline_jumping',
               'volleyball_spiking', 'walking'])
# Read from the checkpoint's own stored results, so the annotations cannot drift
# from the model actually being plotted. The hardcoded pair that used to live
# here (6.55, 87.6) belonged to the RETIRED UCF-11 checkpoint.
#
# ⚠ These are SINGLE-VIEW numbers, matching the single-window forward pass the
# figure itself runs. And the exit tick is quoted beside the HYBRID accuracy,
# never the certain-tick one: certain-tick inspects all T ticks and never stops,
# so "certain-tick exit" (the old label on this line) is a category error. Only
# Algorithm 1's hybrid policy defines an exit.
_R = torch.load(CKPT, map_location="cpu", weights_only=False).get("results", {})
MEAN_EXIT = float(_R.get("mean_exit_tick", float("nan")))
ACC_CERTAIN = float(_R.get("test_acc_certain", float("nan")))
ACC_HYBRID = float(_R.get("test_acc_hybrid", float("nan")))
HEADER = (f"SyncCaps {CFG['arm_label']}  ·  {CFG['dset_label']}  ·  "
          f"{ACC_CERTAIN:.1f} certain / {ACC_HYBRID:.1f} hybrid  (single view)")
INK, MUT, GRID = "#1b2733", "#5b6b78", "#dfe3e8"


def _softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def polished(name, sync, logit, out, nrow=8, ncol=16, seed=3):
    """CTM Fig-2a-style grid of sync-unit dynamics + prediction-forming ribbon."""
    Tn = sync.shape[0]
    rng = np.random.default_rng(seed)
    var = sync.var(0)                                   # drop flattest quartile
    active = np.where(var > np.percentile(var, 25))[0]
    idx = rng.choice(active, nrow * ncol, replace=False)
    colors = plt.cm.hsv(rng.permutation(np.linspace(0, 1, nrow * ncol)))

    fig = plt.figure(figsize=(13.2, 8.4), dpi=200)
    outer = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.05], hspace=0.16,
                             left=0.05, right=0.985, top=0.885, bottom=0.085)
    grid = outer[0].subgridspec(nrow, ncol, hspace=0.32, wspace=0.20)
    ticks = np.arange(1, Tn + 1)
    for a_i, (k, c) in enumerate(zip(idx, colors)):
        ax = fig.add_subplot(grid[a_i // ncol, a_i % ncol])
        ax.plot(ticks, sync[:, k], color=c, lw=1.0, solid_capstyle="round")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID); s.set_linewidth(0.6)

    # ---- prediction-forming ribbon -----------------------------------------
    rib = fig.add_subplot(outer[1])
    probs = _softmax(logit)                             # [T,11]
    win = int(logit.mean(0).argmax())
    for c in range(probs.shape[1]):
        if c == win:
            continue
        rib.plot(ticks, probs[:, c], color="#c9ced5", lw=1.0, zorder=2)
    rib.plot(ticks, probs[:, win], color="#c0392b", lw=2.6, zorder=4)
    rib.axvline(MEAN_EXIT, ls="--", color="#e0a020", lw=1.8, zorder=3)
    rib.text(MEAN_EXIT + 0.15, 0.93,
             f"mean hybrid exit, Alg. 1  ({MEAN_EXIT:.1f} / 16)",
             color="#b07d10", fontsize=9, va="top", ha="left")
    rib.text(Tn, probs[-1, win] + 0.02, f"p({CLASS_NAMES[win]})",
             color="#c0392b", fontsize=10, va="bottom", ha="right", fontweight="bold")
    rib.set_xlim(1, Tn); rib.set_ylim(0, 1.0)
    rib.set_xticks(range(2, Tn + 1, 2))
    rib.set_yticks([0, 0.5, 1.0])
    rib.set_xlabel("internal tick  $t$  (frame = tick)", fontsize=10, color=INK)
    rib.set_ylabel("class prob.", fontsize=10, color=INK)
    rib.tick_params(labelsize=8.5, colors=MUT)
    for sp in ("top", "right"):
        rib.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        rib.spines[sp].set_color(GRID)

    # ---- titles -------------------------------------------------------------
    fig.text(0.05, 0.955, "Neural synchronisation dynamics",
             fontsize=18, fontweight="bold", ha="left", color=INK)
    fig.text(0.985, 0.957, HEADER,
             fontsize=10.5, ha="right", va="center", color=MUT)
    fig.text(0.05, 0.917,
             f"one {name} clip  ·  each panel is one synchronising unit  "
             r"$\mathrm{sync}_{k,t}=\alpha_{k,t}/\sqrt{\beta_{k,t}}$"
             f"  over ticks $t=1\\ldots16$  ·  {nrow*ncol} of {CFG['n_synch']} units  ·  "
             "colours arbitrary",
             fontsize=11, ha="left", color=MUT)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", out, "(.png/.pdf)")


def grid_by_decay(name, sync, rho, out, nrow=8, ncol=16, seed=3):
    """CTM-2a grid, traces coloured by learned decay rho, panels ordered by rho."""
    Tn = sync.shape[0]
    rng = np.random.default_rng(seed)
    var = sync.var(0)
    active = np.where(var > np.percentile(var, 25))[0]
    idx = rng.choice(active, nrow * ncol, replace=False)
    idx = idx[np.argsort(rho[idx])]                    # lay out low -> high rho
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=float(rho.max()))
    ticks = np.arange(1, Tn + 1)

    fig = plt.figure(figsize=(13.2, 8.4), dpi=200)
    outer = fig.add_gridspec(2, 1, height_ratios=[10, 0.5], hspace=0.14,
                             left=0.05, right=0.985, top=0.885, bottom=0.085)
    grid = outer[0].subgridspec(nrow, ncol, hspace=0.32, wspace=0.20)
    for a_i, k in enumerate(idx):
        ax = fig.add_subplot(grid[a_i // ncol, a_i % ncol])
        ax.plot(ticks, sync[:, k], color=RHO_CMAP(norm(rho[k])), lw=1.15,
                solid_capstyle="round")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID); s.set_linewidth(0.6)

    cax = fig.add_subplot(outer[1])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=RHO_CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_label(r"learned decay  $\rho$   (0 = order-agnostic accumulator "
                 r"$\to$  larger = recency-weighted / order-sensitive)",
                 fontsize=10.5, color=INK)
    cb.ax.tick_params(labelsize=8.5, colors=MUT)

    fig.text(0.05, 0.955, "Synchronisation dynamics coloured by learned decay",
             fontsize=18, fontweight="bold", ha="left", color=INK)
    fig.text(0.985, 0.957, HEADER,
             fontsize=10.5, ha="right", va="center", color=MUT)
    fig.text(0.05, 0.917,
             f"one {name} clip  ·  {nrow*ncol} of 1024 synchronising units, "
             r"$\mathrm{sync}_{k,t}=\alpha/\sqrt{\beta}$"
             r"  ·  ordered by $\rho$  ·  97% of units learn $\rho<0.05$ "
             "(near order-agnostic)",
             fontsize=11, ha="left", color=MUT)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", out, "(.png/.pdf)")


def overlay_on_frame(name, sync, rho, left, frames, out, tick=8, per_cell=14, seed=5):
    """Overlay each grid cell's sync traces (coloured by rho) on the frame region
    the cell encodes. Pairs anchored at their left neuron's cell (d // 256)."""
    Tn = sync.shape[0]
    cell = left // CELL                                # [1024] in 0..8
    rng = np.random.default_rng(seed)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=float(rho.max()))
    ticks = np.arange(1, Tn + 1)

    fig = plt.figure(figsize=(9.6, 10.2), dpi=200)
    fx, fy, fw, fh = 0.04, 0.055, 0.92, 0.80          # frame axes (fig fraction)
    axf = fig.add_axes([fx, fy, fw, fh])
    axf.imshow(frames[tick - 1], aspect="auto"); axf.axis("off")
    axf.imshow(np.ones((*SIZE, 4)) * [1, 1, 1, 0.28], aspect="auto")  # dim frame

    mw, mh = fw / 3 * 0.86, fh / 3 * 0.86
    for c in range(9):
        gy, gx = divmod(c, 3)
        cx = fx + (gx + 0.5) / 3 * fw
        cy = fy + (1 - (gy + 0.5) / 3) * fh
        ax = fig.add_axes([cx - mw / 2, cy - mh / 2, mw, mh])
        pool = np.where(cell == c)[0]
        pick = pool[np.argsort(sync[:, pool].var(0))[-per_cell:]]  # liveliest
        for k in pick:
            ax.plot(ticks, sync[:, k], color=RHO_CMAP(norm(rho[k])),
                    lw=1.1, alpha=0.92, solid_capstyle="round")
        ax.set_xticks([]); ax.set_yticks([])
        ax.patch.set_facecolor("white"); ax.patch.set_alpha(0.55)
        for s in ax.spines.values():
            s.set_edgecolor("#2e7d5b"); s.set_linewidth(1.4); s.set_alpha(0.9)

    # colorbar
    cax = fig.add_axes([fx + 0.15, 0.028, fw - 0.30, 0.016])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=RHO_CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_label(r"learned decay  $\rho$   (blue = order-agnostic  $\to$  "
                 r"red = recency-weighted)", fontsize=9.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUT)

    fig.text(fx, 0.975, "Synchronisation traces overlaid on the frame grid",
             fontsize=16.5, fontweight="bold", ha="left", color=INK)
    fig.text(fx, 0.945,
             f"SyncCaps {CFG['arm_label']} · {CFG['dset_label']} · one {name} clip "
             f"(frame at tick {tick}/16)  ·  "
             r"each 3×3 panel = its capsule-grid cell's $\mathrm{sync}_{k,t}$ "
             f"traces ({per_cell} liveliest units, coloured by $\\rho$)",
             fontsize=10, ha="left", color=MUT)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", out, "(.png/.pdf)")


def overlay_arcs(name, sync, z, rho, left, right, frames, out, tick=8,
                 n_web=70, n_hot=14, seed=7):
    """CTM Fig-2b-style web: arcs between 3x3 grid cells on one frame, coloured
    by learned decay rho. Arc = a synchronising pair (left->right neuron cells);
    width/alpha = |sync| at the final tick; node size = regional activation."""
    Tn = sync.shape[0]
    ci, cj = left // CELL, right // CELL             # each pair's two grid cells
    mag = np.abs(sync[-1])                            # settled sync strength
    zt = np.abs(z[-1]).reshape(9, CELL).mean(1)       # per-cell activation
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=float(rho.max()))
    rng = np.random.default_rng(seed)

    cross = np.where(ci != cj)[0]
    web = cross[np.argsort(mag[cross])[-n_web:]]      # strongest cross couplings
    hot = cross[np.argsort(rho[cross])[-n_hot:]]      # rare recency-weighted ones
    mmax = mag[web].max() + 1e-9

    def cell_xy(c):                                   # image coords (224x224)
        gy, gx = divmod(int(c), 3)
        return (gx + 0.5) / 3 * SIZE[1], (gy + 0.5) / 3 * SIZE[0]

    fig = plt.figure(figsize=(9.6, 10.0), dpi=200)
    axf = fig.add_axes([0.03, 0.055, 0.94, 0.83])
    axf.imshow(frames[tick - 1], aspect="auto")
    axf.imshow(np.ones((*SIZE, 4)) * [1, 1, 1, 0.45], aspect="auto")  # dim frame
    axf.set_xlim(0, SIZE[1]); axf.set_ylim(SIZE[0], 0); axf.axis("off")

    def draw(ks, hot_layer):
        for k in ks:
            xi, yi = cell_xy(ci[k]); xj, yj = cell_xy(cj[k])
            w = mag[k] / mmax
            rad = 0.20 * np.sign(rng.uniform(-1, 1)) + rng.uniform(-0.12, 0.12)
            col = RHO_CMAP(norm(rho[k]))
            lw = (1.6 + 3.0 * w) if hot_layer else (0.5 + 2.6 * w)
            al = 0.97 if hot_layer else (0.20 + 0.55 * w)
            p = FancyArrowPatch((xi, yi), (xj, yj), arrowstyle="-",
                                connectionstyle=f"arc3,rad={rad}",
                                color=col, lw=lw, alpha=al,
                                zorder=5 if hot_layer else 4,
                                capstyle="round")
            axf.add_patch(p)

    draw(web, False)
    draw(hot, True)                                   # recency arcs on top
    for c in range(9):                                # nodes
        x, y = cell_xy(c)
        s = 120 + 900 * (zt[c] / (zt.max() + 1e-9))
        axf.scatter([x], [y], s=s, color="#2e7d5b", edgecolor="white",
                    linewidth=1.6, zorder=6)

    cax = fig.add_axes([0.18, 0.03, 0.64, 0.016])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=RHO_CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_label(r"learned decay  $\rho$   (blue = order-agnostic  $\to$  "
                 r"red = recency-weighted; thick red arcs highlighted)",
                 fontsize=9.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUT)

    fig.text(0.03, 0.965, "Synchronisation web on the capsule grid",
             fontsize=17, fontweight="bold", ha="left", color=INK)
    fig.text(0.03, 0.935,
             f"SyncCaps {CFG['arm_label']} · {CFG['dset_label']} · one {name} clip "
             f"(frame at tick {tick}/16)  ·  "
             f"each arc = one synchronising pair between two grid cells  ·  "
             r"node size $\propto$ regional activation, arc width $\propto$ "
             r"$|\mathrm{sync}|$",
             fontsize=9.5, ha="left", color=MUT)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", out, "(.png/.pdf)")


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    model = build_model()
    rho = model.sync.rho.detach().numpy()
    left = model.sync.left.numpy()
    right = model.sync.right.numpy()
    for name, path in CLIPS.items():
        # The figure caption claims a CORRECTLY-CLASSIFIED clip, so make that
        # selection reproducible instead of hand-picked: keep the configured
        # clip if the model gets it right, otherwise scan the class's other
        # test-group clips in sorted order and take the first correct one. If
        # none is correct, fall back and SAY SO rather than captioning a
        # misclassified clip as correct.
        path, correct = select_clip(model, name, configured=path)
        clip = load_clip(path)
        z, sync, logit = capture(model, clip)
        frames = load_display_frames(path)
        pred = int(logit.mean(0).argmax())
        flag = "correct" if correct else "MISCLASSIFIED - caption must not say correct"
        print(f"[{name}] {os.path.basename(path)} sync {sync.shape} -> "
              f"pred {CLASS_NAMES[pred]}  [{flag}]")
        np.savez(f"{outdir}/dyn_{name}.npz", z=z, sync=sync, logit=logit)
        polished(name, sync, logit, f"{outdir}/fig_neuron_dynamics_{name}")
        grid_by_decay(name, sync, rho, f"{outdir}/fig_dynamics_by_decay_{name}")
        overlay_on_frame(name, sync, rho, left, frames,
                         f"{outdir}/fig_sync_overlay_{name}")
        overlay_arcs(name, sync, z, rho, left, right, frames,
                     f"{outdir}/fig_sync_web_{name}")
