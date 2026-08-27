"""
SyncCaps Fig. (circuits-and-systems): a *fixed* linear read-out W applied
per tick, overlaid on the real video-frame (=tick) sequence.

Message (paper section 8): dynamic routing -> replaced by the SAME fixed
linear map stamped identically on every frame. Straight-line, statically
schedulable, constant MACs/tick, no data-dependent control flow.
"""
import os
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---- palette ----------------------------------------------------------------
C_ENC   = "#3b6ea5"   # encoder / frame border (blue)
C_W     = "#2e7d5b"   # read-out W (teal-green)
C_RED   = "#c0392b"   # removed routing
INK     = "#1b2733"
MUT     = "#5b6b78"
BAND    = "#fbfaf0"   # banner fill

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK, "axes.edgecolor": INK,
})

N_SHOW   = 6
TICK_IDS = [int(round(v)) for v in np.linspace(1, 16, N_SHOW)]   # 1,4,7,10,13,16


def load_frames(path, n=N_SHOW):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def rrect(ax, x, y, w, h, fc, ec, lw=1.4, alpha=1.0, rad=0.012, ls="-", z=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, ls=ls,
                       mutation_aspect=1.0, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=INK, lw=1.6, z=4, style="-|>", ms=9, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        lw=lw, color=color, zorder=z, ls=ls,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def build(clip_path, action, out_path):
    frames = load_frames(clip_path)
    FIG_W, FIG_H = 13.6, 6.2
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)

    # ---- geometry of the filmstrip -----------------------------------------
    left, right = 0.05, 0.985
    gap = 0.014
    fw = (right - left - gap * (N_SHOW - 1)) / N_SHOW
    fig_ar = FIG_W / FIG_H
    fh = fw * 0.75 * fig_ar          # 160x120 frame aspect, corrected for fig aspect
    ftop = 0.865
    fy = ftop - fh
    xs = [left + i * (fw + gap) for i in range(N_SHOW)]
    cxs = [x + fw / 2 for x in xs]

    # ---- frame images FIRST (bottom of the z-stack) ------------------------
    for x, fr in zip(xs, frames):
        ax = fig.add_axes([x, fy, fw, fh]); ax.imshow(fr, aspect="auto"); ax.axis("off")

    # ---- one overlay ON TOP of every frame ---------------------------------
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")

    # ---- title --------------------------------------------------------------
    ov.text(left, 0.965, "A fixed linear read-out applied per tick",
            fontsize=17, fontweight="bold", ha="left", va="center", color=INK)
    ov.text(left, 0.930,
            f"one clip  ->  frame-as-tick sequence   ( t = 1 ... T,   T = 16 @ 5 fps,   {action} )",
            fontsize=11.5, ha="left", va="center", color=MUT)

    # ---- frame borders, tick labels, and the stamped W ---------------------
    for i, (x, cx) in enumerate(zip(xs, cxs)):
        rrect(ov, x, fy, fw, fh, fc="none", ec=C_ENC, lw=2.0, rad=0.006, z=5)
        ov.text(cx, ftop + 0.022, f"t = {TICK_IDS[i]}",
                ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold")
        # the W stamp, OVERLAPPING the lower band of the frame
        sw, sh = fw * 0.62, 0.060
        sx, sy = cx - sw / 2, fy + 0.022
        rrect(ov, sx, sy, sw, sh, fc=C_W, ec="white", lw=1.8, alpha=0.94, rad=0.012, z=6)
        ov.text(cx, sy + sh * 0.62, r"$W$", ha="center", va="center",
                fontsize=13.5, color="white", fontweight="bold", zorder=7)
        ov.text(cx, sy + sh * 0.24, "fixed", ha="center", va="center",
                fontsize=7.5, color="white", style="italic", zorder=7)

    # ---- "same fixed map" bracket tying every W stamp ----------------------
    by = fy - 0.040
    bx0, bx1 = cxs[0], cxs[-1]
    for cx in cxs:
        ov.add_line(Line2D([cx, cx], [fy + 0.010, by], color=C_W, lw=1.3, zorder=4))
    ov.add_line(Line2D([bx0, bx1], [by, by], color=C_W, lw=2.4, zorder=4))
    ov.add_line(Line2D([bx0, bx0], [by, by + 0.012], color=C_W, lw=2.4, zorder=4))
    ov.add_line(Line2D([bx1, bx1], [by, by + 0.012], color=C_W, lw=2.4, zorder=4))
    ov.text((bx0 + bx1) / 2, by - 0.026,
            r"$\mathbf{the\ same\ fixed\ map\ }W\mathbf{\ at\ every\ tick}$"
            "  —  shared weights | frozen pair-index buffers | fixed decay $r=e^{-\\rho}$",
            ha="center", va="top", fontsize=11, color=C_W)

    # ---- per-tick logit lane -----------------------------------------------
    ly = by - 0.140
    lh, lw_ = 0.058, fw * 0.52
    for i, cx in enumerate(cxs):
        arrow(ov, (cx, by - 0.052), (cx, ly + lh), color=MUT, lw=1.3, ms=8, z=4)
        rrect(ov, cx - lw_ / 2, ly, lw_, lh, fc="#eef2f6", ec=MUT, lw=1.3, rad=0.010, z=5)
        ov.text(cx, ly + lh / 2, rf"$\mathrm{{logit}}_{{{TICK_IDS[i]}}}$",
                ha="center", va="center", fontsize=10, color=INK, zorder=6)
    ov.text(left, ly + lh + 0.028, "one logit per tick",
            ha="left", va="center", fontsize=9.5, color=MUT, style="italic")

    # ---- frame-as-tick loop axis + certain-tick exit + y-hat ---------------
    ax_y = ly - 0.098
    x_end = right - 0.135
    arrow(ov, (left, ax_y), (x_end, ax_y), color=MUT, lw=1.7, ms=13)
    for i, cx in enumerate(cxs):                     # drop logits onto the axis
        arrow(ov, (cx, ly - 0.006), (cx, ax_y + 0.006), color=MUT, lw=1.1, ms=7, z=4)
    ov.text(left, ax_y - 0.030, r"frame-as-tick loop,  $t = 1 \ldots T$"
            "   ·   inference: prefix-average logits, exit at the first tick whose certainty clears $\\theta$",
            ha="left", va="center", fontsize=9.5, color=MUT, style="italic")

    ex_x = cxs[3]   # ~ tick 10-11, near the measured mean hybrid exit 11.0/16
    ov.plot([ex_x], [ax_y], marker="*", ms=22, color="#e0a020",
            mec=INK, mew=0.9, zorder=6)
    # The exit is defined ONLY by the hybrid policy: certain-tick inspects every
    # tick and never stops, so "certain-tick exit" was a category error. The
    # distribution is also bimodal, so the mean is labelled as a mean, not as a
    # typical stopping point.
    ov.annotate("hybrid early exit (Alg. 1)\nmean 11.0 / 16 ticks",
                # place BELOW the axis and left-anchored: at tick ~11 the old
                # above-right position collided with the logit boxes
                xy=(ex_x, ax_y), xytext=(ex_x + 0.012, ax_y - 0.085),
                ha="left", va="top", fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))

    # y-hat output pill at the end of the loop
    yw, yh = right - x_end - 0.005, 0.070
    rrect(ov, x_end + 0.008, ax_y - yh / 2, yw, yh, fc="#e8f5ee", ec=C_W, lw=1.8, rad=0.016, z=5)
    ov.text(x_end + 0.008 + yw / 2, ax_y, r"$\hat{y}$  action class",
            ha="center", va="center", fontsize=11, color=C_W, fontweight="bold", zorder=6)

    # ---- bottom banner: the circuits-and-systems payoff --------------------
    band_y, band_h = 0.075, 0.160
    rrect(ov, left, band_y, right - left, band_h, fc=BAND, ec="#c9c39a", lw=1.4, rad=0.010, z=2)
    ov.text(left + 0.020, band_y + band_h - 0.038,
            "Deterministic, statically schedulable read-out — no input-dependent control flow",
            ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)
    ov.text(left + 0.020, band_y + 0.052,
            r"straight-line multiply-accumulate  |  constant $n_{\mathrm{synch}}(3{+}C)=212{,}992$ MACs / tick (UCF101, $n_{\mathrm{synch}}{=}2048$)",
            ha="left", va="center", fontsize=10.6, color=MUT)

    cw = 0.315
    cx = right - cw - 0.018
    rrect(ov, cx, band_y + 0.026, cw, band_h - 0.050, fc="#fdecea", ec=C_RED, lw=1.4, ls="--", rad=0.010, z=3)
    ov.text(cx + cw / 2, band_y + band_h - 0.052,
            "replaces  dynamic routing", ha="center", va="center",
            fontsize=10, color=C_RED, fontweight="bold")
    ov.text(cx + cw / 2, band_y + 0.058,
            r"iterative $O(r\,N_{\mathrm{prim}}C)$ | softmax re-estimated per pass"
            "\ndata-dependent  |  branch on input",
            ha="center", va="center", fontsize=8.8, color=C_RED, linespacing=1.35)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", out_path, "(.png/.pdf)")


if __name__ == "__main__":
    import sys
    # Default output directory. This previously pointed at a scratch path on the
    # authors' machine, which does not exist anywhere else; it is now repo-relative.
    OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/rendered"
    # UCF101 official split-1 TEST group, matching every other figure in the paper
    build("UCF101_full/Diving/v_Diving_g01_c01.avi", "Diving",
          f"{OUT}/fig_fixedmap_perframe_Diving")
    build("UCF101_full/Basketball/v_Basketball_g02_c01.avi", "Basketball",
          f"{OUT}/fig_fixedmap_perframe_Basketball")
