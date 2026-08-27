"""Small-multiples synchronisation webs across all 11 UCF-11 activities, using
each clip's own PEAK-ACTIVE tick (argmax_t mean_k |z_t[i]z_t[j]|, the peak
instantaneous co-activation -- NOT the accumulated |sync|, which grows
monotonically and peaks trivially at t=16 for every clip).

Per panel: backdrop = peak-active frame; arcs coloured by |sync| at that tick
(bold red = peak active coupling); the fixed top-rho arcs are overlaid faintly
(dashed) -- identical on every panel, to contrast the fixed order-sensitive
wiring against the content-driven active coupling.

New copy -- does not overwrite fig_sync_web_activities.*
"""
import glob, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

spec = importlib.util.spec_from_file_location("m", "figures/scripts/make_neuron_dynamics_figure.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
CELL, SIZE, INK, MUT = m.CELL, m.SIZE, m.INK, m.MUT
CMAP = plt.cm.YlOrRd
FIXED_RED = "#c0392b"                       # faint dashed = fixed rho arcs
N_FIXED = 4                                 # top-rho cross arcs to overlay

# Classes, clip source and caption all follow the config selected in
# make_neuron_dynamics_figure.py (SYNCCAPS_FIG_CFG=headline|legacy), so the two
# figures can never be built from different checkpoints by accident.
CLASSES = m.CFG["web_classes"]
CLIP_GLOB = m.CFG["clip_glob"]
SUBTITLE = m.CFG["subtitle"]
SUF = "" if m.CFG["num_classes"] == 11 else "_ucf101"


def rc(c):
    gy, gx = divmod(int(c), 3); return (gy + 1, gx + 1)


def cell_xy(c):
    gy, gx = divmod(int(c), 3)
    return (gx + 0.5) / 3 * SIZE[1], (gy + 0.5) / 3 * SIZE[0]


def peak_tick(z, left, right):
    """Peak instantaneous co-activation frame (varies per clip)."""
    return int(np.abs(z[:, left] * z[:, right]).mean(1).argmax())


def draw_panel(ax, frame, sync_t, z_t, ci, cj, cross, fixed, topk=22, seed=7):
    mag = np.abs(sync_t)
    zt = np.abs(z_t).reshape(9, CELL).mean(1)
    sel = cross[np.argsort(mag[cross])[-topk:]]
    mmax = mag[sel].max() + 1e-9
    peak = cross[int(np.argmax(mag[cross]))]
    rng = np.random.default_rng(seed)

    ax.imshow(frame, aspect="auto")
    ax.imshow(np.ones((*SIZE, 4)) * [1, 1, 1, 0.5], aspect="auto")
    ax.set_xlim(0, SIZE[1]); ax.set_ylim(SIZE[0], 0); ax.axis("off")

    for k in sel:                                    # content-driven active web
        xi, yi = cell_xy(ci[k]); xj, yj = cell_xy(cj[k])
        w = mag[k] / mmax
        rad = 0.2 * np.sign(rng.uniform(-1, 1)) + rng.uniform(-0.1, 0.1)
        bold = (k == peak)
        ax.add_patch(FancyArrowPatch(
            (xi, yi), (xj, yj), arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            color=("#b3132a" if bold else CMAP(0.25 + 0.75 * w)),
            lw=(4.2 if bold else 0.6 + 2.4 * w),
            alpha=(1.0 if bold else 0.35 + 0.5 * w),
            zorder=(6 if bold else 4), capstyle="round"))
    for k in fixed:                                  # FIXED rho arcs, faint dashed
        xi, yi = cell_xy(ci[k]); xj, yj = cell_xy(cj[k])
        ax.add_patch(FancyArrowPatch(
            (xi, yi), (xj, yj), arrowstyle="-",
            connectionstyle="arc3,rad=0.22",
            color=FIXED_RED, lw=1.6, alpha=0.5, linestyle=(0, (5, 3)),
            zorder=8, capstyle="round"))
    for c in range(9):
        x, y = cell_xy(c)
        s = 40 + 260 * (zt[c] / (zt.max() + 1e-9))
        ax.scatter([x], [y], s=s, color="#2e7d5b", edgecolor="white",
                   linewidth=1.1, zorder=9)
    return peak


def main():
    model = m.build_model()
    left = model.sync.left.numpy(); right = model.sync.right.numpy()
    rho = model.sync.rho.detach().numpy()
    ci, cj = left // CELL, right // CELL
    cross = np.where(ci != cj)[0]
    fixed = cross[np.argsort(rho[cross])[-N_FIXED:]]     # same on every panel

    nrow, ncol = 3, 4
    fig = plt.figure(figsize=(13.8, 11.2), dpi=200)
    gs = fig.add_gridspec(nrow, ncol, left=0.02, right=0.98, top=0.865,
                          bottom=0.05, hspace=0.30, wspace=0.05)
    for i, cls in enumerate(CLASSES):
        # Same reproducible rule as the dynamics figure: prefer a correctly
        # classified clip, and mark the panel honestly when none is found.
        v, ok_sel = m.select_clip(model, cls)
        assert v, f"no clip found for {cls} via {CLIP_GLOB}"
        z, sync, logit = m.capture(model, m.load_clip(v))
        frames = m.load_display_frames(v)
        ti = peak_tick(z, left, right)
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        peak = draw_panel(ax, frames[ti], sync[ti], z[ti], ci, cj, cross, fixed)
        ok = "OK" if int(logit.mean(0).argmax()) == m.CLASS_NAMES.index(cls) else "x"
        ax.set_title(f"{cls.replace('_',' ')}  [{ok}]   peak-tick {ti+1}/16\n"
                     f"active peak {rc(ci[peak])}-{rc(cj[peak])}",
                     fontsize=10.3, color=INK, pad=5)

    lax = fig.add_subplot(gs[2, 3]); lax.axis("off")
    lax.legend(handles=[
        Line2D([0], [0], color="#b3132a", lw=4, label="peak active coupling (per clip)"),
        Line2D([0], [0], color=FIXED_RED, lw=1.8, ls=(0, (5, 3)),
               label="fixed ρ arcs (same every panel)")],
        loc="upper center", fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    lax.text(0.5, 0.55,
             "Backdrop = each clip's peak-active frame\n(peak instantaneous co-activation). The\n"
             "active peak moves per activity; the dashed\nρ arcs are a fixed weight, identical everywhere.",
             ha="center", va="center", fontsize=9, color=MUT, linespacing=1.45)
    pos = lax.get_position()
    cax = fig.add_axes([pos.x0 + 0.012, pos.y0 + 0.015, pos.width - 0.024, 0.013])
    cb = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=CMAP),
                      cax=cax, orientation="horizontal")
    cb.set_label("arc shade = |sync| at peak-active tick", fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUT)

    fig.text(0.02, 0.965, "Synchronisation webs at each activity's peak-active tick",
             fontsize=17.5, fontweight="bold", ha="left", color=INK)
    fig.text(0.02, 0.935,
             SUBTITLE + " · backdrop = peak instantaneous-co-activation frame"
             " · dashed = fixed ρ arcs overlaid",
             fontsize=10.5, ha="left", color=MUT)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/rendered/fig_sync_web_activities_peaktick{SUF}.{ext}",
                    dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"saved figures/rendered/fig_sync_web_activities_peaktick{SUF} (.png/.pdf)")


if __name__ == "__main__":
    main()
