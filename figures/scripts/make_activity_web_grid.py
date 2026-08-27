"""Small-multiples synchronisation webs across all 11 UCF-11 activities.

Colour = |sync| (ACTIVITY, per-clip normalised) so the salient coupling is the
red arc -- unlike the rho-coloured web where red is a fixed model parameter.
Shows the peak active coupling landing at a different grid coordinate per action.
"""
import glob, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch

spec = importlib.util.spec_from_file_location("m", "figures/scripts/make_neuron_dynamics_figure.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
CELL, SIZE = m.CELL, m.SIZE
INK, MUT = m.INK, m.MUT
CMAP = plt.cm.YlOrRd                         # weak (yellow) -> strong (red)

CLASSES = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding',
           'soccer_juggling', 'swing', 'tennis_swing', 'trampoline_jumping',
           'volleyball_spiking', 'walking']


def rc(c):
    gy, gx = divmod(int(c), 3)
    return (gy + 1, gx + 1)


def cell_xy(c):
    gy, gx = divmod(int(c), 3)
    return (gx + 0.5) / 3 * SIZE[1], (gy + 0.5) / 3 * SIZE[0]


def draw_web(ax, frame, sync, z, ci, cj, cross, tick=8, topk=22, seed=7):
    mag = np.abs(sync[-1])
    zt = np.abs(z[-1]).reshape(9, CELL).mean(1)
    sel = cross[np.argsort(mag[cross])[-topk:]]
    mmax = mag[sel].max() + 1e-9
    peak = cross[int(np.argmax(mag[cross]))]
    rng = np.random.default_rng(seed)

    ax.imshow(frame, aspect="auto")
    ax.imshow(np.ones((*SIZE, 4)) * [1, 1, 1, 0.5], aspect="auto")
    ax.set_xlim(0, SIZE[1]); ax.set_ylim(SIZE[0], 0); ax.axis("off")

    for k in sel:
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
    for c in range(9):
        x, y = cell_xy(c)
        s = 40 + 260 * (zt[c] / (zt.max() + 1e-9))
        ax.scatter([x], [y], s=s, color="#2e7d5b", edgecolor="white",
                   linewidth=1.1, zorder=7)
    return peak


def main():
    model = m.build_model()
    left = model.sync.left.numpy(); right = model.sync.right.numpy()
    ci, cj = left // CELL, right // CELL
    cross = np.where(ci != cj)[0]

    nrow, ncol = 3, 4
    fig = plt.figure(figsize=(13.8, 11.2), dpi=200)
    gs = fig.add_gridspec(nrow, ncol, left=0.02, right=0.98, top=0.865,
                          bottom=0.05, hspace=0.30, wspace=0.05)
    for i, cls in enumerate(CLASSES):
        vids = sorted(glob.glob(f"UCF11_updated_mpg/{cls}/**/*.mpg", recursive=True))
        clip = m.load_clip(vids[0]); z, sync, logit = m.capture(model, clip)
        frames = m.load_display_frames(vids[0])
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        peak = draw_web(ax, frames[7], sync, z, ci, cj, cross)
        ok = "OK" if int(logit.mean(0).argmax()) == m.CLASS_NAMES.index(cls) else "x"
        ax.set_title(f"{cls.replace('_',' ')}  [{ok}]\npeak {rc(ci[peak])}-{rc(cj[peak])}",
                     fontsize=10.5, color=INK, pad=5)

    # legend cell (12th slot)
    lax = fig.add_subplot(gs[2, 3]); lax.axis("off")
    cb = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=CMAP),
                      ax=lax, orientation="horizontal", fraction=0.5, pad=0.05)
    cb.set_label("arc shade = |sync| within clip\n(bold red = peak active coupling)",
                 fontsize=9.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUT)
    lax.text(0.5, 0.62,
             "Nodes = 3x3 capsule cells\n(size proportional to activation).\n\n"
             "The peak active coupling sits at a\ndifferent coordinate per activity -\n"
             "unlike the rho-red arc, which is a\nfixed weight (always (3,1)-(1,1)).",
             ha="center", va="center", fontsize=9.2, color=MUT, linespacing=1.4)

    fig.text(0.02, 0.965, "Where synchrony concentrates differs by activity",
             fontsize=18, fontweight="bold", ha="left", color=INK)
    fig.text(0.02, 0.935,
             "SyncCaps B1 · UCF-11 · one clip per class (frame at tick 8/16) · "
             "arcs coloured by |sync| (activity), peak coupling in bold red",
             fontsize=11, ha="left", color=MUT)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/rendered/fig_sync_web_activities.{ext}",
                    dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved figures/rendered/fig_sync_web_activities (.png/.pdf)")


if __name__ == "__main__":
    main()
