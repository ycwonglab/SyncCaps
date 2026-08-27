"""Fig. 7 — the learned tick strategy, on the CORRECTED headline checkpoint.

Three panels:
  (a) test accuracy vs tick budget under the three readout policies
  (b) distribution of Algorithm 1 hybrid exit ticks
  (c) histogram of the learned per-pair memory e^{-rho}

Panel (c) carries the paper's most robust interpretive claim: if e^{-rho}
concentrates near 1, the statistic sits within a few percent of the rho=0 limit,
where sync_{k,T} reduces to a count-normalised Gram entry -- i.e. ORDER-AGNOSTIC
second-order pooling, not temporal binding. The three independent frame-shuffle
controls agree, which is why this claim survived every correction.

Every accuracy here is computed by `_readout_metrics` / `hybrid_readout`, the
same functions behind every number in the ledger, so the curves cannot drift
from the tables. Single-view (one temporal window), matching the checkpoint's
own stored `mean_exit_tick`; multi-clip figures would need their own annotation.

Usage:  python figures/scripts/make_tick_strategy_figure.py [outdir]
        SYNCCAPS_FIG_CFG=legacy python figures/scripts/make_tick_strategy_figure.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("m", "figures/scripts/make_neuron_dynamics_figure.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
INK, MUT = m.INK, m.MUT

from src.training.exp_base import UCF11VideoDataset, make_official_split1, DEVICE
from src.models.temporal_routing import hybrid_readout
from src.training.synccaps_probe_experiment import _collect_logits, _readout_metrics

C_CERT, C_FIN, C_HYB = "#2e7d5b", "#5b6b78", "#c0392b"


def collect(model):
    ds = UCF11VideoDataset("UCF101_full", sequence_length=16, sample_fps=5.0,
                           augment=False, cache_dir=".cache")
    _, _, te = make_official_split1(ds, val_groups=())
    loader = DataLoader(Subset(ds, te), 4, shuffle=False, num_workers=0)
    return _collect_logits(model.to(DEVICE), loader)


def main(outdir="figures/rendered"):
    model = m.build_model()
    rho = model.sync.rho.detach().cpu().numpy()
    lg, y = collect(model)
    T = lg.shape[-1]

    # (a) three policies as a function of the tick budget
    cert, fin, hyb = [], [], []
    for tp in range(1, T + 1):
        r = _readout_metrics(lg[..., :tp], y)
        cert.append(r["acc_certain"]); fin.append(r["acc_final"]); hyb.append(r["acc_hybrid"])
    # (b) exit ticks under the full budget
    _, idx, _ = hybrid_readout(lg)
    exits = (idx + 1).numpy()
    mem = np.exp(-rho)

    fig = plt.figure(figsize=(15.2, 4.5), dpi=200)
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.76, bottom=0.15, wspace=0.26)
    ticks = np.arange(1, T + 1)

    ax = fig.add_subplot(gs[0])
    for v, c, lab in ((cert, C_CERT, "certain-tick (scans all ticks)"),
                      (fin, C_FIN, "final-tick"), (hyb, C_HYB, "hybrid, Alg. 1 (causal)")):
        ax.plot(ticks, v, color=c, lw=2.1, marker="o", ms=3.4, label=lab)
    ax.set_xlabel("tick budget  $T'$"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("(a) accuracy vs tick budget", fontsize=11.5, color=INK, pad=7)
    ax.legend(fontsize=8.4, frameon=False, loc="lower right")
    ax.grid(alpha=.25, lw=.6)

    ax = fig.add_subplot(gs[1])
    ax.hist(exits, bins=np.arange(.5, T + 1.5), color=C_HYB, alpha=.85, edgecolor="white")
    ax.axvline(exits.mean(), color=INK, ls="--", lw=1.5)
    f1, fT = (exits == 1).mean() * 100, (exits == T).mean() * 100
    ax.text(.30, .93, f"mean {exits.mean():.1f} / {T}\n"
                      f"but {f1:.0f}% exit at tick 1\nand {fT:.0f}% run the full budget",
            transform=ax.transAxes, va="top", fontsize=8.8, color=INK, linespacing=1.5)
    ax.set_xlabel("exit tick (hybrid, Alg. 1)"); ax.set_ylabel("clips")
    # The distribution is BIMODAL, so the mean names a tick almost no clip uses.
    # rev14 described this as "distributed across the sequence"; it is not.
    ax.set_title("(b) the causal readout is bimodal, not gradual",
                 fontsize=11.5, color=INK, pad=7)
    ax.grid(alpha=.25, lw=.6, axis="y")

    ax = fig.add_subplot(gs[2])
    ax.hist(mem, bins=60, color="#3b6ea5", alpha=.85, edgecolor="white")
    ax.axvline(np.median(mem), color=INK, ls="--", lw=1.5)
    ax.set_xlabel(r"learned per-pair memory  $e^{-\rho}$"); ax.set_ylabel("pairs")
    ax.set_title("(c) the learned decay IS engaged", fontsize=11.5, color=INK, pad=7)
    ax.text(.03, .97, f"median {np.median(mem):.3f}   min {mem.min():.3f}\n"
                      "$\\rho \\rightarrow 0$ (= order-agnostic Gram) would sit at 1.0",
            transform=ax.transAxes, va="top", fontsize=8.4, color=MUT, linespacing=1.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)          # headroom so the note clears the bars
    ax.grid(alpha=.25, lw=.6, axis="y")

    fig.text(.055, .93, "The learned tick strategy", fontsize=16.5,
             fontweight="bold", ha="left", color=INK)
    fig.text(.055, .865,
             f"SyncCaps {m.CFG['arm_label']} · {m.CFG['dset_label']} · seed 42 · "
             f"single view · {len(y)} test clips · certain-tick scans every tick and "
             "never stops, so only the hybrid policy defines an exit",
             fontsize=9.6, ha="left", color=MUT)
    os.makedirs(outdir, exist_ok=True)
    suf = "" if m.CFG["num_classes"] == 11 else "_ucf101"
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/fig_tick_strategy{suf}.{ext}", dpi=200,
                    facecolor="white", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    print(f"certain@16 {cert[-1]:.2f} | final@16 {fin[-1]:.2f} | hybrid@16 {hyb[-1]:.2f}")
    print(f"mean exit {exits.mean():.2f} (BIMODAL: {(exits==1).mean()*100:.1f}% at tick 1, "
          f"{(exits==T).mean()*100:.1f}% at tick {T}) | e^-rho median {np.median(mem):.4f} "
          f"min {mem.min():.4f} max {mem.max():.4f}")
    print(f"saved {outdir}/fig_tick_strategy{suf} (.png/.pdf)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figures/rendered")
