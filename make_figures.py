#!/usr/bin/env python3
"""
make_figures.py -- paper Figures 2 and 3.

    figures/02_ci_auc.png   (paper Figure 2)
    figures/03_ci_f1.png    (paper Figure 3)

95% t-interval on each model's own mean across the held-out regions -- NOT the
CI of the paired difference. Overlapping intervals here therefore do not imply a
non-significant paired test; the paired numbers are in tables.tex.

Style and palette match the existing CI figures: validated categorical slots 1
and 2 (all-pairs CVD dE 9.2, normal-vision 24.0), marker shape doubling the hue
encoding, no title or subtitle so the caption carries the caveats.

Because contiguous regions can hold a single class, the number of scored regions
can differ per species; the x tick labels state it so a narrow interval built on
fewer folds is never read as a tighter result.
"""

import json
import os

import numpy as np

FIG_ROOT = "figures"
SPECIES = [("Caretta caretta", "Mediterranean"),
           ("Cetorhinus maximus", "NE Atlantic"),
           ("Balaenoptera musculus", "California Current")]
def models(rf_key="rf_full"):
    return [("tabicl", "TabICLv2", "#2a78d6", "s"),
            (rf_key, "RF-full", "#eb6834", "o")]
MODELS = models()
METRICS = [("auc", "ROC-AUC", "02_ci_auc.png"), ("f1", "F1", "03_ci_f1.png")]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e4e3df", "#fcfcfb"


def ci(v):
    from scipy import stats
    v = np.asarray(v, float)
    n = v.size
    if n < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))


def load(fig_root):
    out = {}
    for sp, _ in SPECIES:
        p = os.path.join(fig_root, sp.replace(" ", "_"), "results.json")
        if os.path.exists(p):
            with open(p) as fh:
                out[sp] = json.load(fh)
    return out


def make(D, metric, label, fname, fig_root, dpi, rf_key="rf_full"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    present = [(sp, b) for sp, b in SPECIES if sp in D and D[sp]["folds"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(len(present), dtype=float)
    bounds = []
    MODELS_ = models(rf_key)
    for (key, name, colour, mark), dx in zip(MODELS_, (-0.15, +0.15)):
        means, halves = [], []
        for sp, _ in present:
            m, h = ci([f[f"{metric}_{key}"] for f in D[sp]["folds"]])
            means.append(m)
            halves.append(h)
            bounds += [m - h, m + h]
        xs = x + dx
        for keep, cap in ((np.array(halves) > 0, 4), (np.array(halves) == 0, 0)):
            if not keep.any():
                continue
            ax.errorbar(x=xs[keep], y=np.array(means)[keep],
                        yerr=np.array(halves)[keep], color=colour, capsize=cap,
                        capthick=1.6, elinewidth=1.6, linestyle="None",
                        marker=mark, markersize=9, mfc=colour, mec=SURFACE,
                        mew=1.4, zorder=4)
    ax.axhline(1.0, color=INK_3, lw=1.0, ls=":", zorder=1)
    bounds.append(1.0)
    lo, hi = min(bounds), max(bounds)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"$\\it{{{sp.split()[0]}}}$\n$\\it{{{sp.split()[1]}}}$\n{b}\n"
         f"({len(D[sp]['folds'])} regions)" for sp, b in present], fontsize=9)
    ax.tick_params(axis="both", colors=INK_2, labelsize=9, length=3)
    ax.set_xlim(-0.55, len(present) - 0.45)
    ax.set_ylabel(f"mean {label} over held-out regions", fontsize=10.5, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    handles = [Line2D([], [], color=c, marker=mk, markersize=9, mfc=c, mec=SURFACE,
                      mew=1.4, linestyle="None", label=n)
               for _k, n, c, mk in MODELS_]
    handles.append(Line2D([], [], color=INK_3, lw=1.0, ls=":", label="metric maximum = 1.0"))
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=9.5, labelcolor=INK,
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               handletextpad=0.6, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = os.path.join(fig_root, fname)
    os.makedirs(fig_root, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-root", default=FIG_ROOT)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--rf-key", default="rf_full",
                    help="rf_full or rf_tuned")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    D = load(a.fig_root)
    if not D:
        raise SystemExit(f"no results under {a.fig_root}")
    for m, lab, fn in METRICS:
        make(D, m, lab, fn, a.out_dir or a.fig_root, a.dpi, a.rf_key)
