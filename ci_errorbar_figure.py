#!/usr/bin/env python3
"""
ci_errorbar_figure.py — mean AUC and mean F1 with 95% confidence intervals for
TabICL vs RandomForest (full data), all three species, spatial-block CV.

One figure per metric:
    figures/21_ci_auc.png
    figures/22_ci_f1.png

This plots the CI of *each model's own mean* over the five held-out regions --
NOT the CI of the paired per-fold difference, which is what
`tabicl_v2_report.py` reports and what the significance tests are built on. The
two answer different questions: this figure asks "how well is each model
pinned down", the paired CI asks "do they differ". Overlapping intervals here
therefore do NOT imply a non-significant paired difference -- Balaenoptera is
exactly that case (intervals overlap, paired p = 0.016), because the paired
test cancels the fold-to-fold difficulty that dominates both models' spread.
The figures carry no title or subtitle, so that caveat has to live in the
caption wherever they are used.

Reads only committed artefacts (`figures/<species>/tabicl_f1_spatial.json`);
nothing is refitted.

Two things the intervals will do that are correct but look wrong:
  * with 5 folds a t-CI has 4 df, so t* = 2.776 and the intervals are wide;
  * they are not bounded by the metric's range, so they can cross 1.0. The
    figure draws that bound rather than clipping the interval into it.

Neither figure draws the all-HIGH reference line. Test subsamples are
class-balanced, so the trivial all-HIGH classifier scores F1 = 0.667 -- that,
not 0, is the floor the F1 figure should be read against, and with no line on
the plot the caption has to say so.

CLI:
    python ci_errorbar_figure.py
    python ci_errorbar_figure.py --dpi 600
"""

import argparse
import json
import os

import numpy as np

FIG_ROOT = "figures"
SPECIES = [
    ("Caretta caretta", "Mediterranean"),
    ("Cetorhinus maximus", "NE Atlantic"),
    ("Balaenoptera musculus", "California Current"),
]
MODELS = [("tabicl", "TabICLv2"), ("rf_full", "RF-full")]
METRICS = [("auc", "ROC-AUC", "21_ci_auc.png"), ("f1", "F1", "22_ci_f1.png")]

# Validated categorical slots 1-2 (see the dataviz palette): all-pairs CVD
# dE 24.7, normal-vision 33.6, both >= 3:1 on the light surface. Marker shape
# doubles the encoding so identity never rests on hue alone.
C_TABICL, C_RF = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e4e3df", "#fcfcfb"

# Kept for reference: class-balanced test subsamples mean the trivial
# all-HIGH classifier scores this, so it is the floor F1 is read against.
# No longer drawn on the figure -- state it in the caption instead.
F1_TRIVIAL = 2 / 3


def load_folds(species, fig_root=FIG_ROOT, scheme="spatial"):
    path = os.path.join(fig_root, species.replace(" ", "_"),
                        f"tabicl_f1_{scheme}.json")
    with open(path) as fh:
        return json.load(fh)["folds"]


def ci(vals):
    """Mean and half-width of a 95% t-CI. Bounded metrics are not clipped."""
    from scipy import stats
    v = np.asarray(vals, float)
    n = v.size
    half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean(), half


def make_figure(data, metric, label, out, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    x = np.arange(len(SPECIES), dtype=float)
    off = 0.15
    bounds = []

    for (key, name), color, mark, dx in (
            (MODELS[0], C_TABICL, "s", -off), (MODELS[1], C_RF, "o", +off)):
        means, halves = [], []
        for sp, _ in SPECIES:
            m, h = ci([f[f"{metric}_{key}"] for f in data[sp]])
            means.append(m)
            halves.append(h)
            bounds += [m - h, m + h]
        ax.errorbar(x=x + dx, y=means, yerr=halves, color=color, capsize=4,
                    capthick=1.6, elinewidth=1.6, linestyle="None",
                    marker=mark, markersize=8, mfc=color, mec=SURFACE,
                    mew=1.4, label=name, zorder=4)

    ax.axhline(1.0, color=INK_3, lw=1.0, ls=":", zorder=1)
    bounds.append(1.0)

    lo, hi = min(bounds), max(bounds)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\it{{{sp.split()[0]}}}$\n$\\it{{{sp.split()[1]}}}$"
                        f"\n{basin}" for sp, basin in SPECIES], fontsize=9.5)
    ax.tick_params(axis="both", colors=INK_2, labelsize=9.5, length=3)
    ax.set_xlim(-0.55, len(SPECIES) - 0.45)
    ax.set_ylabel(f"mean {label} over 5 held-out regions",
                  fontsize=10.5, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    handles = [
        Line2D([], [], color=C_TABICL, marker="s", markersize=8, mfc=C_TABICL,
               mec=SURFACE, mew=1.4, linestyle="None", label="TabICLv2"),
        Line2D([], [], color=C_RF, marker="o", markersize=8, mfc=C_RF,
               mec=SURFACE, mew=1.4, linestyle="None", label="RF-full"),
        Line2D([], [], color=INK_3, lw=1.0, ls=":", label="metric maximum = 1.0"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=9.5,
               labelcolor=INK, loc="lower center", bbox_to_anchor=(0.5, -0.008),
               handletextpad=0.6, columnspacing=1.6)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[saved] {out}")


def build(fig_root=FIG_ROOT, dpi=300):
    data = {sp: load_folds(sp, fig_root) for sp, _ in SPECIES}
    for metric, label, fname in METRICS:
        make_figure(data, metric, label, os.path.join(fig_root, fname), dpi)
        print(f"\n{label}")
        for sp, _ in SPECIES:
            parts = []
            for key, name in MODELS:
                m, h = ci([f[f"{metric}_{key}"] for f in data[sp]])
                parts.append(f"{name} {m:.4f} [{m-h:.4f}, {m+h:.4f}]")
            print(f"  {sp:<24} " + "   ".join(parts))
        print()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-root", default=FIG_ROOT)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)
    return build(args.fig_root, args.dpi)


if __name__ == "__main__":
    raise SystemExit(main())
