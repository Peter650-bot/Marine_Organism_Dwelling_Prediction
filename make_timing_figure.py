#!/usr/bin/env python3
"""
make_timing_figure.py -- TabICLv2 (no KV cache) vs the tuned Random Forest on
the loggerhead turtle: wall-clock time per held-out fold (top) and ROC-AUC
(bottom), at a 2,000-row context and at full context.

    figures/timing_context_vs_rf_tuned.png
    figures/timing_context_vs_rf_tuned.pdf

Rows used from time.csv (all with 4 CPU threads for both models):
  2,000-row context (3 draws per fold, averaged within the fold first)
    TabICLv2   speedup_E_context2k      tabicl_total_s, auc_tabicl
    tuned RF   rf_tuned_context2k       rf_tuned_total_s, auc_rf_tuned
  full context
    TabICLv2   speedup_B_4threads       tabicl_total_s, auc_tabicl
    tuned RF   rf_tuned_full_4threads   rf_tuned_total_s, auc_rf_tuned
The bars figure (make_bars) reads context_sweep.jsonl instead; see there.

Tuned-RF time is the nested-CV hyperparameter search plus the final fit and
predict. Folds are matched across the two models. The script prints every
plotted number so the figure can be checked.

Palette matches make_figures.py (slots validated: CVD dE 24.7, normal 33.6).
"""

import os

import numpy as np

CSV = "time.csv"
FIG_ROOT = "figures"
OUT = "timing_context_vs_rf_tuned"
TABICL, RF = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e4e3df", "#fcfcfb"
LABEL_BOX = dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none")
SPECIES = "Caretta caretta"
# (context label, TabICLv2 experiment, tuned-RF experiment)
BAR_CONTEXTS = [("subsampled", 2000), ("full context", 0)]
GROUPS = [("subsampled", "speedup_E_context2k", "rf_tuned_context2k"),
          ("full context", "speedup_B_4threads", "rf_tuned_full_4threads")]


def per_fold(df, exp, t_col, auc_col):
    """Fold-level time / AUC / context size, draws averaged within a fold."""
    d = df[(df.species == SPECIES) & (df.experiment == exp)]
    g = d.groupby("fold").agg(t=(t_col, "mean"), auc=(auc_col, "mean"),
                              n=("n_context", "mean"))
    assert len(g) and not g.isna().any().any(), f"missing values for {exp}"
    return g


def ratio_label(t_icl, t_rf):
    r = t_rf / t_icl
    return f"{r:.1f}× faster" if r >= 1.05 else (
        f"{1 / r:.1f}× slower" if r <= 0.95 else "same time")


def make(dpi):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = pd.read_csv(CSV)
    rng = np.random.default_rng(0)
    fig, (ax_t, ax_a) = plt.subplots(2, 1, figsize=(4.8, 6.2), sharex=True,
                                     gridspec_kw={"height_ratios": [1.15, 1]})
    x = np.arange(len(GROUPS), dtype=float)
    dx = 0.17
    ticks = []
    print(f"{'context':24s} {'model':8s} {'folds':>5s} {'time_s':>8s} {'auc':>6s}")

    for i, (label, exp_icl, exp_rf) in enumerate(GROUPS):
        icl = per_fold(df, exp_icl, "tabicl_total_s", "auc_tabicl")
        rf = per_fold(df, exp_rf, "rf_tuned_total_s", "auc_rf_tuned")
        assert list(icl.index) == list(rf.index), f"fold mismatch for {label}"
        n_ctx = int(round(icl.n.mean()))
        ticks.append(f"{n_ctx:,} rows\n({label})")

        for g, colour, mark, xm, side in ((icl, TABICL, "s", x[i] - dx, -1),
                                          (rf, RF, "o", x[i] + dx, +1)):
            for ax, v, fmt in ((ax_t, g.t.values, "{:,.0f} s"),
                               (ax_a, g.auc.values, "{:.3f}")):
                ax.scatter(xm + rng.uniform(-0.04, 0.04, v.size), v, s=22,
                           marker=mark, facecolor="none", edgecolor=colour,
                           lw=1.0, alpha=0.75, zorder=3)
                ax.scatter([xm], [v.mean()], s=90, marker=mark, color=colour,
                           edgecolor=SURFACE, lw=1.4, zorder=5)
                ax.annotate(fmt.format(v.mean()), (xm, v.mean()),
                            xytext=(9 * side, 0), textcoords="offset points",
                            va="center", ha="left" if side > 0 else "right",
                            fontsize=8, color=INK_2, bbox=LABEL_BOX)
            print(f"{n_ctx:,} rows ({label})".ljust(24),
                  f"{'TabICLv2' if colour == TABICL else 'tunedRF':8s} "
                  f"{len(g):5d} {g.t.mean():8.1f} {g.auc.mean():6.3f}")

        ax_t.text(x[i], 420, ratio_label(icl.t.mean(), rf.t.mean()), ha="center",
                  va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    ax_t.text(np.mean(x), 900, "TabICLv2 speed relative to the tuned RF",
              ha="center", va="bottom", fontsize=8.5, color=INK_2)

    ax_t.set_yscale("log")
    ax_t.set_ylim(5, 1500)
    ax_t.set_ylabel("wall-clock time per fold (s, log)", fontsize=10, color=INK_2)
    ax_a.set_ylim(0.65, 1.03)
    ax_a.axhline(1.0, color=INK_3, lw=1.0, ls=":", zorder=1)
    ax_a.set_ylabel("ROC-AUC on held-out region", fontsize=10, color=INK_2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ticks, fontsize=8.5)
    ax_a.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax_a.set_xlabel("Loggerhead turtle, Mediterranean: context size (4 folds)",
                    fontsize=9, color=INK_2, labelpad=8)
    for ax in (ax_t, ax_a):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors=INK_2, labelsize=8.5, length=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    handles = [
        Line2D([], [], marker="s", color=TABICL, mec=SURFACE, mew=1.4, markersize=9,
               ls="None", label="TabICLv2 (mean)"),
        Line2D([], [], marker="o", color=RF, mec=SURFACE, mew=1.4, markersize=9,
               ls="None", label="Tuned RF, nested-CV search + fit (mean)"),
        Line2D([], [], marker="o", mfc="none", mec=INK_3, ls="None", markersize=5,
               label="individual folds"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=1, fontsize=8.5, labelcolor=INK,
               loc="lower center", bbox_to_anchor=(0.5, -0.01), handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=1.2)
    os.makedirs(FIG_ROOT, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_ROOT, f"{OUT}.{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {out}")
    plt.close(fig)


def make_bars(dpi):
    """Grouped bars, time (left) and AUC (right), from the context sweep
    (context_sweep.jsonl; one run in which both models were timed back to back,
    and whose full-context tuned RF reproduces results.json, 0.899). Time
    whiskers span the fastest to slowest fold; AUC whiskers are the bootstrap
    95% CI from make_context_sweep_figure.summarise. AUC bars grow from 0.5
    (chance), so bar length is the skill above a random ranking."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    import make_context_sweep_figure as cs
    summ, _, folds = cs.summarise(cs.load())
    fig, (ax_t, ax_a) = plt.subplots(1, 2, figsize=(7.4, 3.9))
    x = np.arange(len(BAR_CONTEXTS), dtype=float)
    w = 0.3       # bar width; the 2px surface gap comes from the edge line
    ticks = []
    AUC_BASE = 0.5

    for i, (label, c) in enumerate(BAR_CONTEXTS):
        icl, rf = summ["tabicl", c], summ["rf_tuned", c]
        ticks.append(f"{icl['n']:,.0f} rows\n({label})")
        for v, colour, xm in ((icl, TABICL, x[i] - w / 2), (rf, RF, x[i] + w / 2)):
            # time: whisker = fastest..slowest fold; AUC: bootstrap 95% CI
            for ax, m, lo, hi, base, fmt in (
                    (ax_t, v["t"], v["t_min"], v["t_max"], 0.0, "{:,.0f} s"),
                    (ax_a, v["auc"], v["lo"], v["hi"], AUC_BASE, "{:.3f}")):
                ax.bar(xm, m - base, bottom=base, width=w, color=colour,
                       edgecolor=SURFACE, linewidth=2, zorder=3)
                ax.vlines(xm, lo, hi, color=INK_2, lw=1.0, zorder=4)
                ax.hlines([lo, hi], xm - 0.035, xm + 0.035, color=INK_2, lw=1.0, zorder=4)
                ax.annotate(fmt.format(m), (xm, max(m, hi)), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=8.5, color=INK, zorder=5)
            print(f"bars: {label:12s} {'TabICLv2' if colour == TABICL else 'tunedRF':8s} "
                  f"{v['t']:6.1f} s [{v['t_min']:.1f}, {v['t_max']:.1f}]  "
                  f"AUC {v['auc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
        ax_t.text(x[i], 205, ratio_label(icl["t"], rf["t"]),
                  ha="center", va="bottom", fontsize=9, color=INK,
                  fontweight="bold")

    ax_t.set_ylim(0, 225)
    ax_t.set_ylabel("wall-clock mean time per fold (s)", fontsize=10, color=INK_2)
    ax_t.set_title("Time (lower is faster)", fontsize=10, color=INK, loc="left")
    ax_a.set_ylim(AUC_BASE, 1.06)
    ax_a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_ylabel("ROC-AUC on held-out region\n(bars start at 0.5 = chance)",
                    fontsize=10, color=INK_2)
    ax_a.set_title("ROC-AUC (higher is better)", fontsize=10, color=INK, loc="left")
    for ax in (ax_t, ax_a):
        ax.set_facecolor(SURFACE)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8.5)
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors=INK_2, labelsize=8.5, length=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    handles = [Patch(color=TABICL, label="TabICLv2"),
               Patch(color=RF, label="Tuned RF, nested-CV search + fit"),
               Line2D([], [], color=INK_2, lw=1.0,
                      label="whiskers: time, fastest–slowest fold; AUC, bootstrap 95% CI")]
    fig.legend(handles=handles, frameon=False, ncol=2, fontsize=8.5, labelcolor=INK,
               loc="lower center", bbox_to_anchor=(0.5, -0.01), handlelength=1.4,
               columnspacing=1.6)
    fig.suptitle("Loggerhead turtle, Mediterranean: mean over 4 held-out folds",
                 fontsize=10, color=INK_2, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=2.5)
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_ROOT, f"{OUT}_bars.{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=int, default=300)
    dpi = ap.parse_args().dpi
    make(dpi)
    make_bars(dpi)
