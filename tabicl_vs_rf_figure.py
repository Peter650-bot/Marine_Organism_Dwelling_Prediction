#!/usr/bin/env python3
"""TabICL vs RF-full: AUC with 95% CIs under random and spatial-block CV.

Reads the per-fold AUCs recorded by ``tabicl_benchmark.py`` (full-context mode)
from ``figures/<species>/tabicl_results.json`` and renders a two-panel figure:

  A  mean AUC per model per CV scheme, 95% CI (t, 4 df) + the 5 fold values
  B  the paired difference (TabICL - RF-full) with its 95% CI against zero

Panel B is the one that answers "is the difference real": the random-CV CI
excludes zero (p = 0.004), the spatial-CV CI does not (p = 0.28).

Usage:
    python3 tabicl_vs_rf_figure.py \
        [--results figures/Caretta_caretta/tabicl_results.json] \
        [--out TabICL_vs_RF_full_AUC.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# --- palette (validated categorical slots 1-2, light surface) -----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_TABICL = "#2a78d6"   # slot 1, blue
C_RF = "#eb6834"       # slot 2, orange

SCHEME_LABEL = {"random": "Random 5-fold CV", "spatial": "Spatial-block CV"}
SCHEME_SUB = {
    "random": "optimistic — inflated by spatial autocorrelation",
    "spatial": "honest — transfer to unseen regions",
}


def ci95(x):
    """Mean and half-width of the 95% t-CI of the mean."""
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return float(x.mean()), 0.0
    sem = x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * sem)


def collect(results_path):
    res = json.loads(Path(results_path).read_text())
    pts = {}
    for p in res["points"]:
        if p.get("mode") != "full":
            continue
        tab = np.asarray(p["auc_tabicl_folds"], float)
        rf = np.asarray(p["auc_rf_full_folds"], float)
        delta = tab - rf
        d_mean, d_half = ci95(delta)
        t_stat, p_val = stats.ttest_rel(tab, rf)
        pts[p["scheme"]] = {
            "tabicl": tab,
            "rf": rf,
            "tabicl_ci": ci95(tab),
            "rf_ci": ci95(rf),
            "delta": delta,
            "delta_mean": d_mean,
            "delta_half": d_half,
            "p": float(p_val),
            "wins": int((delta > 0).sum()),
            "n_folds": tab.size,
            "context_rows": p.get("context_rows"),
        }
    return res, pts


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="figures/Caretta_caretta/tabicl_results.json")
    ap.add_argument("--out", default="TabICL_vs_RF_full_AUC.png")
    ap.add_argument("--species", default=None)
    args = ap.parse_args()

    res, pts = collect(args.results)
    species = args.species or res.get("species", "")
    schemes = [s for s in ("random", "spatial") if s in pts]
    rng = np.random.default_rng(42)  # jitter only; nothing statistical

    fig = plt.figure(figsize=(13.4, 7.3), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.42, 1.0], left=0.088, right=0.988,
        top=0.795, bottom=0.175, wspace=0.20,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    style_axes(axA)
    style_axes(axB)

    # ---------------- Panel A: absolute AUC ---------------------------------
    # Layout per scheme group: a heading row, then TabICL, then RF-full.
    rows, ylabels, headers = [], [], []
    y = 0.0
    for si, scheme in enumerate(schemes):
        headers.append((y, scheme))
        y -= 0.72
        for model, key, color in (
            ("TabICL", "tabicl", C_TABICL),
            ("RF-full", "rf", C_RF),
        ):
            rows.append((y, scheme, model, key, color))
            ylabels.append((y, model))
            y -= 1.0
        y -= 1.15  # gap between scheme groups

    for ypos, scheme, model, key, color in rows:
        d = pts[scheme]
        folds = d[key]
        mean, half = d[key + "_ci"]
        jit = rng.uniform(-0.15, 0.15, folds.size)
        axA.scatter(folds, ypos + jit, s=26, color=color, alpha=0.30,
                    edgecolors="none", zorder=3)
        axA.plot([mean - half, mean + half], [ypos, ypos], color=color,
                 lw=2.0, solid_capstyle="butt", zorder=4)
        for xcap in (mean - half, mean + half):
            axA.plot([xcap, xcap], [ypos - 0.15, ypos + 0.15], color=color,
                     lw=2.0, solid_capstyle="round", zorder=4)
        axA.scatter([mean], [ypos], s=115, color=color, zorder=5,
                    edgecolors=SURFACE, linewidths=2.0)
        axA.text(mean, ypos + 0.20, f"{mean:.3f}", ha="center", va="bottom",
                 fontsize=10.5, color=INK, fontweight="bold")
        axA.text(mean + half + 0.008, ypos, f"95% CI ±{half:.3f}", ha="left",
                 va="center", fontsize=8.4, color=MUTED)

    axA.set_yticks([yp for yp, _ in ylabels])
    axA.set_yticklabels([m for _, m in ylabels], fontsize=10.5)
    for tick, (_, m) in zip(axA.get_yticklabels(), ylabels):
        tick.set_color(C_TABICL if m == "TabICL" else C_RF)
        tick.set_fontweight("bold")

    ytop = max(r[0] for r in rows) + 1.35
    ybot = min(r[0] for r in rows) - 0.85
    for si, (yh, scheme) in enumerate(headers):
        axA.text(0.006, yh, SCHEME_LABEL[scheme], transform=axA.get_yaxis_transform(),
                 ha="left", va="center", fontsize=11.5, color=INK, fontweight="bold")
        axA.text(0.006, yh - 0.36, SCHEME_SUB[scheme],
                 transform=axA.get_yaxis_transform(), ha="left", va="center",
                 fontsize=8.6, color=MUTED, style="italic")
        if si > 0:
            axA.axhline(yh + 0.62, color=GRID, lw=1.0, zorder=1)

    axA.set_xlim(0.55, 1.115)
    axA.set_ylim(ybot, ytop)
    axA.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axA.set_xlabel("ROC-AUC  (0.5 = chance)", fontsize=10, color=INK_2, labelpad=8)
    axA.axvline(1.0, color=AXIS, lw=0.9, ls=(0, (3, 3)), zorder=2)
    axA.text(1.0, ytop - 0.12, "AUC = 1.0", fontsize=8.5, color=MUTED,
             ha="center", va="top")
    axA.set_title(
        "A · Mean AUC with 95% CI   ·   faint dots = the 5 individual folds",
        fontsize=11.5, color=INK, loc="left", pad=12, fontweight="bold",
    )

    # ---------------- Panel B: paired difference ----------------------------
    ypos_map, yb = {}, 0.0
    for scheme in schemes:
        ypos_map[scheme] = yb
        yb -= 2.15

    axB.axvline(0.0, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    for scheme, ypos in ypos_map.items():
        d = pts[scheme]
        lo, hi = d["delta_mean"] - d["delta_half"], d["delta_mean"] + d["delta_half"]
        sig = lo > 0 or hi < 0
        col = INK if sig else MUTED
        jit = rng.uniform(-0.13, 0.13, d["delta"].size)
        axB.text(0.006, ypos + 0.92, SCHEME_LABEL[scheme],
                 transform=axB.get_yaxis_transform(), ha="left", va="center",
                 fontsize=11.5, color=INK, fontweight="bold")
        axB.scatter(d["delta"], ypos + jit, s=26, color=col, alpha=0.28,
                    edgecolors="none", zorder=3)
        axB.plot([lo, hi], [ypos, ypos], color=col, lw=2.0,
                 solid_capstyle="butt", zorder=4)
        for xcap in (lo, hi):
            axB.plot([xcap, xcap], [ypos - 0.13, ypos + 0.13], color=col, lw=2.0,
                     solid_capstyle="round", zorder=4)
        axB.scatter([d["delta_mean"]], [ypos], s=115, color=col, zorder=5,
                    edgecolors=SURFACE, linewidths=2.0)
        axB.text(d["delta_mean"], ypos + 0.24, f"Δ = {d['delta_mean']:+.4f}",
                 ha="center", va="bottom", fontsize=10.5, color=INK,
                 fontweight="bold", zorder=6,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        verdict = ("excludes 0 — significant" if sig
                   else "includes 0 — NOT significant")
        axB.text(
            0.006, ypos - 0.45,
            f"95% CI [{lo:+.3f}, {hi:+.3f}] — {verdict}\n"
            f"paired t p = {d['p']:.3f} · TabICL wins {d['wins']}/{d['n_folds']} folds",
            transform=axB.get_yaxis_transform(), ha="left", va="top", fontsize=8.8,
            color=INK_2 if sig else MUTED, linespacing=1.5, zorder=6,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
        )

    all_d = np.concatenate([pts[s]["delta"] for s in schemes])
    lo_all = min(all_d.min(), min(pts[s]["delta_mean"] - pts[s]["delta_half"] for s in schemes))
    hi_all = max(all_d.max(), max(pts[s]["delta_mean"] + pts[s]["delta_half"] for s in schemes))
    span = hi_all - lo_all
    axB.set_xlim(lo_all - 0.22 * span, hi_all + 0.22 * span)
    axB.set_yticks([])
    axB.set_ylim(min(ypos_map.values()) - 1.30, max(ypos_map.values()) + 1.35)
    axB.set_xlabel("Paired Δ AUC per fold  (TabICL − RF-full) — right of 0 = TabICL ahead",
                   fontsize=10, color=INK_2, labelpad=8)
    axB.text(0.0, max(ypos_map.values()) + 1.28, "no difference", fontsize=8.5,
             color=INK_2, ha="center", va="top")
    axB.set_title(
        "B · Paired difference with 95% CI",
        fontsize=11.5, color=INK, loc="left", pad=12, fontweight="bold",
    )

    # ---------------- titles, legend, footnote ------------------------------
    fig.text(0.012, 0.968, "TabICL vs RandomForest (full training fold)",
             fontsize=17, color=INK, fontweight="bold", va="top")
    ctx = pts[schemes[0]]["context_rows"]
    fig.text(
        0.012, 0.912,
        f"{species} · dwelling HIGH/LOW · both models see the identical "
        f"{ctx:,} training rows and are scored on the identical {res.get('n_test', '?')} "
        f"test cells per fold · 5 folds",
        fontsize=9.8, color=INK_2, va="top",
    )

    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=9, color=C_TABICL,
                   markeredgecolor=SURFACE, markeredgewidth=1.6,
                   label="TabICL (tabular foundation model, in-context)"),
        plt.Line2D([], [], marker="o", ls="none", ms=9, color=C_RF,
                   markeredgecolor=SURFACE, markeredgewidth=1.6,
                   label="RF-full (RandomForest, supervised on the same rows)"),
    ]
    leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.988, 0.985),
                     frameon=False, fontsize=10, handletextpad=0.6, labelspacing=0.5)
    for txt in leg.get_texts():
        txt.set_color(INK_2)

    fig.text(
        0.012, 0.022,
        "95% CIs are t-intervals over the 5 CV folds (4 df). Under random CV TabICL "
        "wins 5/5 folds and the CI clears zero, but both models sit at ~0.99 — the "
        "metric is saturated and inflated by spatial autocorrelation.\nUnder "
        "spatial-block CV the +0.037 lead is driven almost entirely by one hard "
        "region (fold 4: TabICL 0.767 vs RF-full 0.611); the CI spans zero, so TabICL "
        "cannot be claimed to beat RandomForest where it matters.",
        fontsize=8.6, color=MUTED, va="bottom", linespacing=1.5,
    )

    out = Path(args.out)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out.resolve()}")
    for s in schemes:
        d = pts[s]
        print(
            f"  {s:8s} TabICL {d['tabicl_ci'][0]:.4f} ±{d['tabicl_ci'][1]:.4f} | "
            f"RF-full {d['rf_ci'][0]:.4f} ±{d['rf_ci'][1]:.4f} | "
            f"Δ {d['delta_mean']:+.4f} [{d['delta_mean']-d['delta_half']:+.4f}, "
            f"{d['delta_mean']+d['delta_half']:+.4f}] p={d['p']:.4f}"
        )


if __name__ == "__main__":
    main()
