#!/usr/bin/env python3
"""
tabicl_vs_rf_roc.py — actual ROC curves for TabICL vs RandomForest (full data),
Caretta caretta, spatial-block CV.

Complements `tabicl_vs_rf_figure.py`, which plots AUC point estimates + CIs. This
one plots the curves those AUCs summarise, which is the only view that shows
*where* on the operating range the two models differ.

Source: figures/Caretta_caretta/tabicl_f1_spatial.json — the deterministic replay
produced by tabicl_f1_benchmark.py, which stores per-cell predicted probabilities
and true labels for all 5 spatial folds (300 balanced test cells each). That file
carries a reproduction check against tabicl_results.json (max abs AUC diff 0.0),
asserted below, so these curves cannot silently decouple from the published AUCs.

Random CV is NOT plotted: only the spatial run stored per-cell probabilities, and
both models exceed 0.98 there, so the curves would be visually degenerate anyway.

  Panel A  all 5 folds (faint) + vertically averaged mean ROC per model
  Panel B  fold 4 alone — the hard held-out region where RF collapses

Markers show each model's actual operating point at the fixed P=0.5 threshold,
which is where TabICL's miscalibration becomes visible: it ranks well but sits
too far down the curve.

Writes figures/Caretta_caretta/26_roc_tabicl_vs_rf_spatial.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

FIG_DIR = os.path.join("figures", "Caretta_caretta")
SRC = "tabicl_f1_spatial.json"
OUT = "26_roc_tabicl_vs_rf_spatial.png"

# Okabe-Ito slots; validated colourblind-safe against the light surface.
C_ICL, C_RF = "#0072B2", "#D55E00"
INK, MUTED, GRID, AXIS = "#1b1b19", "#6b6b66", "#e6e6e2", "#c9c9c4"
GRID_N = 512


def mean_roc(curves):
    """Vertically averaged ROC: interpolate each fold's TPR onto a common FPR grid."""
    grid = np.linspace(0.0, 1.0, GRID_N)
    tprs = [np.interp(grid, fpr, tpr) for fpr, tpr in curves]
    for t in tprs:
        t[0] = 0.0
    return grid, np.mean(tprs, axis=0), np.std(tprs, axis=0)


def operating_point(y, p, thr=0.5):
    """(FPR, TPR) actually achieved at a fixed probability threshold."""
    pred = p >= thr
    pos, neg = y == 1, y == 0
    return (pred & neg).sum() / neg.sum(), (pred & pos).sum() / pos.sum()


def main():
    with open(os.path.join(FIG_DIR, SRC)) as fh:
        d = json.load(fh)

    chk = d["reproduction_check_vs_tabicl_results_json"]
    assert chk["auc_tabicl_max_abs_diff"] == 0.0 and chk["auc_rf_full_max_abs_diff"] == 0.0, (
        f"stored probabilities no longer reproduce the published AUCs: {chk}")

    folds = d["folds"]
    models = [("TabICL", "proba_tabicl", "auc_tabicl", C_ICL),
              ("RandomForest · full data", "proba_rf_full", "auc_rf_full", C_RF)]

    # Recompute AUCs from the stored probabilities rather than trusting the field.
    for f in folds:
        y = np.asarray(f["y_true"])
        for _, pk, ak, _ in models:
            got = roc_auc_score(y, np.asarray(f[pk]))
            assert abs(got - f[ak]) < 1e-9, f"fold {f['fold']} {pk}: {got} vs {f[ak]}"

    worst = min(folds, key=lambda f: f["auc_rf_full"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 6.1))

    # ---- Panel A: all folds + mean ---------------------------------------
    for name, pk, ak, colour in models:
        curves = []
        for f in folds:
            y, p = np.asarray(f["y_true"]), np.asarray(f[pk])
            fpr, tpr, _ = roc_curve(y, p)
            curves.append((fpr, tpr))
            axA.plot(fpr, tpr, color=colour, lw=0.9, alpha=0.30, zorder=2)
        grid, mtpr, _ = mean_roc(curves)
        mean_auc = float(np.mean([f[ak] for f in folds]))
        sd = float(np.std([f[ak] for f in folds], ddof=1))
        axA.plot(grid, mtpr, color=colour, lw=2.8, zorder=4,
                 label=f"{name}\n  mean AUC {mean_auc:.3f}  (sd {sd:.3f})")

    axA.set_title("A · All 5 held-out regions", fontsize=11.5, loc="left", color=INK)
    axA.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95,
               edgecolor="#d8d8d4", labelspacing=0.9)

    # ---- Panel B: the worst region ---------------------------------------
    for name, pk, ak, colour in models:
        y, p = np.asarray(worst["y_true"]), np.asarray(worst[pk])
        fpr, tpr, _ = roc_curve(y, p)
        axB.plot(fpr, tpr, color=colour, lw=2.8, zorder=4,
                 label=f"{name}\n  AUC {worst[ak]:.3f}   F1 {worst['f1_' + pk[6:]]:.3f}")
        ofpr, otpr = operating_point(y, p)
        axB.plot([ofpr], [otpr], marker="o", ms=10, color=colour, zorder=6,
                 markeredgecolor="white", markeredgewidth=1.6)

    axB.annotate("operating points\nat the fixed P = 0.50 cut",
                 xy=(0.52, 0.30), fontsize=8.6, color=MUTED, ha="left")
    axB.set_title(f"B · Fold {worst['fold']} — the hard region "
                  "(RF collapses, TabICL holds)", fontsize=11.5, loc="left", color=INK)
    axB.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95,
               edgecolor="#d8d8d4", labelspacing=0.9)

    for ax in (axA, axB):
        ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.2, zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)

    fig.suptitle("Caretta caretta — ROC under spatial-block CV · "
                 "TabICL (no training) vs. RandomForest (fitted on the full grid)",
                 fontsize=13, x=0.008, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG_DIR, OUT)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[saved] {out}")
    print(f"  reproduction check vs tabicl_results.json: {chk}")
    print(f"  worst fold = {worst['fold']}")
    print(f"\n  {'fold':>5}{'AUC TabICL':>13}{'AUC RF':>10}{'F1 TabICL':>12}{'F1 RF':>9}"
          f"{'predHIGH ICL':>14}{'predHIGH RF':>13}")
    for f in folds:
        print(f"  {f['fold']:>5}{f['auc_tabicl']:>13.4f}{f['auc_rf_full']:>10.4f}"
              f"{f['f1_tabicl']:>12.4f}{f['f1_rf_full']:>9.4f}"
              f"{f['pred_pos_rate_tabicl']:>14.3f}{f['pred_pos_rate_rf_full']:>13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
