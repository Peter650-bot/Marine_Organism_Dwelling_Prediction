#!/usr/bin/env python3
"""
spatial_kshot_figure.py — single-panel spatial-block-CV k-shot learning curve for
Caretta caretta, adding TabICL to the TabLLM/classical comparison.

This is the honest-CV analogue of the right panel of
`figures/Caretta_caretta/21_tabllm_learning_curve*.png`, with two additions:

  * the TabICL k-shot curve (from tabicl_results_kshot.json), and
  * TabICL's FULL-CONTEXT spatial score as a reference line, because that -- not
    the k-shot points -- is the configuration the paper's headline rests on.

Reads committed artefacts only; nothing is re-run or re-fit. Writes
figures/Caretta_caretta/25_spatial_kshot_tabicl.png

Baseline provenance: the classical k-shot baselines are taken from the TabICL
results file, so every series in the k-shot panel comes from one internally
consistent run. The TabLLM run is paired to it (same folds, exemplars and test
cells, seed_variant="full").

Cross-file agreement, measured 2026-08-11 -- LogisticRegression and kNN are
BIT-IDENTICAL across the two files and GradientBoosting differs by <=0.001, which
confirms the pairing is real. RandomForest is the sole exception (max 0.0253, at
spatial k=32): it is the only baseline with substantial internal randomness
(bootstrap resampling + per-split feature subsampling), so a differing
random_state between the two harnesses moves it while leaving the deterministic
learners untouched. The assertion below pins that ceiling so the gap cannot widen
unnoticed; if it ever trips, re-check the seeding before re-plotting.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join("figures", "Caretta_caretta")
SCHEME = "spatial"
SHOTS = [8, 16, 32]

# Okabe-Ito subset; validated colourblind-safe (worst adjacent CVD dE 9.6 deutan,
# normal-vision dE 20.0). Marker shapes carry identity as well as colour, so the
# series stay separable in greyscale and under the contrast WARN.
STYLE = {
    "tabicl": dict(color="#0072B2", marker="o", lw=2.6, ms=9,  z=5, label="TabICL (tabular FM)"),
    "llm":    dict(color="#D55E00", marker="s", lw=2.6, ms=8,  z=5, label="TabLLM · Qwen2.5-72B"),
    "rf":     dict(color="#009E73", marker="^", lw=1.4, ms=6,  z=3, label="RandomForest"),
    "gbt":    dict(color="#E69F00", marker="v", lw=1.4, ms=6,  z=3, label="GradientBoosting"),
    "lr":     dict(color="#CC79A7", marker="D", lw=1.4, ms=5,  z=3, label="LogisticRegression"),
    "knn":    dict(color="#56B4E9", marker="P", lw=1.4, ms=6,  z=3, label="kNN"),
}


def _load(name):
    with open(os.path.join(FIG_DIR, name)) as fh:
        return json.load(fh)


def _by_k(points, key, scheme=SCHEME):
    """Pull {k: value} for one metric key, restricted to `scheme`."""
    return {p["k"]: p[key] for p in points
            if p["scheme"] == scheme and p.get("k") in SHOTS}


def main():
    icl = _load("tabicl_results_kshot.json")
    llm = _load("tabllm_results.json")
    full = _load("tabicl_results_full.json")

    # TabLLM's curve: variant="full" is the like-for-like serialization (all features).
    llm_pts = [p for p in llm["points"] if p.get("variant") == "full"]

    series = {
        "tabicl": _by_k(icl["points"], "auc_tabicl"),
        "llm":    _by_k(llm_pts,       "auc_llm"),
        "rf":     _by_k(icl["points"], "auc_rf_kshot"),
        "gbt":    _by_k(icl["points"], "auc_gbt_kshot"),
        "lr":     _by_k(icl["points"], "auc_lr_kshot"),
        "knn":    _by_k(icl["points"], "auc_knn_kshot"),
    }

    # Guard the cross-file baseline agreement rather than trusting it. Reported
    # per series, because WHICH learner drifts is the diagnostic (see docstring).
    drift = {}
    for key, llm_key in [("rf", "auc_rf_kshot"), ("gbt", "auc_gbt_kshot"),
                         ("lr", "auc_lr_kshot"), ("knn", "auc_knn_kshot")]:
        other = _by_k(llm_pts, llm_key)
        drift[key] = max(abs(series[key][k] - other[k]) for k in SHOTS)
    assert drift["rf"] <= 0.030, f"RandomForest drift grew to {drift['rf']:.4f}"
    assert max(drift[k] for k in ("gbt", "lr", "knn")) <= 0.002, (
        f"a deterministic baseline drifted -- the runs may no longer be paired: {drift}")

    rf_full = _by_k(icl["points"], "auc_rf_full")
    icl_full = next(p["auc_tabicl"] for p in full["points"] if p["scheme"] == SCHEME)
    rf_full_mean = sum(rf_full.values()) / len(rf_full)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    # --- reference lines (recessive, drawn first) ---------------------------
    # Labelled in-plot in the empty band below 0.5 between k=16 and k=32, so the
    # text cannot collide with the k=32 markers sitting at ~0.50-0.51.
    ax.axhline(0.5, color="#9a9a96", ls=":", lw=1.2, zorder=1)
    ax.text(23, 0.474, "chance (0.50)", va="center", ha="center",
            fontsize=8.5, color="#6b6b66")

    ax.plot(SHOTS, [rf_full[k] for k in SHOTS], color="#3a3a38", ls="--", lw=1.8, zorder=2)
    ax.text(32.6, rf_full_mean, "RF · full data (ceiling)", va="center",
            ha="left", fontsize=8.5, color="#3a3a38")

    ax.axhline(icl_full, color=STYLE["tabicl"]["color"], ls="-.", lw=2.0, alpha=0.85, zorder=2)
    ax.text(32.6, icl_full, "TabICL · full context", va="center", ha="left",
            fontsize=8.5, color=STYLE["tabicl"]["color"], fontweight="bold")

    # --- k-shot curves ------------------------------------------------------
    for key, vals in series.items():
        s = STYLE[key]
        ax.plot(SHOTS, [vals[k] for k in SHOTS], color=s["color"], marker=s["marker"],
                lw=s["lw"], ms=s["ms"], zorder=s["z"], markeredgecolor="white",
                markeredgewidth=0.9, label=s["label"])

    ax.set_xlabel("labelled examples in context, k")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(SHOTS)
    ax.set_xlim(6.6, 32.9)
    ax.set_ylim(0.36, 0.95)
    ax.grid(axis="y", color="#e6e6e2", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c9c4")

    # Only the k-shot series go in the legend -- the two reference lines are
    # labelled directly on the plot, so repeating them here would just cost space.
    # Anchored into the empty band between the highest k-shot point (~0.684) and
    # the RF ceiling (~0.871) so it cannot occlude either reference line.
    ax.legend(fontsize=8.4, loc="upper left", bbox_to_anchor=(0.015, 0.845),
              borderaxespad=0.0, frameon=True, framealpha=0.95,
              edgecolor="#d8d8d4", ncol=2, columnspacing=1.1, handlelength=2.4,
              borderpad=0.6)

    fig.subplots_adjust(right=0.78)
    out = os.path.join(FIG_DIR, "25_spatial_kshot_tabicl.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[saved] {out}")
    print("  cross-file baseline drift (TabICL file vs TabLLM file): "
          + ", ".join(f"{k}={v:.4f}" for k, v in drift.items()))
    print(f"  TabICL full-context (spatial): {icl_full:.4f}")
    print(f"  RF-full ceiling (spatial, mean over k): {rf_full_mean:.4f}")
    print("\n  k-shot values plotted (spatial CV):")
    print("    k  " + "".join(f"{STYLE[s]['label'][:12]:>14}" for s in series))
    for k in SHOTS:
        print(f"   {k:>2}  " + "".join(f"{series[s][k]:>14.3f}" for s in series))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
