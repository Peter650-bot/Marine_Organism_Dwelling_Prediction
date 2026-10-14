#!/usr/bin/env python3
"""
evaluate.py -- TabICLv2 vs MaxEnt on MarineSPEED spatial folds: paired
comparison with the species as the replication unit.

    results/summary.json
    figures/auc_paired.{png,pdf}   per-species AUC, TabICLv2 vs MaxEnt

Folds: MarineSPEED's spatial "pseudodisc" 5-fold CV only. Test presences lie
in a held-out disc and the background within 200 km of it is excluded, so a
model cannot score by interpolating between nearby training presences.

Per-fold AUC = held-out presences vs held-out background (the standard
presence-only AUC). A species' AUC is the mean of its 5 folds. Tests across
species: Wilcoxon signed-rank on the paired difference, and a 10,000-replicate
bootstrap over species (seed 0) for the 95% CI of the mean difference.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import runner  # noqa: E402
from make_context_sweep_figure import GRID, INK_2, INK_3, SURFACE, TABICL, style  # noqa: E402

RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
MAXENT = "#2f9e6e"
MODELS = [("tabicl", "TabICLv2", TABICL), ("maxent", "MaxEnt", MAXENT)]
KIND = "pseudodisc"
B, BOOT_SEED = 10_000, 0


def load(model, prefix=""):
    with open(os.path.join(RES, f"{prefix}{model}.jsonl")) as fh:
        recs = [json.loads(line) for line in fh]
    for r in recs:      # every logged AUC must match the stored scores
        assert abs(runner.auc(r["y_test"], r["p_test"]) - r["auc"]) < 1e-5
    return recs


def species_auc(recs):
    """{species: mean AUC over the 5 spatial folds}, complete species only."""
    by = {}
    for r in recs:
        if r["kind"] == KIND:
            by.setdefault(r["species"], {})[r["k"]] = r["auc"]
    return {sp: float(np.mean(list(v.values()))) for sp, v in by.items() if len(v) == 5}


def paired(d, rng):
    from scipy.stats import wilcoxon
    d = np.asarray(d)
    boot = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": float(wilcoxon(d).pvalue), "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum())}


def main():
    recs = {m: load(m) for m, *_ in MODELS}
    auc = {m: species_auc(r) for m, r in recs.items()}
    species = sorted(set(auc["tabicl"]) & set(auc["maxent"]))
    rng = np.random.default_rng(BOOT_SEED)
    t = np.array([auc["tabicl"][sp] for sp in species])
    m = np.array([auc["maxent"][sp] for sp in species])
    s = paired(t - m, rng)
    out = {"folds": "MarineSPEED pseudodisc (spatial) 5-fold", "n_species": len(species),
           "mean_auc": {"tabicl": float(t.mean()), "maxent": float(m.mean())},
           "tabicl_minus_maxent": s}

    print(f"{len(species)} species with all 5 spatial folds for both models\n")
    print(f"mean ROC-AUC   TabICLv2 {t.mean():.4f}   MaxEnt {m.mean():.4f}")
    print(f"TabICLv2 - MaxEnt  {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  "
          f"median {s['median']:+.4f}  Wilcoxon p={s['wilcoxon_p']:.2g}  "
          f"TabICLv2 higher in {s['wins']}/{s['n']}")

    print("\nwall-clock fit+predict per fold (s): median [min, max]")
    for key, name, _ in MODELS:
        v = np.array([r["total_s"] for r in recs[key] if r["species"] in species])
        out[f"time_{key}"] = {"median": float(np.median(v)), "min": float(v.min()),
                              "max": float(v.max()), "total_h": float(v.sum() / 3600)}
        print(f"  {name:9s} {np.median(v):7.1f} [{v.min():.1f}, {v.max():.1f}]  "
              f"total {v.sum() / 3600:.2f} h")

    out["per_species"] = {sp: {"tabicl": auc["tabicl"][sp], "maxent": auc["maxent"][sp]}
                          for sp in species}
    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[saved] {os.path.join(RES, 'summary.json')}")
    plot_paired(t, m, s)


def _save(fig, name, dpi=300):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIG, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {path}")


def plot_paired(t, m, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor=SURFACE)
    lo = max(0.0, np.floor(min(t.min(), m.min()) * 20) / 20)
    ax.plot([lo, 1], [lo, 1], color=INK_3, lw=1.0, ls=":", zorder=1)
    ax.scatter(m, t, s=24, color=np.where(t > m, TABICL, MAXENT), alpha=0.85,
               edgecolor=SURFACE, lw=0.6, zorder=3)
    ax.text(0.04, 0.96, f"TabICLv2 higher: {s['wins']}/{s['n']} species\n"
            f"mean diff {s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]\n"
            f"Wilcoxon p = {s['wilcoxon_p']:.2g}",
            transform=ax.transAxes, va="top", fontsize=7.8, color=INK_2)
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("MaxEnt ROC-AUC (spatial folds)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("TabICLv2 ROC-AUC (spatial folds)", fontsize=9.5, color=INK_2)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.tight_layout()
    _save(fig, "auc_paired")
    plt.close(fig)


if __name__ == "__main__":
    main()
