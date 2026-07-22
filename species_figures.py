#!/usr/bin/env python3
"""
species_figures.py — richer, informative figure battery for the multi-species
retention SDM (per-species folders + the pooled folder).

The minimal `summary.png` written by species_pipeline.py shows only the retention
map, the sightings map, the rho scatter and one ROC. This module adds the full
machine-learning diagnostic set the paper-style analysis warrants:

  per species  (figures/<species>/):
    00_summary.png                 (kept, written by species_pipeline)
    01_fields.png                  retention + weighted-sightings maps
    02_roc_random_vs_spatial.png   THE headline: random-CV ROC vs spatial-block ROC
    03_pr_confusion.png            precision-recall + confusion matrix (spatial holdout)
    04_permutation_importance.png  model-agnostic importance, retention highlighted
    05_gbr_pred_vs_obs.png         GBR predicted-vs-observed, random vs spatial
    06_predicted_prob_map.png      RF P(high-use) surface vs actual high cells
    07_feature_correlation.png     feature + target correlation heatmap
    08_shap_summary.png            SHAP beeswarm (skipped gracefully if shap absent)

  pooled  (figures/pooled/):
    01_cv_auc.png                  leave-region-out vs leave-species-out AUC
    02_roc_transfer.png            ROC for both pooled CV schemes
    03_per_species_metrics.png     AUC & R^2, random vs spatial, per species
    04_per_species_rho.png         per-species Spearman rho with bootstrap CI
    05_pooled_perm_importance.png  pooled permutation importance (incl. species one-hot)

Two entry points:
  * the make_*_figures() functions, called by species_pipeline.py with objects it
    already computed (no recompute);
  * a standalone CLI that reconstructs the per-cell data from the *cached*
    retention field (fast — no re-simulation) and regenerates every figure:

        python species_figures.py \
          --run "Caretta caretta:Datasets/Oceanic_data.nc:Datasets/caretta_data.csv" \
          --run "Cetorhinus maximus:Datasets/NE_Atlantic.nc:Datasets/<multispp>.csv"
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import (roc_auc_score, roc_curve, r2_score,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)
from sklearn.inspection import permutation_importance

import retention_core as rc

warnings.filterwarnings("ignore")

DPI = 150


# ---------------------------------------------------------------------------
#  model factories (kept identical to species_pipeline.cv_scores)
# ---------------------------------------------------------------------------
def _rf(seed=rc.SEED):
    return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed,
                                  class_weight="balanced")


def _gbr(seed=rc.SEED):
    return GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                     random_state=seed)


def _oof_proba(make_model, X, y, splitter, groups=None):
    """Out-of-fold P(class=1) across a splitter; NaN where a cell was never tested."""
    proba = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_model().fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    return proba


def _oof_pred(make_model, X, y, splitter, groups=None):
    """Out-of-fold regression predictions across a splitter."""
    pred = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def _splitters(y_cls, groups, seed=rc.SEED, n_splits=5):
    """Return (random_cls, random_reg, spatial) splitters; spatial may be None."""
    ng = len(np.unique(groups))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gkf = GroupKFold(n_splits=min(n_splits, ng)) if ng >= 2 else None
    return skf, kf, gkf


# ===========================================================================
#  PER-SPECIES FIGURES
# ===========================================================================
def make_species_figures(out_dir, species, vel, fields, X, y_reg, y_cls, groups, stats):
    """Write the full per-species figure battery. All inputs are already computed
    by species_pipeline.run_species (or reconstructed from cache by the CLI)."""
    os.makedirs(out_dir, exist_ok=True)
    Xs = StandardScaler().fit_transform(X)
    skf, kf, gkf = _splitters(y_cls, groups)
    lon, lat = vel["lon"], vel["lat"]

    _fig_fields(out_dir, species, lon, lat, fields, stats)
    _fig_roc(out_dir, species, Xs, y_cls, skf, gkf, groups)
    _fig_pr_confusion(out_dir, species, Xs, y_cls, gkf, groups)
    _fig_perm_importance(out_dir, species, Xs, y_cls, gkf, groups)
    _fig_gbr(out_dir, species, Xs, y_reg, kf, gkf, groups)
    _fig_prob_map(out_dir, species, Xs, y_cls, groups, fields, X)
    _fig_feature_corr(out_dir, species, X, y_reg)
    _fig_shap(out_dir, species, Xs, y_reg)
    print(f"  [figures] wrote diagnostic set -> {out_dir}/")


def _fig_fields(out_dir, species, lon, lat, fields, stats):
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    im0 = ax[0].pcolormesh(lon, lat, fields["retention"], shading="auto", cmap="YlOrRd")
    ax[0].set_title("Ensemble particle retention"); plt.colorbar(im0, ax=ax[0], shrink=0.85)
    im1 = ax[1].pcolormesh(lon, lat, fields["sights"], shading="auto", cmap="YlGnBu")
    ax[1].set_title(f"{species}\nweighted sightings"); plt.colorbar(im1, ax=ax[1], shrink=0.85)
    mask = (fields["retention"] > 0) | (fields["sights"] > 0)
    ax[2].scatter(fields["retention"][mask], fields["sights"][mask], s=4, alpha=0.35, color="#444")
    ax[2].set_xlabel("retention (norm)"); ax[2].set_ylabel("sightings (norm)")
    ci = stats.get("ci", [np.nan, np.nan])
    ax[2].set_title(f"rho={stats['rho']:.3f}  perm p={stats['perm_p']:.3f}\n"
                    f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  n={stats['n']}")
    for a in ax[:2]:
        a.set_xlabel("lon"); a.set_ylabel("lat")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_fields.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def _fig_roc(out_dir, species, Xs, y_cls, skf, gkf, groups):
    """Headline figure: random-CV ROC (optimistic) vs spatial-block ROC (honest).
    The gap between them is the spatial-autocorrelation inflation."""
    fig, ax = plt.subplots(figsize=(6.4, 6))
    p_rand = _oof_proba(_rf, Xs, y_cls, skf)
    m = ~np.isnan(p_rand)
    if m.sum() and len(np.unique(y_cls[m])) > 1:
        fpr, tpr, _ = roc_curve(y_cls[m], p_rand[m])
        auc = roc_auc_score(y_cls[m], p_rand[m])
        ax.plot(fpr, tpr, lw=2.4, color="#d1495b", label=f"random CV   AUC={auc:.3f}")
    if gkf is not None:
        p_sp = _oof_proba(_rf, Xs, y_cls, gkf, groups)
        m = ~np.isnan(p_sp)
        if m.sum() and len(np.unique(y_cls[m])) > 1:
            fpr, tpr, _ = roc_curve(y_cls[m], p_sp[m])
            auc = roc_auc_score(y_cls[m], p_sp[m])
            ax.plot(fpr, tpr, lw=2.4, color="#1f7a8c", label=f"spatial-block CV   AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.6, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{species} — RF high-use ROC\nrandom CV is optimistic; spatial CV is honest")
    ax.legend(loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_roc_random_vs_spatial.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def _fig_pr_confusion(out_dir, species, Xs, y_cls, gkf, groups):
    """Precision-recall + confusion matrix under the honest spatial holdout."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    if gkf is not None:
        p = _oof_proba(_rf, Xs, y_cls, gkf, groups)
        m = ~np.isnan(p)
        if m.sum() and len(np.unique(y_cls[m])) > 1:
            prec, rec, _ = precision_recall_curve(y_cls[m], p[m])
            ap = average_precision_score(y_cls[m], p[m])
            ax[0].plot(rec, prec, lw=2.2, color="#1f7a8c", label=f"AP={ap:.3f}")
            base = y_cls[m].mean()
            ax[0].axhline(base, ls="--", color="k", lw=0.8, label=f"prevalence={base:.3f}")
            cm = confusion_matrix(y_cls[m], (p[m] >= 0.5).astype(int))
            im = ax[1].imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax[1].text(j, i, str(v), ha="center", va="center",
                           color="white" if v > cm.max() / 2 else "black", fontsize=13)
            ax[1].set_xticks([0, 1]); ax[1].set_yticks([0, 1])
            ax[1].set_xticklabels(["low", "high"]); ax[1].set_yticklabels(["low", "high"])
            ax[1].set_xlabel("predicted"); ax[1].set_ylabel("actual")
            plt.colorbar(im, ax=ax[1], shrink=0.8)
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
    ax[0].set_title("Precision-recall (spatial holdout)"); ax[0].legend(loc="upper right")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02)
    ax[1].set_title("Confusion matrix @0.5 (spatial holdout)")
    fig.suptitle(f"{species}", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_pr_confusion.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def _fig_perm_importance(out_dir, species, Xs, y_cls, gkf, groups):
    """Model-agnostic permutation importance (spatial holdout) — honest unlike Gini
    which inflates the high-cardinality lat/lon. Retention is highlighted."""
    if gkf is None or y_cls.sum() == 0:
        return
    ng = len(np.unique(groups))
    tr, te = next(GroupKFold(n_splits=min(5, ng)).split(Xs, y_cls, groups))
    if y_cls[tr].sum() == 0 or y_cls[te].sum() == 0:
        return
    rf = _rf().fit(Xs[tr], y_cls[tr])
    pi = permutation_importance(rf, Xs[te], y_cls[te], n_repeats=30,
                                random_state=rc.SEED, scoring="roc_auc")
    order = np.argsort(pi.importances_mean)
    names = np.array(rc.FEATURES)[order]
    means = pi.importances_mean[order]
    errs = pi.importances_std[order]
    colors = ["#d1495b" if n == "retention" else "#8d99ae" for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(names, means, xerr=errs, color=colors, capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("permutation importance (Δ AUC when shuffled)")
    ax.set_title(f"{species} — feature importance (spatial holdout)\nretention in red")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_permutation_importance.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def _fig_gbr(out_dir, species, Xs, y_reg, kf, gkf, groups):
    """GBR predicted-vs-observed sighting density: random vs spatial holdout."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for k, (split, grp, title) in enumerate([(kf, None, "random CV"),
                                              (gkf, groups, "spatial-block CV")]):
        if split is None:
            ax[k].text(0.5, 0.5, "n/a (single block)", ha="center", va="center")
            ax[k].set_title(title); continue
        pred = _oof_pred(_gbr, Xs, y_reg, split, grp)
        m = ~np.isnan(pred)
        r2 = r2_score(y_reg[m], pred[m]) if m.sum() > 2 else np.nan
        ax[k].scatter(y_reg[m], pred[m], s=6, alpha=0.3, color="#1f7a8c")
        lim = [min(y_reg[m].min(), pred[m].min()), max(y_reg[m].max(), pred[m].max())]
        ax[k].plot(lim, lim, "k--", lw=0.9)
        ax[k].set_xlabel("observed sightings (norm)"); ax[k].set_ylabel("predicted")
        ax[k].set_title(f"{title}\nR$^2$={r2:.3f}")
    fig.suptitle(f"{species} — GBR regression", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_gbr_pred_vs_obs.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def _fig_prob_map(out_dir, species, Xs, y_cls, groups, fields, X):
    """RF predicted P(high-use) over the domain (trained on a spatial holdout's
    train split) vs the actual high-use cells."""
    ng = len(np.unique(groups))
    if ng >= 2 and y_cls.sum():
        tr, te = next(GroupKFold(n_splits=min(5, ng)).split(Xs, y_cls, groups))
        if y_cls[tr].sum() == 0:
            tr = np.arange(len(y_cls))
        rf = _rf().fit(Xs[tr], y_cls[tr])
    else:
        rf = _rf().fit(Xs, y_cls)
    prob = rf.predict_proba(Xs)[:, 1]
    lons = X[:, rc.FEATURES.index("longitude")]
    lats = X[:, rc.FEATURES.index("latitude")]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sc = ax[0].scatter(lons, lats, c=prob, s=8, cmap="magma", vmin=0, vmax=1)
    ax[0].set_title("RF predicted P(high-use)"); plt.colorbar(sc, ax=ax[0], shrink=0.85)
    hi = y_cls == 1
    ax[1].scatter(lons, lats, s=6, color="#dddddd", label="all cells")
    ax[1].scatter(lons[hi], lats[hi], s=12, color="#d1495b", label="actual high-use")
    ax[1].set_title(f"{species} — actual high-use cells"); ax[1].legend(loc="best")
    for a in ax:
        a.set_xlabel("lon"); a.set_ylabel("lat")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_predicted_prob_map.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def _fig_feature_corr(out_dir, species, X, y_reg):
    M = np.column_stack([X, y_reg])
    labels = rc.FEATURES + ["sightings"]
    C = np.corrcoef(M, rowvar=False)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for (i, j), v in np.ndenumerate(C):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.6 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(f"{species} — feature & target correlation")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_feature_correlation.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def _fig_shap(out_dir, species, Xs, y_reg):
    try:
        import shap
    except Exception:
        print("  [figures] shap not installed — skipping 08_shap_summary.png")
        return
    gbr = _gbr().fit(Xs, y_reg)
    try:
        expl = shap.TreeExplainer(gbr)
        sv = expl.shap_values(Xs)
    except Exception as e:
        print(f"  [figures] SHAP failed ({e}) — skipping")
        return
    fig = plt.figure(figsize=(8, 5.5))
    shap.summary_plot(sv, Xs, feature_names=rc.FEATURES, show=False, plot_size=None)
    plt.title(f"{species} — SHAP (GBR sightings)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_shap_summary.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


# ===========================================================================
#  POOLED FIGURES
# ===========================================================================
def make_pooled_figures(pooled_dir, usable, Xsc, y, g_region, g_species,
                        n_region_folds, n_species_folds, names, res):
    os.makedirs(pooled_dir, exist_ok=True)
    auc_r = res["auc_leave_region_out"]
    auc_s = res["auc_leave_species_out"]

    # 01 — CV AUC comparison
    fig, ax = plt.subplots(figsize=(6.4, 5))
    bars = ax.bar(["leave-region-out\n(interpolation)", "leave-species-out\n(transfer)"],
                  [auc_r, auc_s], color=["#1f7a8c", "#d1495b"])
    ax.axhline(0.5, color="k", ls="--", lw=0.9, label="chance")
    ax.set_ylim(0, 1); ax.set_ylabel("AUC")
    ax.set_title(f"Pooled model ({len(names)} species)")
    for b, v in zip(bars, [auc_r, auc_s]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(pooled_dir, "01_cv_auc.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)

    # 02 — ROC for both transfer schemes
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for groups, nf, color, lab in [(g_region, n_region_folds, "#1f7a8c", "leave-region-out"),
                                    (g_species, n_species_folds, "#d1495b", "leave-species-out")]:
        p = _oof_proba(_rf, Xsc, y, GroupKFold(n_splits=nf), groups)
        m = ~np.isnan(p)
        if m.sum() and len(np.unique(y[m])) > 1:
            fpr, tpr, _ = roc_curve(y[m], p[m])
            ax.plot(fpr, tpr, lw=2.3, color=color,
                    label=f"{lab}  AUC={roc_auc_score(y[m], p[m]):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.6, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Pooled model ROC — interpolation vs transfer")
    ax.legend(loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(pooled_dir, "02_roc_transfer.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)

    # 03 — per-species AUC & R^2, random vs spatial
    sp_names = [r["species"] for r in usable]
    x = np.arange(len(sp_names)); w = 0.2
    fig, ax = plt.subplots(figsize=(max(7, 2 * len(sp_names) + 4), 5))
    ax.bar(x - 1.5 * w, [r["cv"]["auc_random"] for r in usable], w, label="AUC random", color="#d1495b")
    ax.bar(x - 0.5 * w, [r["cv"]["auc_spatial"] for r in usable], w, label="AUC spatial", color="#edae49")
    ax.bar(x + 0.5 * w, [r["cv"]["r2_random"] for r in usable], w, label="R$^2$ random", color="#1f7a8c")
    ax.bar(x + 1.5 * w, [max(r["cv"]["r2_spatial"], 0) for r in usable], w, label="R$^2$ spatial", color="#66a182")
    ax.axhline(0.5, color="k", ls=":", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(sp_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Per-species skill: random CV (inflated) vs spatial-block CV (honest)")
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(pooled_dir, "03_per_species_metrics.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)

    # 04 — per-species rho with bootstrap CI
    rhos = [r["stats"]["rho"] for r in usable]
    cis = np.array([r["stats"]["ci"] for r in usable])
    lo = np.array(rhos) - cis[:, 0]; hi = cis[:, 1] - np.array(rhos)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.8 * len(sp_names) + 2)))
    ax.errorbar(rhos, range(len(sp_names)), xerr=[lo, hi], fmt="o", color="#1f7a8c",
                capsize=4, ms=8)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(range(len(sp_names))); ax.set_yticklabels(sp_names)
    ax.set_xlabel("Spearman rho (retention vs sightings)")
    ax.set_title("Per-species retention–sightings correlation\n(bars = bootstrap 95% CI)")
    plt.tight_layout()
    fig.savefig(os.path.join(pooled_dir, "04_per_species_rho.png"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)

    # 05 — pooled permutation importance (features + species one-hot)
    tr, te = next(GroupKFold(n_splits=n_region_folds).split(Xsc, y, g_region))
    if len(np.unique(y[tr])) > 1 and len(np.unique(y[te])) > 1:
        rf = _rf().fit(Xsc[tr], y[tr])
        pi = permutation_importance(rf, Xsc[te], y[te], n_repeats=30,
                                    random_state=rc.SEED, scoring="roc_auc")
        labels = rc.FEATURES + [f"is:{n}" for n in sp_names]
        labels = labels[:Xsc.shape[1]]
        order = np.argsort(pi.importances_mean)
        lab = np.array(labels)[order]
        means = pi.importances_mean[order]; errs = pi.importances_std[order]
        colors = ["#d1495b" if n == "retention" else
                  ("#edae49" if n.startswith("is:") else "#8d99ae") for n in lab]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.barh(lab, means, xerr=errs, color=colors, capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("permutation importance (Δ AUC)")
        ax.set_title("Pooled model importance (region holdout)\nretention red, species-ID amber")
        plt.tight_layout()
        fig.savefig(os.path.join(pooled_dir, "05_pooled_perm_importance.png"),
                    bbox_inches="tight", dpi=DPI)
        plt.close(fig)
    print(f"  [figures] wrote pooled diagnostic set -> {pooled_dir}/")


# ===========================================================================
#  Standalone regeneration from cache (no re-simulation)
# ===========================================================================
def _reconstruct(species, velocity_nc, species_csv, fig_dir, max_steps=rc.MAX_STEPS):
    """Rebuild per-cell data using the CACHED retention field — fast, no sim."""
    tag = species.replace(" ", "_").replace("/", "_")
    out_dir = os.path.join(fig_dir, tag)
    vel = rc.load_velocity(velocity_nc)
    cache = os.path.join(out_dir,
                         f".retention_{os.path.basename(velocity_nc)}_{vel['u'].shape}_{max_steps}.npy")
    matches = glob.glob(os.path.join(out_dir, ".retention_*.npy"))
    if os.path.exists(cache):
        retention = np.load(cache)
    elif matches:
        retention = np.load(matches[0])
        print(f"  [regen] using cached retention {os.path.basename(matches[0])}")
    else:
        raise FileNotFoundError(f"no cached retention in {out_dir}; run species_pipeline.py first")
    df_med = rc.load_sightings(species_csv, species, vel)
    X, y_reg, lats, lons, fields = rc.build_features(vel, retention, df_med)
    thr = np.percentile(y_reg[y_reg > 0], 50) if (y_reg > 0).any() else 0.01
    y_cls = (y_reg > thr).astype(int)
    groups = rc.spatial_groups(lats, lons)
    # stats: prefer the saved results.json (matches the pipeline's N=1000 run)
    rj = os.path.join(out_dir, "results.json")
    if os.path.exists(rj):
        stats = json.load(open(rj))["stats"]
    else:
        rho, p = spearmanr(*( (lambda mm: (fields["retention"][mm], fields["sights"][mm]))
                              ((fields["retention"] > 0) | (fields["sights"] > 0)) ))
        stats = {"rho": float(rho), "p": float(p), "perm_p": np.nan,
                 "ci": [np.nan, np.nan], "n": int(((fields["retention"] > 0) | (fields["sights"] > 0)).sum())}
    return dict(out_dir=out_dir, species=species, vel=vel, fields=fields, X=X,
                y_reg=y_reg, y_cls=y_cls, groups=groups, stats=stats)


def _regen_pooled(per, fig_dir, seed=rc.SEED):
    usable = [p for p in per]
    if len(usable) < 2:
        return
    pooled_dir = os.path.join(fig_dir, "pooled")
    names = [p["species"] for p in usable]
    Xs, ys, gr, gsp, oh = [], [], [], [], []
    off = 0
    for si, p in enumerate(usable):
        n = len(p["X"])
        Xs.append(p["X"]); ys.append(p["y_cls"])
        gr.append(p["groups"] + off); gsp.append(np.full(n, si))
        o = np.zeros((n, len(usable))); o[:, si] = 1; oh.append(o)
        off += int(p["groups"].max()) + 1
    Xfull = np.hstack([np.vstack(Xs), np.vstack(oh)])
    y = np.concatenate(ys)
    g_region = np.concatenate(gr); g_species = np.concatenate(gsp)
    Xsc = StandardScaler().fit_transform(Xfull)
    n_region_folds = min(5, len(np.unique(g_region)))
    n_species_folds = len(np.unique(g_species))
    # build a results-shaped dict expected by make_pooled_figures
    usable_res = []
    for p in usable:
        rj = os.path.join(p["out_dir"], "results.json")
        usable_res.append(json.load(open(rj)) if os.path.exists(rj)
                          else {"species": p["species"], "stats": p["stats"],
                                "cv": {"auc_random": np.nan, "auc_spatial": np.nan,
                                       "r2_random": np.nan, "r2_spatial": np.nan}})
    pr = os.path.join(pooled_dir, "pooled_results.json")
    res = json.load(open(pr)) if os.path.exists(pr) else {
        "auc_leave_region_out": np.nan, "auc_leave_species_out": np.nan}
    make_pooled_figures(pooled_dir, usable_res, Xsc, y, g_region, g_species,
                        n_region_folds, n_species_folds, names, res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[],
                    help="species:velocity_nc:species_csv  (repeatable)")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args(argv)
    if not args.run:
        ap.error("provide one or more --run species:velocity_nc:species_csv")
    runs = [tuple(r.split(":", 2)) for r in args.run]
    per = []
    for sp, vel, csv in runs:
        print(f"\n=== {sp} ===")
        d = _reconstruct(sp, vel, csv, args.fig_dir)
        make_species_figures(d["out_dir"], d["species"], d["vel"], d["fields"],
                             d["X"], d["y_reg"], d["y_cls"], d["groups"], d["stats"])
        per.append(d)
    if len(per) >= 2:
        print("\n=== pooled ===")
        _regen_pooled(per, args.fig_dir)
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
