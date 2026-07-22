#!/usr/bin/env python3
"""
ablation_study.py — systematic feature-ablation for the retention SDM.

Question: how much predictive skill does the *retention* feature add, net of
geography and the static flow field, and is that skill real (not just retention
acting as a proxy for a spatial coordinate)?

The retention field is reconstructed with the SAME parameters and SAME seeds as
enhanced_analysis.py (Sections 1-2), so this script is consistent with figures
01-19. It then evaluates feature subsets with the same model families used in
the paper (Random Forest classifier -> AUC, Gradient Boosting regressor -> R^2),
under TWO validation schemes:

  * random      : standard stratified / k-fold CV (in-domain skill, optimistic)
  * spatial-block: leave-spatial-block-out CV (honest under spatial autocorrelation)

Experiments
-----------
  Cumulative tiers   B0 geography      [lat, lon]
                     B1 + static ocean [+ mean_speed, vorticity, dist_to_coast]
                     B2 + retention    (full model)            <- B2-B1 = value of retention
  LOFO               drop each feature singly from the full set
  Geography-control  full set MINUS [lat, lon] (does retention carry non-positional info?)
  Spatial-null       retention shuffled within latitude bands (proxy-for-coordinate check)
  Permutation imp.   model-agnostic importance on the full RF (fairer than Gini)

Outputs a printed table and figures/20_ablation.png (does NOT touch 01-19).

Run from the Datasets/ directory (same convention as enhanced_analysis.py), or
pass --velocity / --species-csv / --fig-dir explicitly.
"""

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score

import retention_core as rc  # shared, validated pipeline building blocks
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# Simulation params, FEATURES, and the pipeline building blocks now live in
# retention_core (single source of truth, reproduces enhanced_analysis.py).
SEED = rc.SEED
FEATURES = rc.FEATURES


# ============================================================================
#  Evaluation
# ============================================================================
spatial_groups = rc.spatial_groups


def evaluate(X_sub, y_reg, y_cls, groups, n_splits=5):
    """Return (AUC_random, AUC_spatial, R2_random, R2_spatial) for a feature subset."""
    Xs = StandardScaler().fit_transform(X_sub)

    def auc_cv(splitter, split_args):
        aucs = []
        for tr, te in splitter.split(*split_args):
            if y_cls[tr].sum() == 0 or y_cls[te].sum() == 0:
                continue
            m = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED,
                                       class_weight="balanced").fit(Xs[tr], y_cls[tr])
            aucs.append(roc_auc_score(y_cls[te], m.predict_proba(Xs[te])[:, 1]))
        return float(np.mean(aucs)) if aucs else np.nan

    def r2_cv(splitter, split_args):
        r2s = []
        for tr, te in splitter.split(*split_args):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                          random_state=SEED).fit(Xs[tr], y_reg[tr])
            r2s.append(r2_score(y_reg[te], m.predict(Xs[te])))
        return float(np.mean(r2s)) if r2s else np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)  # regression random split
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    return (auc_cv(skf, (Xs, y_cls)), auc_cv(gkf, (Xs, y_cls, groups)),
            r2_cv(kf, (Xs, y_reg)), r2_cv(gkf, (Xs, y_reg, groups)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--velocity", default="Oceanic_data.nc")
    ap.add_argument("--species-csv", default="caretta_data.csv")
    ap.add_argument("--species", default="Caretta caretta")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args(argv)
    os.makedirs(args.fig_dir, exist_ok=True)

    print("Loading velocity...")
    vel = rc.load_velocity(args.velocity)
    # Cache the (slow) ensemble retention field, keyed by velocity file + grid shape.
    cache = os.path.join(args.fig_dir,
                         f".retention_cache_{os.path.basename(args.velocity)}_{vel['u'].shape}.npy")
    if os.path.exists(cache):
        print(f"  using cached retention: {cache}")
        retention = np.load(cache)
    else:
        print("  simulating retention (same seeds as pipeline)...")
        interp, _, _ = rc.make_interp(vel["u"], vel["v"], vel["lon"], vel["lat"])
        retention, _, _ = rc.simulate_retention(vel["lon"], vel["lat"], len(vel["time"]), interp)
        np.save(cache, retention)
    df_med = rc.load_sightings(args.species_csv, args.species, vel)
    print(f"  sightings in domain: {len(df_med)}")
    X, y_reg, lats, lons, _fields = rc.build_features(vel, retention, df_med)
    thr = np.percentile(y_reg[y_reg > 0], 50) if (y_reg > 0).any() else 0.01
    y_cls = (y_reg > thr).astype(int)
    groups = spatial_groups(lats, lons)
    idx = {f: i for i, f in enumerate(FEATURES)}
    print(f"  cells: {len(X)} | positives: {y_cls.sum()} | spatial blocks: {len(np.unique(groups))}")

    def sub(cols):
        return X[:, [idx[c] for c in cols]]

    rows = []  # (label, feature_set, results-tuple)

    # --- cumulative tiers ---
    B0 = ["latitude", "longitude"]
    B1 = B0 + ["mean_speed", "vorticity", "dist_to_coast"]
    B2 = B1 + ["retention"]
    for label, cols in [("B0 geography", B0), ("B1 +static ocean", B1), ("B2 +retention (full)", B2)]:
        rows.append((label, cols, evaluate(sub(cols), y_reg, y_cls, groups)))

    # --- LOFO from full ---
    for f in FEATURES:
        cols = [c for c in FEATURES if c != f]
        rows.append((f"LOFO -{f}", cols, evaluate(sub(cols), y_reg, y_cls, groups)))

    # --- geography-controlled (no lat/lon) ---
    geo_ctrl = ["retention", "mean_speed", "vorticity", "dist_to_coast"]
    rows.append(("no-geography", geo_ctrl, evaluate(sub(geo_ctrl), y_reg, y_cls, groups)))

    # --- spatial-null: shuffle retention within latitude bands ---
    rng = np.random.default_rng(SEED)
    Xn = X.copy()
    bands = np.floor((lats - lats.min()) / 1.0).astype(int)
    for b in np.unique(bands):
        m = bands == b
        perm = rng.permutation(np.where(m)[0])
        Xn[m, idx["retention"]] = X[perm, idx["retention"]]
    rows.append(("retention spatial-null", FEATURES,
                 evaluate(Xn[:, [idx[c] for c in FEATURES]], y_reg, y_cls, groups)))

    # --- print table ---
    print("\n" + "=" * 86)
    print(f"{'experiment':<26}{'AUC rand':>10}{'AUC spat':>10}{'R2 rand':>10}{'R2 spat':>10}")
    print("=" * 86)
    full = next(r for r in rows if r[0].startswith("B2"))[2]
    b1 = next(r for r in rows if r[0].startswith("B1"))[2]
    for label, cols, (ar, asp, rr, rsp) in rows:
        print(f"{label:<26}{ar:>10.4f}{asp:>10.4f}{rr:>10.4f}{rsp:>10.4f}")
    print("-" * 86)
    print(f"{'>> retention gain (B2-B1)':<26}{full[0]-b1[0]:>10.4f}{full[1]-b1[1]:>10.4f}"
          f"{full[2]-b1[2]:>10.4f}{full[3]-b1[3]:>10.4f}")

    # --- permutation importance on full RF (spatial-honest holdout) ---
    Xs = StandardScaler().fit_transform(sub(FEATURES))
    gkf = GroupKFold(n_splits=5)
    tr, te = next(gkf.split(Xs, y_cls, groups))
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED,
                                class_weight="balanced").fit(Xs[tr], y_cls[tr])
    perm = permutation_importance(rf, Xs[te], y_cls[te], n_repeats=20,
                                  random_state=SEED, scoring="roc_auc")
    print("\nPermutation importance (full RF, spatial holdout, ΔAUC):")
    order = np.argsort(perm.importances_mean)[::-1]
    for i in order:
        print(f"  {FEATURES[i]:<14} {perm.importances_mean[i]:+.4f} ± {perm.importances_std[i]:.4f}")

    # --- figure 20 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = [r[0] for r in rows]
    auc_r = [r[2][0] for r in rows]; auc_s = [r[2][1] for r in rows]
    yy = np.arange(len(labels))
    axes[0].barh(yy - 0.2, auc_r, height=0.4, label="random CV", color="steelblue")
    axes[0].barh(yy + 0.2, auc_s, height=0.4, label="spatial-block CV", color="indianred")
    axes[0].set_yticks(yy); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Classifier AUC"); axes[0].set_title("Feature ablation — AUC")
    axes[0].axvline(0.5, color="k", ls="--", lw=0.8); axes[0].legend(); axes[0].invert_yaxis()
    # sort bars AND labels together (ascending -> largest lands at top of barh);
    # using a feature-length index avoids the earlier label/bar misalignment bug.
    fi = np.arange(len(FEATURES))
    asc = np.argsort(perm.importances_mean)
    axes[1].barh(fi, perm.importances_mean[asc], color="teal",
                 xerr=perm.importances_std[asc])
    axes[1].set_yticks(fi); axes[1].set_yticklabels([FEATURES[i] for i in asc])
    axes[1].set_xlabel("Permutation ΔAUC"); axes[1].set_title("Permutation importance (full RF)")
    plt.tight_layout()
    out = os.path.join(args.fig_dir, "20_ablation.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
