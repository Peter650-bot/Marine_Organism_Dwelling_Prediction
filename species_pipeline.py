#!/usr/bin/env python3
"""
species_pipeline.py — generalised, multi-species retention SDM.

Replaces the Caretta-hardcoded enhanced_analysis.py with a parameterised engine:

    run_species(species, velocity_nc, species_csv, fig_dir) -> results dict
    run_pooled(per_species_results)                          -> pooled results

Per species it reproduces the paper's analysis (ensemble retention, Spearman +
permutation + bootstrap association, RF classifier, GBR regressor) but reports
metrics under BOTH random CV (optimistic, comparable to the paper) and
spatial-block CV (honest under spatial autocorrelation — see the ablation
finding that the paper's scores are inflated). The pooled model stacks all
species' per-cell matrices with species identity and evaluates transferability
via leave-one-species-out and leave-one-region-out CV.

All heavy lifting is delegated to retention_core.py, which reproduces
enhanced_analysis.py Sections 1-2/4a exactly under the default seeds — so the
Caretta/Mediterranean run is a faithful regression test of this refactor.

CLI
---
    # one species
    python species_pipeline.py --velocity Datasets/Oceanic_data.nc \
        --species-csv Datasets/caretta_data.csv --species "Caretta caretta"

    # several species (per-species runs + pooled comparison); repeat --run
    python species_pipeline.py \
        --run "Caretta caretta:Datasets/Oceanic_data.nc:Datasets/caretta_data.csv" \
        --run "Cetorhinus maximus:Datasets/ocean_basking_NEAtlantic.nc:Datasets/<multispp>.csv"
"""

import argparse
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
from sklearn.metrics import roc_auc_score, r2_score, roc_curve
from sklearn.inspection import permutation_importance

import retention_core as rc
import species_figures as sf

warnings.filterwarnings("ignore")


# ============================================================================
#  Stats + model helpers
# ============================================================================
def association_stats(retention_norm, sights, seed=rc.SEED, n_perm=1000, n_boot=1000):
    """Spearman rho on the union mask + permutation p + bootstrap 95% CI.
    Matches enhanced_analysis.py Section 3 (and reproduces rho=0.362 for Caretta)."""
    mask = (retention_norm > 0) | (sights > 0)
    r = retention_norm[mask]
    s = sights[mask]
    if len(r) <= 2:
        return {"rho": np.nan, "p": np.nan, "perm_p": np.nan, "ci": [np.nan, np.nan], "n": int(len(r))}
    rng = np.random.default_rng(seed)
    rho, p = spearmanr(r, s)
    perm = np.array([spearmanr(r, rng.permutation(s))[0] for _ in range(n_perm)])
    perm_p = float(np.mean(np.abs(perm) >= abs(rho)))
    boot = np.array([spearmanr(*(lambda i: (r[i], s[i]))(rng.integers(0, len(r), len(r))))[0]
                     for _ in range(n_boot)])
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    return {"rho": float(rho), "p": float(p), "perm_p": perm_p, "ci": ci, "n": int(len(r))}


def cv_scores(X, y_reg, y_cls, groups, seed=rc.SEED, n_splits=5):
    """Return AUC (RF) and R^2 (GBR) under random CV and spatial-block CV."""
    Xs = StandardScaler().fit_transform(X)

    def auc_cv(splitter, args):
        out = []
        for tr, te in splitter.split(*args):
            if y_cls[tr].sum() == 0 or y_cls[te].sum() == 0:
                continue
            m = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed,
                                       class_weight="balanced").fit(Xs[tr], y_cls[tr])
            out.append(roc_auc_score(y_cls[te], m.predict_proba(Xs[te])[:, 1]))
        return float(np.mean(out)) if out else np.nan

    def r2_cv(splitter, args):
        out = []
        for tr, te in splitter.split(*args):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                          random_state=seed).fit(Xs[tr], y_reg[tr])
            out.append(r2_score(y_reg[te], m.predict(Xs[te])))
        return float(np.mean(out)) if out else np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ng = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_splits, ng)) if ng >= 2 else None
    res = {"auc_random": auc_cv(skf, (Xs, y_cls)), "r2_random": r2_cv(kf, (Xs, y_reg))}
    res["auc_spatial"] = auc_cv(gkf, (Xs, y_cls, groups)) if gkf else np.nan
    res["r2_spatial"] = r2_cv(gkf, (Xs, y_reg, groups)) if gkf else np.nan
    return res


# ============================================================================
#  Per-species run
# ============================================================================
def run_species(species, velocity_nc, species_csv, fig_dir="figures",
                window_years=2, min_year=rc.ALTIMETRY_START_YEAR, equator_band=0.0,
                max_steps=rc.MAX_STEPS, verbose=True):
    """Full per-species analysis. Returns a results dict and, for pooling, the
    per-cell feature matrix + targets + spatial groups."""
    tag = species.replace(" ", "_").replace("/", "_")
    out_dir = os.path.join(fig_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    def log(*a):
        if verbose:
            print(*a)

    log(f"\n{'='*70}\nSPECIES: {species}\n{'='*70}")
    vel = rc.load_velocity(velocity_nc)
    log(f"  velocity: {velocity_nc}  lon[{vel['lon'].min():.2f},{vel['lon'].max():.2f}] "
        f"({vel['lon_conv']}) {vel['t_start'].date()}..{vel['t_end'].date()} n_t={len(vel['time'])}")

    # cache the slow retention sim, keyed by cube + sim params
    cache = os.path.join(out_dir, f".retention_{os.path.basename(velocity_nc)}_{vel['u'].shape}_{max_steps}.npy")
    if os.path.exists(cache):
        retention = np.load(cache); log("  retention: cached")
    else:
        log("  simulating retention...")
        interp, _, _ = rc.make_interp(vel["u"], vel["v"], vel["lon"], vel["lat"])
        retention, _, _ = rc.simulate_retention(vel["lon"], vel["lat"], len(vel["time"]),
                                                 interp, max_steps=max_steps)
        np.save(cache, retention)

    df_med = rc.load_sightings(species_csv, species, vel, window_years=window_years,
                               min_year=min_year, equator_band=equator_band)
    log(f"  sightings in domain: {len(df_med)}")
    if len(df_med) < 5:
        log("  INSUFFICIENT data (<5) — skipping modeling.")
        return {"species": species, "n_sightings": len(df_med), "sufficient": False}

    X, y_reg, lats, lons, fields = rc.build_features(vel, retention, df_med)
    thr = np.percentile(y_reg[y_reg > 0], 50) if (y_reg > 0).any() else 0.01
    y_cls = (y_reg > thr).astype(int)
    groups = rc.spatial_groups(lats, lons)

    stats = association_stats(fields["retention"], fields["sights"])
    log(f"  Spearman rho={stats['rho']:.4f} (p={stats['p']:.2e}, perm_p={stats['perm_p']:.4f}, "
        f"95%CI={[round(c,3) for c in stats['ci']]})")
    cv = cv_scores(X, y_reg, y_cls, groups)
    log(f"  RF AUC  random={cv['auc_random']:.4f}  spatial={cv['auc_spatial']:.4f}")
    log(f"  GBR R2  random={cv['r2_random']:.4f}  spatial={cv['r2_spatial']:.4f}")

    # retention's honest marginal importance (permutation, spatial holdout)
    ret_imp = np.nan
    ng = len(np.unique(groups))
    if ng >= 2 and y_cls.sum() > 0:
        Xs = StandardScaler().fit_transform(X)
        tr, te = next(GroupKFold(n_splits=min(5, ng)).split(Xs, y_cls, groups))
        if y_cls[tr].sum() and y_cls[te].sum():
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=rc.SEED,
                                        class_weight="balanced").fit(Xs[tr], y_cls[tr])
            pi = permutation_importance(rf, Xs[te], y_cls[te], n_repeats=20,
                                        random_state=rc.SEED, scoring="roc_auc")
            ret_imp = float(pi.importances_mean[rc.FEATURES.index("retention")])
    log(f"  retention permutation ΔAUC (spatial holdout): {ret_imp:+.4f}")

    _save_species_figure(out_dir, species, vel, fields, X, y_reg, y_cls, groups, stats)
    sf.make_species_figures(out_dir, species, vel, fields, X, y_reg, y_cls, groups, stats)

    results = {"species": species, "n_sightings": int(len(df_med)), "sufficient": True,
               "stats": stats, "cv": cv, "retention_perm_dAUC": ret_imp,
               "fig_dir": out_dir}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # attach arrays for pooling (not serialised)
    results["_pool"] = {"X": X, "y_reg": y_reg, "y_cls": y_cls, "groups": groups}
    return results


def _save_species_figure(out_dir, species, vel, fields, X, y_reg, y_cls, groups, stats):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    lon, lat = vel["lon"], vel["lat"]
    im0 = ax[0, 0].pcolormesh(lon, lat, fields["retention"], shading="auto", cmap="YlOrRd")
    ax[0, 0].set_title("Particle retention (ensemble)"); plt.colorbar(im0, ax=ax[0, 0])
    im1 = ax[0, 1].pcolormesh(lon, lat, fields["sights"], shading="auto", cmap="YlGnBu")
    ax[0, 1].set_title(f"{species} weighted sightings"); plt.colorbar(im1, ax=ax[0, 1])
    mask = (fields["retention"] > 0) | (fields["sights"] > 0)
    ax[1, 0].scatter(fields["retention"][mask], fields["sights"][mask], s=3, alpha=0.4)
    ax[1, 0].set_xlabel("retention"); ax[1, 0].set_ylabel("sightings")
    ax[1, 0].set_title(f"rho={stats['rho']:.3f} (perm p={stats['perm_p']:.3f})")
    # ROC under a spatial holdout
    ng = len(np.unique(groups))
    if ng >= 2 and y_cls.sum():
        Xs = StandardScaler().fit_transform(X)
        tr, te = next(GroupKFold(n_splits=min(5, ng)).split(Xs, y_cls, groups))
        if y_cls[tr].sum() and y_cls[te].sum():
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=rc.SEED,
                                        class_weight="balanced").fit(Xs[tr], y_cls[tr])
            fpr, tpr, _ = roc_curve(y_cls[te], rf.predict_proba(Xs[te])[:, 1])
            ax[1, 1].plot(fpr, tpr, lw=2,
                          label=f"AUC={roc_auc_score(y_cls[te], rf.predict_proba(Xs[te])[:,1]):.3f}")
    ax[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[1, 1].set_xlabel("FPR"); ax[1, 1].set_ylabel("TPR")
    ax[1, 1].set_title("RF ROC (spatial holdout)"); ax[1, 1].legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "summary.png")
    fig.savefig(out, bbox_inches="tight", dpi=150); plt.close(fig)
    print(f"  [saved] {out}")


# ============================================================================
#  Pooled model
# ============================================================================
def run_pooled(per_species, fig_dir="figures", seed=rc.SEED):
    """Stack all species' per-cell matrices with species identity; evaluate
    retention's value with leave-one-species-out and leave-one-region-out CV."""
    usable = [r for r in per_species if r.get("sufficient") and "_pool" in r]
    if len(usable) < 2:
        print("\nPooled model needs >=2 species with sufficient data — skipping.")
        return None
    print(f"\n{'='*70}\nPOOLED MODEL ({len(usable)} species)\n{'='*70}")

    names = [r["species"] for r in usable]
    Xs, ys, gs_region, gs_species, onehot = [], [], [], [], []
    region_offset = 0
    for si, r in enumerate(usable):
        p = r["_pool"]
        n = len(p["X"])
        Xs.append(p["X"]); ys.append(p["y_cls"])
        gs_region.append(p["groups"] + region_offset)   # keep regions disjoint across species
        gs_species.append(np.full(n, si))
        oh = np.zeros((n, len(usable))); oh[:, si] = 1
        onehot.append(oh)
        region_offset += int(p["groups"].max()) + 1
    X = np.vstack(Xs)
    Xfull = np.hstack([X, np.vstack(onehot)])     # features + species one-hot
    y = np.concatenate(ys)
    g_region = np.concatenate(gs_region)
    g_species = np.concatenate(gs_species)
    Xsc = StandardScaler().fit_transform(Xfull)

    def loso_auc(groups, label, n_splits):
        out = []
        for tr, te in GroupKFold(n_splits=n_splits).split(Xsc, y, groups):
            # skip degenerate folds whose train or test set is single-class
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed,
                                       class_weight="balanced").fit(Xsc[tr], y[tr])
            out.append(roc_auc_score(y[te], m.predict_proba(Xsc[te])[:, 1]))
        mean = float(np.mean(out)) if out else np.nan
        print(f"  {label}: AUC = {mean:.4f}  (n folds={len(out)})")
        return mean

    # region scheme: group the fine spatial blocks into <=5 folds (one block per
    # fold would give single-class test sets). species scheme: one species per fold.
    n_region_folds = min(5, len(np.unique(g_region)))
    n_species_folds = len(np.unique(g_species))
    auc_region = loso_auc(g_region, "leave-one-region-out", n_region_folds)
    auc_species = loso_auc(g_species, "leave-one-SPECIES-out (transfer)", n_species_folds)

    # retention permutation importance in the pooled model (region holdout)
    tr, te = next(GroupKFold(n_splits=n_region_folds).split(Xsc, y, g_region))
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed,
                                class_weight="balanced").fit(Xsc[tr], y[tr])
    pi = permutation_importance(rf, Xsc[te], y[te], n_repeats=20, random_state=seed, scoring="roc_auc")
    ret_imp = float(pi.importances_mean[rc.FEATURES.index("retention")])
    print(f"  pooled retention permutation ΔAUC: {ret_imp:+.4f}")

    res = {"species": names, "n_species": len(names),
           "auc_leave_region_out": auc_region,
           "auc_leave_species_out": auc_species, "pooled_retention_perm_dAUC": ret_imp}
    pooled_dir = os.path.join(fig_dir, "pooled")
    os.makedirs(pooled_dir, exist_ok=True)
    with open(os.path.join(pooled_dir, "pooled_results.json"), "w") as f:
        json.dump(res, f, indent=2)

    # pooled summary figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(["leave-region-out", "leave-species-out\n(transfer)"],
              [auc_region, auc_species], color=["steelblue", "indianred"])
    ax[0].axhline(0.5, color="k", ls="--", lw=0.8)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("AUC")
    ax[0].set_title(f"Pooled model ({len(names)} species)")
    for i, v in enumerate([auc_region, auc_species]):
        ax[0].text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax[1].barh(names, [r["stats"]["rho"] for r in usable], color="teal")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("per-species Spearman rho")
    ax[1].set_title(f"retention–sightings rho per species\n(pooled retention ΔAUC={ret_imp:+.3f})")
    plt.tight_layout()
    fig.savefig(os.path.join(pooled_dir, "pooled_summary.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {pooled_dir}/pooled_summary.png + pooled_results.json")
    sf.make_pooled_figures(pooled_dir, usable, Xsc, y, g_region, g_species,
                           n_region_folds, n_species_folds, names, res)
    return res


# ============================================================================
#  CLI
# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--velocity"); ap.add_argument("--species-csv"); ap.add_argument("--species")
    ap.add_argument("--run", action="append", default=[],
                    help="species:velocity_nc:species_csv  (repeatable)")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--window-years", type=int, default=2)
    ap.add_argument("--min-year", type=int, default=rc.ALTIMETRY_START_YEAR)
    ap.add_argument("--equator-band", type=float, default=0.0)
    # In-context TabLLM extension (off by default; hits the Anthropic API).
    ap.add_argument("--tabllm", action="store_true",
                    help="also run the in-context TabLLM learning curve per species")
    ap.add_argument("--tabllm-model", default="claude-haiku-4-5",
                    help="LLM id; for vLLM the served model name "
                         "(e.g. unsloth/Llama-3.2-3B-Instruct)")
    ap.add_argument("--tabllm-base-url", default=os.environ.get("VLLM_BASE_URL"),
                    help="OpenAI-compatible vLLM endpoint (e.g. https://<tunnel>/v1); "
                         "default $VLLM_BASE_URL. Set -> FREE inference via vLLM.")
    ap.add_argument("--tabllm-api-key", default=os.environ.get("VLLM_API_KEY"),
                    help="API key for the vLLM/OpenAI endpoint (default $VLLM_API_KEY)")
    ap.add_argument("--tabllm-n-test", type=int, default=800)
    ap.add_argument("--tabllm-n-vote", type=int, default=1)
    ap.add_argument("--tabllm-max-retries", type=int, default=4,
                    help="per-request retries before a call is treated as failed "
                         "(lower -> a dead vLLM server fails fast)")
    ap.add_argument("--tabllm-error-abort", type=int, default=25,
                    help="abort a batched TabLLM run after this many failed "
                         "requests, keeping the cache clean (0 disables)")
    args = ap.parse_args(argv)

    runs = [tuple(r.split(":", 2)) for r in args.run]
    if args.species and args.velocity and args.species_csv:
        runs.append((args.species, args.velocity, args.species_csv))
    if not runs:
        ap.error("provide --species/--velocity/--species-csv or one or more --run")

    results = [run_species(sp, vel, csv, fig_dir=args.fig_dir,
                           window_years=args.window_years, min_year=args.min_year,
                           equator_band=args.equator_band)
               for sp, vel, csv in runs]
    if len([r for r in results if r.get("sufficient")]) >= 2:
        run_pooled(results, fig_dir=args.fig_dir)

    if args.tabllm:
        import tabllm_pipeline as tl
        for sp, vel, csv in runs:
            cells = tl.load_cells(sp, vel, csv, fig_dir=args.fig_dir,
                                  window_years=args.window_years,
                                  min_year=args.min_year)
            tl.run_curve(cells, model=args.tabllm_model,
                         n_test=args.tabllm_n_test, n_vote=args.tabllm_n_vote,
                         base_url=args.tabllm_base_url, api_key=args.tabllm_api_key,
                         max_retries=args.tabllm_max_retries,
                         error_abort=args.tabllm_error_abort)

    print(f"\n{'='*70}\nDONE — {len(runs)} species\n{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
