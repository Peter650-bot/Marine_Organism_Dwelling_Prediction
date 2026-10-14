#!/usr/bin/env python3
"""
loro15_caretta.py -- TabICLv2 vs nested-tuned RF on Caretta caretta under
contiguous leave-one-region-out CV with 15 k-means regions instead of 5.

Why: with 5 regions only 4 are scoreable, and an exact Wilcoxon signed-rank
test on 4 pairs cannot go below p = 0.125 even when TabICL wins every region.

Everything except the region count matches `run_pipeline.py` + `tune_rf.py`:
same KDE core-area label, features, TabICLv2 config (n_estimators=2), test
subsampling (class-balanced, up to 300 cells), RF grid, and nested inner
leave-one-region-out selection done separately for AUC and F1.

Writes `figures/Caretta_caretta/results_k15.json` (per-fold, saved after each
fold so an interrupted run resumes). Never touches `results.json`.
"""
import json, os, sys, time

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
import label_cv as lcv
import run_pipeline as rp
import tune_rf as tr_

SPECIES = "Caretta caretta"
N_REGIONS = 15
SEED = 42
OUT = os.path.join("figures", "Caretta_caretta", f"results_k{N_REGIONS}.json")


def main():
    import retention_core as rc
    import tabllm_prompt as tp
    from tabicl import TabICLClassifier

    vel_nc, csv, _ = rp.RUNS[SPECIES]
    cells = tp.load_cells(SPECIES, vel_nc, csv)
    X, lats, lons, vel = cells["X"], cells["lats"], cells["lons"], cells["vel"]
    df = rc.load_sightings(csv, SPECIES, vel)
    y = lcv.core_area_label(lcv.kde_density(
        df["longitude"].values, df["latitude"].values,
        df["weighted_records"].values, lons, lats))
    splits, region = lcv.loro_kmeans_folds(lats, lons, n_splits=N_REGIONS, seed=SEED)

    res = {"species": SPECIES, "n_regions": N_REGIONS, "folds": [],
           "folds_unscoreable": []}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    done = {f["fold"] for f in res["folds"]} | {f["fold"] for f in res["folds_unscoreable"]}

    clf = TabICLClassifier(n_estimators=2, random_state=SEED, model_path=rp.CKPT)
    rng = np.random.default_rng(SEED)                 # outer test draws, fold order
    t_start = time.time()
    for i, (tr, te) in enumerate(splits):
        kept = lcv.balanced_subsample(te, y, rp.N_TEST, rng)   # always draw: keeps RNG order
        if i in done:
            continue
        if kept is None or len(np.unique(y[tr])) < 2:
            res["folds_unscoreable"].append({"fold": i, "n_region_cells": int(len(te)),
                                             "n_pos_in_region": int(y[te].sum())})
            print(f"fold {i}: UNSCOREABLE ({int(y[te].sum())} positives)", flush=True)
            json.dump(res, open(OUT, "w"), indent=2)
            continue
        t0 = time.time()
        clf.fit(X[tr], y[tr])
        auc_t, f1_t = rp._metrics(y[kept], clf.predict_proba(X[kept])[:, 1])

        # inner rng seeded per fold so a resumed run reproduces the same draws
        inner_rng = np.random.default_rng(tr_.INNER_SEED + i)
        cfg_a, cfg_f, n_inner = tr_.tune_on_train(X, y, tr, region, inner_rng)
        auc_r, _ = tr_._score(tr_.make_rf(cfg_a).fit(X[tr], y[tr]), X, y, kept)
        _, f1_r = tr_._score(tr_.make_rf(cfg_f).fit(X[tr], y[tr]), X, y, kept)

        res["folds"].append({"fold": i, "n_test": int(len(kept)),
                             "n_pos_test": int(y[kept].sum()),
                             "auc_tabicl": auc_t, "f1_tabicl": f1_t,
                             "auc_rf_tuned": auc_r, "f1_rf_tuned": f1_r,
                             "rf_inner_folds": n_inner})
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"fold {i} [{time.time()-t0:.0f}s, total {time.time()-t_start:.0f}s] "
              f"n_test={len(kept)}  AUC tabicl={auc_t:.4f} rf={auc_r:.4f} | "
              f"F1 tabicl={f1_t:.4f} rf={f1_r:.4f}", flush=True)

    F = sorted(res["folds"], key=lambda f: f["fold"])
    print(f"\n{SPECIES}, {N_REGIONS} regions: {len(F)} scoreable, "
          f"{len(res['folds_unscoreable'])} unscoreable")
    for m in ("auc", "f1"):
        d = np.array([f[f"{m}_tabicl"] - f[f"{m}_rf_tuned"] for f in F])
        p = stats.wilcoxon(d).pvalue
        print(f"{m.upper():3s}  mean diff (TabICL - tuned RF) = {d.mean():+.4f}   "
              f"Wilcoxon p = {p:.4f}")


if __name__ == "__main__":
    main()
