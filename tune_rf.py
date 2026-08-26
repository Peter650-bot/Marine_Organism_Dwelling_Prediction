#!/usr/bin/env python3
"""
tune_rf.py -- nested cross-validation for the Random Forest baseline.

The headline comparison is only worth something if the baseline got a fair
shot. `run_pipeline.py` uses one hard-coded RF configuration; this script
selects RF hyperparameters properly and re-scores it, so the TabICLv2-vs-RF
claim can be restated against a *tuned* baseline instead of a fixed one.

Nesting (this is the part that must not be got wrong)
----------------------------------------------------
The outer loop is the same contiguous leave-one-region-out split the pipeline
uses. Hyperparameters are chosen INSIDE each outer training set, by
leave-one-region-out over the remaining regions -- the inner scheme mirrors the
outer one, so selection never sees the held-out region. Choosing on the test
region would leak and would be a worse criticism than not tuning at all.

Two selections, not one
-----------------------
Selecting on AUC and then reporting F1 would hand RF a config it was never
tuned for. Each configuration is scored on both metrics in the same inner pass,
and the best-by-AUC and best-by-F1 configurations are refit separately. RF is
therefore given its best available shot on each metric independently.

Comparability
-------------
The outer test cells are the pipeline's, not new ones: the RNG draw order of
`run_pipeline.py` is replayed and ASSERTED against the recorded `n_test` /
`n_pos_test` before anything is written. Inner subsampling uses a separate RNG
so it cannot perturb that replay.

Writes `auc_rf_tuned` / `f1_rf_tuned` and the chosen parameters into
`figures/<species>/results.json`, leaving `auc_rf_full` / `f1_rf_full`
untouched for the before/after comparison.

CLI:
    python tune_rf.py                      # all species
    python tune_rf.py --species "Caretta caretta"
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

FIG_ROOT = "figures"
SEED = 42
N_TEST = 300
THRESHOLD = 0.5
INNER_SEED = SEED + 9999          # never touches the outer replay stream

GRID = {
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", 0.5],
    "class_weight": ["balanced", None],
}
BASELINE = {"max_depth": 10, "min_samples_leaf": 1,
            "max_features": "sqrt", "class_weight": "balanced"}


def configs():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def make_rf(cfg, seed=SEED):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, random_state=seed,
                                  n_jobs=-1, **cfg)


def _score(model, X, y, idx):
    from sklearn.metrics import roc_auc_score, f1_score
    p = model.predict_proba(X[idx])[:, 1]
    return (float(roc_auc_score(y[idx], p)),
            float(f1_score(y[idx], (p >= THRESHOLD).astype(int), zero_division=0)))


def tune_on_train(X, y, tr, region, inner_rng, verbose=False):
    """Leave-one-region-out selection INSIDE the outer training set."""
    import label_cv as lcv
    inner_regions = [r for r in np.unique(region[tr])]
    folds = []
    for r in inner_regions:
        itr = tr[region[tr] != r]
        ite = tr[region[tr] == r]
        if len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2:
            continue
        kept = lcv.balanced_subsample(ite, y, N_TEST, inner_rng)
        if kept is None:
            continue
        folds.append((itr, kept))
    if not folds:
        return BASELINE, BASELINE, 0

    cfgs = configs()
    auc_tab = np.full((len(cfgs), len(folds)), np.nan)
    f1_tab = np.full((len(cfgs), len(folds)), np.nan)
    for ci, cfg in enumerate(cfgs):
        for fi, (itr, kept) in enumerate(folds):
            m = make_rf(cfg).fit(X[itr], y[itr])
            auc_tab[ci, fi], f1_tab[ci, fi] = _score(m, X, y, kept)
    a, f = np.nanmean(auc_tab, axis=1), np.nanmean(f1_tab, axis=1)
    best_auc, best_f1 = cfgs[int(np.argmax(a))], cfgs[int(np.argmax(f))]
    if verbose:
        print(f"      inner folds={len(folds)} | best AUC {a.max():.4f} {best_auc}")
        print(f"      {'':17s}| best F1  {f.max():.4f} {best_f1}")
    return best_auc, best_f1, len(folds)


def run_species(species, vel_nc, csv, verbose=True):
    import retention_core as rc
    import label_cv as lcv
    import tabllm_prompt as tp

    path = os.path.join(FIG_ROOT, species.replace(" ", "_"), "results.json")
    if not os.path.exists(path):
        print(f"skip {species}: no {path}")
        return
    with open(path) as fh:
        res = json.load(fh)

    cells = tp.load_cells(species, vel_nc, csv)
    X, lats, lons, vel = cells["X"], cells["lats"], cells["lons"], cells["vel"]
    df = rc.load_sightings(csv, species, vel)
    y = lcv.core_area_label(lcv.kde_density(
        df["longitude"].values, df["latitude"].values,
        df["weighted_records"].values, lons, lats))
    splits, region = lcv.loro_kmeans_folds(lats, lons)

    rng = np.random.default_rng(SEED)              # replay run_pipeline's order
    inner_rng = np.random.default_rng(INNER_SEED)  # separate stream
    recorded = {f["fold"]: f for f in res["folds"]}
    done = 0

    for i, (tr, te) in enumerate(splits):
        kept = lcv.balanced_subsample(te, y, N_TEST, rng)
        if kept is None or len(np.unique(y[tr])) < 2:
            continue
        rec = recorded.get(i)
        if rec is None:
            raise SystemExit(f"{species} fold {i}: scoreable in replay but absent "
                             f"from results.json")
        if len(kept) != rec["n_test"] or int(y[kept].sum()) != rec["n_pos_test"]:
            raise SystemExit(
                f"{species} fold {i}: replay mismatch -- n_test {len(kept)} vs "
                f"{rec['n_test']}, n_pos {int(y[kept].sum())} vs {rec['n_pos_test']}")

        t0 = time.time()
        cfg_a, cfg_f, n_inner = tune_on_train(X, y, tr, region, inner_rng, verbose)
        auc_t, _ = _score(make_rf(cfg_a).fit(X[tr], y[tr]), X, y, kept)
        _, f1_t = _score(make_rf(cfg_f).fit(X[tr], y[tr]), X, y, kept)

        rec["auc_rf_tuned"] = auc_t
        rec["f1_rf_tuned"] = f1_t
        rec["rf_tuned_params_auc"] = {k: str(v) for k, v in cfg_a.items()}
        rec["rf_tuned_params_f1"] = {k: str(v) for k, v in cfg_f.items()}
        rec["rf_inner_folds"] = n_inner
        done += 1
        if verbose:
            print(f"    fold {i} [{time.time()-t0:.0f}s]  "
                  f"AUC {rec['auc_rf_full']:.4f} -> {auc_t:.4f}   "
                  f"F1 {rec['f1_rf_full']:.4f} -> {f1_t:.4f}", flush=True)

    res["rf_tuning"] = {
        "scheme": "nested leave-one-region-out (inner mirrors outer)",
        "grid": {k: [str(x) for x in v] for k, v in GRID.items()},
        "n_estimators": 200,
        "selected_separately_for": ["auc", "f1"],
        "baseline_config": {k: str(v) for k, v in BASELINE.items()},
        "folds_tuned": done,
    }
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    a0 = np.array([f["auc_rf_full"] for f in res["folds"]])
    a1 = np.array([f["auc_rf_tuned"] for f in res["folds"]])
    f0 = np.array([f["f1_rf_full"] for f in res["folds"]])
    f1_ = np.array([f["f1_rf_tuned"] for f in res["folds"]])
    print(f"{species}: {done} folds tuned | "
          f"AUC {a0.mean():.4f} -> {a1.mean():.4f} | "
          f"F1 {f0.mean():.4f} -> {f1_.mean():.4f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", action="append", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    import run_pipeline as rp
    print(f"grid: {len(configs())} configs, n_estimators=200\n")
    for sp in (a.species or list(rp.RUNS)):
        vel_nc, csv, _ = rp.RUNS[sp]
        print(f"== {sp}")
        run_species(sp, vel_nc, csv, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
