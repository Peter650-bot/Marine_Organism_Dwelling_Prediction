#!/usr/bin/env python3
"""
tune_rf_kshot.py -- completes the tuned-RF picture for the paper.

Two computations that `tune_rf.py` does not cover:

1. **RF at k=32, tuned.** The k-shot row must be tuned the way a practitioner
   actually could with only 32 labelled cells: selection by stratified 4-fold CV
   *within those 32 exemplars*. Region-based inner CV is impossible here -- 32
   class-balanced exemplars do not populate enough regions. As with the
   full-context tuner, AUC and F1 configurations are selected separately.
   Exemplars are `run_pipeline._kshot_draw` under the same per-draw seeds, so the
   tuned row sees exactly the rows every other k=32 model saw.

2. **Permutation importance under the tuned RF.** The published table was
   computed with the fixed configuration; once RF-full is tuned, its feature
   attribution has to be recomputed from the tuned model or the two disagree.

Both merge into `figures/<species>/results.json`. The outer test cells are the
pipeline's, replayed and asserted exactly as in `tune_rf.py`.

CLI:
    python tune_rf_kshot.py
"""

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
K = 32
N_DRAWS = 5
N_INNER = 4

GRID = {
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.5],
    "class_weight": ["balanced", None],
}


def configs():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def parse_cfg(d):
    """Rebuild a config dict from the strings tune_rf.py stored in results.json."""
    return {
        "max_depth": None if d["max_depth"] == "None" else int(d["max_depth"]),
        "min_samples_leaf": int(d["min_samples_leaf"]),
        "max_features": (d["max_features"] if d["max_features"] == "sqrt"
                         else float(d["max_features"])),
        "class_weight": None if d["class_weight"] == "None" else d["class_weight"],
    }


def make_rf(cfg, seed=SEED, n_jobs=-1):
    """n_jobs is a pure scheduling knob -- with random_state fixed the fitted
    forest is identical either way. The k=32 fits use n_jobs=1 because thread
    spawn dominates the runtime on 32-row data."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, random_state=seed,
                                  n_jobs=n_jobs, **cfg)


def _sc(m, X, y, idx):
    from sklearn.metrics import roc_auc_score, f1_score
    p = m.predict_proba(X[idx])[:, 1]
    return (float(roc_auc_score(y[idx], p)),
            float(f1_score(y[idx], (p >= THRESHOLD).astype(int), zero_division=0)))


def tune_kshot(Xs, ys):
    """Select on the 32 exemplars alone, by stratified inner CV."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score
    cfgs = configs()
    skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xs, ys))
    A = np.full((len(cfgs), len(folds)), np.nan)
    F = np.full((len(cfgs), len(folds)), np.nan)
    for ci, cfg in enumerate(cfgs):
        for fi, (a, b) in enumerate(folds):
            if len(np.unique(ys[a])) < 2 or len(np.unique(ys[b])) < 2:
                continue
            m = make_rf(cfg, n_jobs=1).fit(Xs[a], ys[a])
            p = m.predict_proba(Xs[b])[:, 1]
            A[ci, fi] = roc_auc_score(ys[b], p)
            F[ci, fi] = f1_score(ys[b], (p >= THRESHOLD).astype(int), zero_division=0)
    ma, mf = np.nanmean(A, axis=1), np.nanmean(F, axis=1)
    if np.all(np.isnan(ma)):
        return cfgs[0], cfgs[0]
    return cfgs[int(np.nanargmax(ma))], cfgs[int(np.nanargmax(mf))]


def main():
    sys.path.insert(0, os.getcwd())
    import retention_core as rc
    import label_cv as lcv
    import tabllm_prompt as tp
    import run_pipeline as rp
    from sklearn.inspection import permutation_importance

    print(f"k={K} grid: {len(configs())} configs\n")
    for species, (vel_nc, csv, _b) in rp.RUNS.items():
        path = os.path.join(FIG_ROOT, species.replace(" ", "_"), "results.json")
        if not os.path.exists(path):
            print(f"skip {species}"); continue
        res = json.load(open(path))

        cells = tp.load_cells(species, vel_nc, csv)
        X, lats, lons, vel = cells["X"], cells["lats"], cells["lons"], cells["vel"]
        df = rc.load_sightings(csv, species, vel)
        y = lcv.core_area_label(lcv.kde_density(
            df["longitude"].values, df["latitude"].values,
            df["weighted_records"].values, lons, lats))
        splits, _ = lcv.loro_kmeans_folds(lats, lons)

        rng = np.random.default_rng(SEED)
        recorded = {f["fold"]: f for f in res["folds"]}
        ks_records = []
        print(f"== {species}")

        for i, (tr, te) in enumerate(splits):
            kept = lcv.balanced_subsample(te, y, N_TEST, rng)
            if kept is None or len(np.unique(y[tr])) < 2:
                continue
            rec = recorded[i]
            assert len(kept) == rec["n_test"] and int(y[kept].sum()) == rec["n_pos_test"], \
                f"{species} fold {i}: replay mismatch"
            if "rf_tuned_params_auc" not in rec:
                raise SystemExit(
                    f"{species} fold {i}: no tuned RF configuration on record. "
                    f"Run `python tune_rf.py` first -- this script reuses the "
                    f"configuration it selected rather than re-selecting one.")

            # ---- 1. permutation importance under the tuned full-context RF ----
            t0 = time.time()
            m = make_rf(parse_cfg(rec["rf_tuned_params_auc"])).fit(X[tr], y[tr])
            pi = permutation_importance(m, X[kept], y[kept], scoring="roc_auc",
                                        n_repeats=10, random_state=SEED, n_jobs=-1)
            rec["perm_importance_tuned"] = {f: float(v) for f, v in
                                            zip(rp.FEATURES, pi.importances_mean)}

            # ---- 2. tuned RF at k=32 ----
            for draw in range(N_DRAWS):
                drng = np.random.default_rng(SEED + 1000 * K + draw)
                sh = rp._kshot_draw(tr, y, K, drng)
                if sh is None:
                    continue
                Xs, ys = X[sh], y[sh]
                ca, cf = tune_kshot(Xs, ys)
                auc, _ = _sc(make_rf(ca, n_jobs=1).fit(Xs, ys), X, y, kept)
                _, f1 = _sc(make_rf(cf, n_jobs=1).fit(Xs, ys), X, y, kept)
                ks_records.append({"fold": i, "draw": draw, "auc": auc, "f1": f1})
            print(f"    fold {i} [{time.time()-t0:.0f}s] done", flush=True)

        res.setdefault("kshot", {}).setdefault(str(K), {})["rf_tuned"] = ks_records
        json.dump(res, open(path, "w"), indent=2)
        if ks_records:
            a = np.array([r["auc"] for r in ks_records])
            f = np.array([r["f1"] for r in ks_records])
            base = np.array([r["auc"] for r in res["kshot"][str(K)]["rf"]])
            print(f"{species}: k={K} RF AUC {base.mean():.3f} -> {a.mean():.3f} "
                  f"(sd {a.std(ddof=1):.3f}) | F1 {f.mean():.3f}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
