#!/usr/bin/env python3
"""
run_pipeline.py -- the main run: retention features, KDE core-area labels
(`label_cv.core_area_label`) and contiguous leave-one-region-out CV
(`label_cv.loro_kmeans_folds`), across every species.

Writes `figures/<species>/results.json`, which every table and figure reads.
Run this first.

Models
------
Full context : TabICLv2 (n_estimators=2, no fitting), RF-full (200 trees,
               depth 10, class-balanced).
k-shot       : TabICLv2, RandomForest, GradientBoosting, kNN, LogisticRegression
               at k in {8, 16, 32}, drawn class-balanced from the training fold.
               Repeated over `N_DRAWS` exemplar draws, because the exemplar draw
               alone moves AUC by as much as 0.105.

TabLLM is not run here -- it needs a GPU-backed vLLM server. Run `run_tabllm.py`
afterwards to add its row; it replays this script's exact folds and exemplars.

Fold accounting
---------------
Contiguous regions can legitimately contain no core-area cells at all -- that is
what genuine extrapolation against a concentrated core looks like. Such folds
cannot yield an AUC and are recorded in `folds_unscoreable`, never silently
dropped. Test subsamples stay class-balanced but shrink to
min(150, n_pos, n_neg) per class, and the realised size is recorded per fold.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

FIG_ROOT = "figures"
_OBIS = "Datasets/obis_seamap_gridded_1deg/occurrences_dist_sp_1deg.csv"
RUNS = {
    "Caretta caretta": ("Datasets/Oceanic_data.nc", "Datasets/caretta_data.csv", "Mediterranean"),
    "Cetorhinus maximus": ("Datasets/NE_Atlantic.nc", _OBIS, "NE Atlantic"),
    "Balaenoptera musculus": ("Datasets/California_Current.nc", _OBIS, "California Current"),
}
FEATURES = ["retention", "mean_speed", "vorticity", "latitude", "longitude", "dist_to_coast"]
KS = [8, 16, 32]
N_DRAWS = 5
N_TEST = 300
THRESHOLD = 0.5
SEED = 42
CKPT = "models/tabicl-classifier-v2-20260212.ckpt"


def _metrics(y, p, thr=THRESHOLD):
    from sklearn.metrics import roc_auc_score, f1_score
    return (float(roc_auc_score(y, p)),
            float(f1_score(y, (p >= thr).astype(int), zero_division=0)))


def _kshot_draw(tr, y, k, rng):
    """Class-balanced k-shot draw from the training fold."""
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
    per = k // 2
    if len(pos) < per or len(neg) < per:
        return None
    return np.concatenate([rng.choice(pos, per, replace=False),
                           rng.choice(neg, per, replace=False)])


def classical(name, seed):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=10,
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "gbt":
        return GradientBoostingClassifier(random_state=seed)
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=seed))
    raise ValueError(name)


def run_species(species, fig_root=FIG_ROOT, device=None, do_kshot=True):
    import retention_core as rc
    import label_cv as lcv
    import tabllm_prompt as tp
    from tabicl import TabICLClassifier
    from sklearn.inspection import permutation_importance

    vel_nc, csv, basin = RUNS[species]
    t0 = time.time()
    cells = tp.load_cells(species, vel_nc, csv)
    X, lats, lons, vel = cells["X"], cells["lats"], cells["lons"], cells["vel"]
    df = rc.load_sightings(csv, species, vel)

    dens = lcv.kde_density(df["longitude"].values, df["latitude"].values,
                          df["weighted_records"].values, lons, lats)
    y = lcv.core_area_label(dens)
    splits, region = lcv.loro_kmeans_folds(lats, lons)
    coords = np.column_stack([np.asarray(lats, float), np.asarray(lons, float)])

    out = {
        "species": species, "basin": basin,
        "label": "KDE core-area (50% volume contour of a weighted Gaussian KDE)",
        "cv": "contiguous leave-one-region-out (k-means, 5 regions)",
        "n_cells": int(len(X)), "prevalence": float(y.mean()),
        "n_occurrence_rows": int(len(df)),
        "threshold": THRESHOLD, "n_estimators": 2, "features": FEATURES,
        "tabllm": None,
        "folds": [], "folds_unscoreable": [], "kshot": {},
    }

    clf = TabICLClassifier(n_estimators=2, device=device, random_state=SEED,
                           model_path=CKPT)
    rng = np.random.default_rng(SEED)

    for i, (tr, te) in enumerate(splits):
        kept = lcv.balanced_subsample(te, y, N_TEST, rng)
        if kept is None or len(np.unique(y[tr])) < 2:
            out["folds_unscoreable"].append(
                {"fold": i, "n_region_cells": int(len(te)),
                 "n_pos_in_region": int(y[te].sum()),
                 "reason": "held-out region contains a single class"})
            print(f"  fold {i}: UNSCOREABLE ({int(y[te].sum())} positives in region)",
                  flush=True)
            continue
        sep = lcv.separation_deg(coords[kept], coords[tr])
        rec = {"fold": i, "n_context": int(len(tr)), "n_test": int(len(kept)),
               "n_pos_test": int(y[kept].sum()),
               "n_region_cells": int(len(te)),
               "separation_deg_median": float(np.median(sep)),
               "separation_deg_p90": float(np.percentile(sep, 90))}

        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[kept])[:, 1]
        rec["auc_tabicl"], rec["f1_tabicl"] = _metrics(y[kept], p)

        rf = classical("rf", SEED)
        rf.fit(X[tr], y[tr])
        prf = rf.predict_proba(X[kept])[:, 1]
        rec["auc_rf_full"], rec["f1_rf_full"] = _metrics(y[kept], prf)

        pi = permutation_importance(rf, X[kept], y[kept], scoring="roc_auc",
                                    n_repeats=10, random_state=SEED, n_jobs=-1)
        rec["perm_importance"] = {f: float(v) for f, v in
                                  zip(FEATURES, pi.importances_mean)}

        out["folds"].append(rec)
        print(f"  [{time.time()-t0:.0f}s] fold {i}: n_test={len(kept)} "
              f"AUC tabicl={rec['auc_tabicl']:.4f} rf={rec['auc_rf_full']:.4f} | "
              f"F1 tabicl={rec['f1_tabicl']:.4f} rf={rec['f1_rf_full']:.4f}", flush=True)

        if do_kshot:
            for k in KS:
                bucket = out["kshot"].setdefault(str(k), {})
                for draw in range(N_DRAWS):
                    drng = np.random.default_rng(SEED + 1000 * k + draw)
                    sh = _kshot_draw(tr, y, k, drng)
                    if sh is None:
                        continue
                    ck = TabICLClassifier(n_estimators=2, device=device,
                                          random_state=SEED, model_path=CKPT)
                    ck.fit(X[sh], y[sh])
                    a, f = _metrics(y[kept], ck.predict_proba(X[kept])[:, 1])
                    bucket.setdefault("tabicl", []).append({"fold": i, "draw": draw,
                                                            "auc": a, "f1": f})
                    for name in ("rf", "gbt", "knn", "logreg"):
                        m = classical(name, SEED)
                        m.fit(X[sh], y[sh])
                        a, f = _metrics(y[kept], m.predict_proba(X[kept])[:, 1])
                        bucket.setdefault(name, []).append({"fold": i, "draw": draw,
                                                            "auc": a, "f1": f})
            print(f"  [{time.time()-t0:.0f}s] fold {i}: k-shot done", flush=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    d = os.path.join(fig_root, species.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[saved] {path}  ({out['elapsed_s']}s, "
          f"{len(out['folds'])} scored / {len(out['folds_unscoreable'])} unscoreable)",
          flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", action="append", default=None, choices=list(RUNS))
    ap.add_argument("--fig-root", default=FIG_ROOT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-kshot", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    for sp in (a.species or list(RUNS)):
        print(f"\n=== {sp} ===", flush=True)
        run_species(sp, a.fig_root, a.device, not a.no_kshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
