#!/usr/bin/env python3
"""
aquamaps.py -- score the published AquaMaps native-range surface as an external
reference, on the IDENTICAL test cells the trained models saw, and merge the
per-fold numbers into `figures/<species>/results.json`.

AquaMaps is a fixed 0.5-degree environmental-envelope surface. It is not fitted
to this data, so it has no training step and no per-fold variation of its own.
Scoring it on the same held-out subsamples is what makes its AUC/F1 directly
comparable to the trained models' -- a single whole-grid number is not.

Two properties of the AquaMaps export must be handled or the result is wrong:

  * Only NON-ZERO probabilities are stored. A cell absent from the species CSV
    means probability 0, not "unknown". Interpolating without zero-filling lets
    `griddata`'s nearest-neighbour fallback smear high suitability across gaps
    and inflates the score. Every AquaMaps ocean cell (`hcaf_r`, exported to
    `hcaf_ocean_cells.csv`) the species CSV omits is zero-filled -- 47% of the
    Mediterranean box for Caretta.
  * Land must stay absent rather than become 0, or the interpolation is pulled
    toward the coast. Zero-fill therefore uses the ocean mask, not a dense grid.

Comparability is enforced, not assumed. `run_pipeline.py` does not persist the
test indices, but it draws them with a single `np.random.default_rng(SEED)`
advanced across folds in order, and `label_cv.balanced_subsample` consumes no
draws on the folds it rejects. The sequence is therefore exactly reproducible by
replaying it, which is what this script does -- and it then ASSERTS the replayed
`n_test` and `n_pos_test` against the recorded values before writing anything,
so a silent divergence cannot slip through.

Honest caveats, to be carried into any caption:
  * AquaMaps is built from occurrence databases that OVERLAP OBIS-SEAMAP, the
    source of this project's sightings, so the comparison is PARTLY CIRCULAR
    and should flatter AquaMaps.
  * P = 0.5 is applied for F1 to match the other models, but AquaMaps
    suitability is not a calibrated P(HIGH) for the core-area target.

CLI:
    python aquamaps.py
"""

import json
import os
import sys

import numpy as np

FIG_ROOT = "figures"
AQ_DIR = os.path.join("Datasets", "aquamaps")
SEED = 42
N_TEST = 300
THRESHOLD = 0.5


def _slug(species):
    return species.replace(" ", "_")


def load_surface(species, aq_dir=AQ_DIR):
    """AquaMaps native-range points, zero-filled over the AquaMaps ocean mask.

    Returns (lon, lat, prob, n_stored, n_zero_filled) on the 0.5-degree grid."""
    import pandas as pd
    sp = pd.read_csv(os.path.join(aq_dir, f"{_slug(species)}_native.csv"))
    ocean = pd.read_csv(os.path.join(aq_dir, "hcaf_ocean_cells.csv"))

    key = lambda df: list(zip(np.round(df["CenterLat"].to_numpy(), 4),
                              np.round(df["CenterLong"].to_numpy(), 4)))
    have = dict(zip(key(sp), sp["Probability"].astype(float).to_numpy()))
    lat, lon, prob = [], [], []
    for (la, lo) in key(ocean):
        lat.append(la)
        lon.append(lo)
        prob.append(have.get((la, lo), 0.0))
    n_fill = sum(1 for p in prob if p == 0.0)
    return (np.array(lon), np.array(lat), np.array(prob), len(have), n_fill)


def regrid(lon_pts, lat_pts, prob, lons, lats):
    """0.5-degree AquaMaps surface -> our 0.125-degree cell centres."""
    from scipy.interpolate import griddata
    pts = np.column_stack([lon_pts, lat_pts])
    targ = np.column_stack([np.asarray(lons, float), np.asarray(lats, float)])
    out = griddata(pts, prob, targ, method="linear")
    miss = ~np.isfinite(out)
    if miss.any():
        out[miss] = griddata(pts, prob, targ[miss], method="nearest")
    return np.clip(out, 0.0, 1.0), int(miss.sum())


def main():
    sys.path.insert(0, os.getcwd())
    import retention_core as rc
    import label_cv as lcv
    import tabllm_prompt as tp
    import run_pipeline as rp
    from sklearn.metrics import roc_auc_score, f1_score

    for species, (vel_nc, csv, _basin) in rp.RUNS.items():
        path = os.path.join(FIG_ROOT, _slug(species), "results.json")
        if not os.path.exists(path):
            print(f"skip {species}: no {path}")
            continue
        with open(path) as fh:
            res = json.load(fh)

        cells = tp.load_cells(species, vel_nc, csv)
        lats, lons, vel = cells["lats"], cells["lons"], cells["vel"]
        df = rc.load_sightings(csv, species, vel)
        dens = lcv.kde_density(df["longitude"].values, df["latitude"].values,
                               df["weighted_records"].values, lons, lats)
        y = lcv.core_area_label(dens)
        splits, _ = lcv.loro_kmeans_folds(lats, lons)

        try:
            lo, la, pr, _n, _f = load_surface(species)
        except FileNotFoundError:
            print(f"skip {species}: no AquaMaps CSV")
            continue
        suit, _nn = regrid(lo, la, pr, lons, lats)

        rng = np.random.default_rng(SEED)          # replay the exact draw order
        recorded = {f["fold"]: f for f in res["folds"]}
        updated = 0
        for i, (tr, te) in enumerate(splits):
            kept = lcv.balanced_subsample(te, y, N_TEST, rng)
            if kept is None or len(np.unique(y[tr])) < 2:
                continue
            rec = recorded.get(i)
            if rec is None:
                raise SystemExit(f"{species} fold {i}: replayed a scoreable fold "
                                 f"that is absent from results.json")
            if len(kept) != rec["n_test"] or int(y[kept].sum()) != rec["n_pos_test"]:
                raise SystemExit(
                    f"{species} fold {i}: replay mismatch -- "
                    f"n_test {len(kept)} vs {rec['n_test']}, "
                    f"n_pos {int(y[kept].sum())} vs {rec['n_pos_test']}")
            s = suit[kept]
            rec["auc_aquamaps"] = float(roc_auc_score(y[kept], s))
            rec["f1_aquamaps"] = float(f1_score(y[kept], (s >= THRESHOLD).astype(int),
                                                zero_division=0))
            updated += 1

        res["aquamaps"] = {
            "source": "AquaMaps native range (hcaf_species_native), zero-filled",
            "untrained": True,
            "note": ("fixed 0.5-degree surface, never fitted to this data; scored "
                     "on the identical test cells as the trained models. Partly "
                     "circular: AquaMaps occurrences overlap OBIS-SEAMAP."),
            "folds_scored": updated,
        }
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)
        a = np.array([f["auc_aquamaps"] for f in res["folds"]], float)
        b = np.array([f["f1_aquamaps"] for f in res["folds"]], float)
        print(f"{species}: {updated} folds verified + scored | "
              f"AUC {a.mean():.4f} (sd {a.std(ddof=1):.4f}, worst {a.min():.4f}) | "
              f"F1 {b.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
