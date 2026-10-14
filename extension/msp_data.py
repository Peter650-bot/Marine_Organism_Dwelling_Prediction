#!/usr/bin/env python3
"""
msp_data.py -- MarineSPEED loaders shared by run_tabicl.py and run_maxent.py,
so both models are trained and tested on byte-identical rows.

Only numpy and pandas are imported here: run_maxent.py runs with elapid's
isolated site directory on PYTHONPATH (older scikit-learn), run_tabicl.py with
the project's normal environment.

Design
  * Features: environmental predictors only. longitude/latitude are dropped.
    From the 71 Bio-ORACLE + MARSPEC layers a de-correlated subset is chosen
    once: greedy |Spearman rho| < 0.7 on the random background, walking a
    priority list (SST, salinity, depth, distance to shore, chlorophyll, ...).
    Saved to results/predictors.json.
  * Folds: the benchmark's own 5-fold splits. "random" = random partition of
    presences + random background; "pseudodisc" = spatial disc partition of
    presences with a 200 km buffer on the background (MarineSPEED's spatial CV).
    Fold-file rows filtered to {species, "background"} align row-for-row with
    the species' presences / the 20,000 background points (the R package's
    kfold_data rule); TRUE = training row, FALSE = test row, NA = excluded.
  * Caps (identical for both models, seeded per species/kind/fold): at most
    PRES_TRAIN / PRES_TEST presences and BG_TRAIN / BG_TEST background points
    per fold. Rows with a missing predictor are dropped.
  * Species subset: stratified by dominant ecoregion (polar / temperate /
    tropical, from traits.csv) x record-count tercile. results/species.json.
"""

import json
import os
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "marinespeed")
RESULTS = os.path.join(HERE, "results")
SEED = 42
PRES_TRAIN, PRES_TEST, BG_TRAIN, BG_TEST = 5000, 2000, 5000, 2000
KINDS = ["pseudodisc"]          # spatial folds only (random folds dropped)
RHO_MAX = 0.7
PRIORITY = ["BO_sstmean", "BO_salinity", "MS_bathy_5m", "MS_biogeo05_dist_shore_5m",
            "BO_chlomean", "BO_sstrange", "BO_dissox", "BO_nitrate", "BO_ph",
            "BO_damean", "BO_parmean", "BO_phosphate", "BO_silicate", "BO_calcite",
            "BO_cloudmean", "MS_biogeo06_bathy_slope_5m", "MS_biogeo11_sss_range_5m",
            "BO_chlorange", "MS_biogeo07_concavity_5m",
            "MS_biogeo03_plan_curvature_5m", "MS_biogeo04_profile_curvature_5m",
            "MS_biogeo01_aspect_EW_5m", "MS_biogeo02_aspect_NS_5m"]
NON_FEATURES = {"species", "longitude", "latitude"}


def _csv(name, **kw):
    return pd.read_csv(os.path.join(DATA, name), **kw)


def load_species():
    return _csv("species.csv.gz")["species"].tolist()


def load_traits():
    return _csv("traits.csv.gz")


def load_bg():
    return _csv("random_background.csv.gz")


def _occ_files():
    d = os.path.join(DATA, "occurrences")
    files = {}
    for f in sorted(os.listdir(d)):                       # R list.files order
        if f.endswith(".csv.gz"):
            sp = f[:-len(".csv.gz")].rstrip("0123456789").rstrip()
            files.setdefault(sp, []).append(os.path.join(d, f))
    return files


_OCC = None


def load_occ(sp):
    global _OCC
    _OCC = _OCC or _occ_files()
    return pd.concat([pd.read_csv(p) for p in _OCC[sp]], ignore_index=True)


# ---- folds ---------------------------------------------------------------
# The pseudodisc background fold file has 20,000 rows per species (~10M rows),
# so each fold file is converted once to a compact cache: an int8 code matrix
# (1 train, 0 test, -1 NA) plus each species' [start, stop) row span.

def _fold_cache(kind, what):
    path = os.path.join(DATA, f"cache_{kind}_{what}.npz")
    if not os.path.exists(path):
        src = os.path.join(DATA, f"{kind}_{what}_5cv_folds.csv.gz")
        df = pd.read_csv(src, dtype={"species": "category"},
                         true_values=["TRUE"], false_values=["FALSE"])
        codes = np.full((len(df), 5), -1, np.int8)
        for j in range(5):
            col = df[f"k{j + 1}"]
            codes[col.astype(str).str.upper().eq("TRUE").values, j] = 1
            codes[col.astype(str).str.upper().eq("FALSE").values, j] = 0
        sp = df["species"].astype(str).values
        change = np.flatnonzero(sp[1:] != sp[:-1]) + 1
        starts = np.r_[0, change]
        stops = np.r_[change, len(sp)]
        names = sp[starts].astype(str)
        assert len(set(names)) == len(names), f"{src}: species rows not contiguous"
        np.savez_compressed(path, codes=codes, names=names, starts=starts, stops=stops)
    z = np.load(path, allow_pickle=False)
    return z["codes"], {n: (a, b) for n, a, b in zip(z["names"], z["starts"], z["stops"])}


_FOLDS = {}


def fold_codes(kind, sp):
    """(occurrence codes, background codes), each n_rows x 5 int8."""
    for what in ("species", "background"):
        if (kind, what) not in _FOLDS:
            _FOLDS[kind, what] = _fold_cache(kind, what)
    oc, ospan = _FOLDS[kind, "species"]
    bc, bspan = _FOLDS[kind, "background"]
    a, b = ospan[sp]
    bs = bspan.get(sp, bspan.get("background"))
    return oc[a:b], bc[bs[0]:bs[1]]


# ---- predictors ----------------------------------------------------------

def select_predictors(bg=None):
    path = os.path.join(RESULTS, "predictors.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)["predictors"]
    bg = load_bg() if bg is None else bg
    env = [c for c in bg.columns if c not in NON_FEATURES]
    order = [c for c in PRIORITY if c in env] + [c for c in env if c not in PRIORITY]
    rho = bg[env].rank().corr().abs()
    kept = []
    for c in order:
        if bg[c].nunique() > 1 and all(rho.loc[c, k] < RHO_MAX for k in kept):
            kept.append(c)
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"predictors": kept, "rule": f"greedy |Spearman rho| < {RHO_MAX} "
                   "on the 20,000 random background points, priority order",
                   "n_candidates": len(env)}, fh, indent=2)
    return kept


# ---- species subset ------------------------------------------------------

def record_counts():
    codes, span = _fold_cache("random", "species")
    return {sp: int(b - a) for sp, (a, b) in span.items()}


def select_species(n=40, seed=SEED):
    path = os.path.join(RESULTS, "species.json")
    if os.path.exists(path):
        with open(path) as fh:
            sel = json.load(fh)
        if sel["n"] == n:
            return sel["species"]
    tr = load_traits().set_index("species")
    counts = pd.Series(record_counts())
    sp = [s for s in load_species() if s in counts.index and s in tr.index]
    t = tr.loc[sp]
    eco = t[["eco_polar", "eco_temperate", "eco_tropical"]].idxmax(axis=1).str[4:]
    tercile = pd.qcut(counts[sp], 3, labels=["few", "mid", "many"])
    strata = pd.DataFrame({"eco": eco, "n": tercile, "count": counts[sp]})
    rng = np.random.default_rng(seed)
    groups = {k: list(g.index) for k, g in strata.groupby(["eco", "n"], observed=True)}
    # proportional allocation, at least one per non-empty stratum
    total = sum(len(v) for v in groups.values())
    quota = {k: max(1, round(n * len(v) / total)) for k, v in groups.items()}
    while sum(quota.values()) > n:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota[max(groups, key=lambda k: len(groups[k]) - quota[k])] += 1
    chosen = []
    for k in sorted(groups):
        chosen += sorted(rng.choice(groups[k], min(quota[k], len(groups[k])),
                                    replace=False).tolist())
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"n": n, "seed": seed, "species": chosen,
                   "strata": {sp: [strata.loc[sp, "eco"], str(strata.loc[sp, "n"]),
                                   int(strata.loc[sp, "count"])] for sp in chosen}},
                  fh, indent=2)
    return chosen


# ---- fold data -----------------------------------------------------------

_BG = None


def _cap(idx, n, rng):
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def fold_data(sp, kind, k, preds=None):
    """Training/test arrays for species sp, fold kind, fold k (1..5).

    Returns dict with Xtr, ytr, Xte, yte (float32 / int8) and row counts."""
    global _BG
    preds = preds or select_predictors()
    _BG = load_bg() if _BG is None else _BG
    occ = load_occ(sp)
    oc, bc = fold_codes(kind, sp)
    assert len(oc) == len(occ), f"{sp}/{kind}: {len(oc)} fold rows vs {len(occ)} presences"
    assert len(bc) == len(_BG), f"{sp}/{kind}: {len(bc)} bg fold rows vs {len(_BG)}"
    rng = np.random.default_rng(SEED + zlib.crc32(f"{sp}|{kind}|{k}".encode()))
    o_ok = occ[preds].notna().all(axis=1).values
    b_ok = _BG[preds].notna().all(axis=1).values
    j = k - 1
    sel = {"tr_pres": _cap(np.flatnonzero((oc[:, j] == 1) & o_ok), PRES_TRAIN, rng),
           "tr_bg": _cap(np.flatnonzero((bc[:, j] == 1) & b_ok), BG_TRAIN, rng),
           "te_pres": _cap(np.flatnonzero((oc[:, j] == 0) & o_ok), PRES_TEST, rng),
           "te_bg": _cap(np.flatnonzero((bc[:, j] == 0) & b_ok), BG_TEST, rng)}
    O, Bk = occ[preds].values.astype(np.float32), _BG[preds].values.astype(np.float32)
    Xtr = np.vstack([O[sel["tr_pres"]], Bk[sel["tr_bg"]]])
    Xte = np.vstack([O[sel["te_pres"]], Bk[sel["te_bg"]]])
    ytr = np.r_[np.ones(len(sel["tr_pres"])), np.zeros(len(sel["tr_bg"]))].astype(np.int8)
    yte = np.r_[np.ones(len(sel["te_pres"])), np.zeros(len(sel["te_bg"]))].astype(np.int8)
    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
            **{f"n_{key}": int(len(v)) for key, v in sel.items()},
            "n_pres_total": int(len(occ))}


def subsample_context(ytr, n, rng):
    """Stratified subsample of training rows (keeps the presence share)."""
    if n >= len(ytr):
        return np.arange(len(ytr))
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    n_pos = max(1, int(round(n * len(pos) / len(ytr))))
    return np.sort(np.r_[rng.choice(pos, n_pos, replace=False),
                         rng.choice(neg, n - n_pos, replace=False)])


if __name__ == "__main__":
    preds = select_predictors()
    print(f"{len(preds)} predictors:", ", ".join(preds))
    species = select_species()
    print(f"{len(species)} species selected")
    for sp in species[:3]:
        for kind in KINDS:
            for k in range(1, 6):
                d = fold_data(sp, kind, k, preds)
                assert d["n_tr_pres"] and d["n_te_pres"] and d["n_tr_bg"] and d["n_te_bg"]
            print(f"  {sp:32s} {kind:10s} k=5: train {d['n_tr_pres']}+{d['n_tr_bg']}, "
                  f"test {d['n_te_pres']}+{d['n_te_bg']} (of {d['n_pres_total']} presences)")
