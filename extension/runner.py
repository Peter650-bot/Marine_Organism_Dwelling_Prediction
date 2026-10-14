"""
runner.py -- the shared benchmark loop for run_tabicl.py and run_maxent.py.

    main   species x spatial (pseudodisc) folds 1..5 -> results/<model>.jsonl
    sweep  SWEEP species x spatial (pseudodisc) fold 1 x context sizes CONTEXTS
           (3 stratified draws per subsampled size, 1 at full context)
           -> results/sweep_<model>.jsonl

A model is a function fit_predict(Xtr, ytr, Xte) -> (scores, fit_s, predict_s).
One JSON line per fold (or per context draw) with the test labels and scores,
so evaluate.py can recompute every AUC. Resumable: finished keys are skipped.
"""

import argparse
import json
import os
import zlib

import numpy as np

import msp_data as md

CONTEXTS = [500, 1000, 2000, 5000, 0]        # 0 = full training fold
DRAWS = 3
N_SWEEP = 6


def auc(y, p):
    """ROC-AUC via the Mann-Whitney rank statistic (ties count one half)."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    r = rankdata(p)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sweep_species(species):
    """One species per (ecoregion x 'many' tercile) first, then fill to N_SWEEP."""
    with open(os.path.join(md.RESULTS, "species.json")) as fh:
        strata = json.load(fh)["strata"]
    out, seen = [], set()
    for tier in ("many", "mid", "few"):
        for sp in species:
            eco = strata[sp][0]
            if strata[sp][1] == tier and (eco, tier) not in seen and len(out) < N_SWEEP:
                out.append(sp)
                seen.add((eco, tier))
    return out


def run(model, fit_predict, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="context-size sweep")
    ap.add_argument("--n-species", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new records (pilot)")
    ap.add_argument("--species", nargs="*", help="restrict to these species")
    a = ap.parse_args(argv)

    preds = md.select_predictors()
    species = md.select_species(a.n_species)
    if a.sweep:
        species = sweep_species(species)
    if a.species:
        species = [s for s in species if s in a.species] or a.species
    out_path = os.path.join(md.RESULTS, ("sweep_" if a.sweep else "") + f"{model}.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                r = json.loads(line)
                done.add((r["species"], r["kind"], r["k"], r.get("context"), r.get("draw")))
    fh = open(out_path, "a")
    new = 0

    def jobs():
        for sp in species:
            if a.sweep:
                for c in CONTEXTS:
                    for d in range(1 if c == 0 else DRAWS):
                        yield sp, "pseudodisc", 1, c or "full", d
            else:
                for kind in md.KINDS:
                    for k in range(1, 6):
                        yield sp, kind, k, None, None

    for sp, kind, k, ctx, draw in jobs():
        if (sp, kind, k, ctx, draw) in done:
            continue
        d = md.fold_data(sp, kind, k, preds)
        if min(d["n_tr_pres"], d["n_tr_bg"], d["n_te_pres"], d["n_te_bg"]) == 0:
            print(f"[skip] {sp} {kind} k={k}: an empty class "
                  f"({d['n_tr_pres']}+{d['n_tr_bg']} / {d['n_te_pres']}+{d['n_te_bg']})")
            continue
        Xtr, ytr = d["Xtr"], d["ytr"]
        if a.sweep and ctx != "full":
            rng = np.random.default_rng(md.SEED + 100 * draw + zlib.crc32(sp.encode()))
            idx = md.subsample_context(ytr, ctx, rng)
            Xtr, ytr = Xtr[idx], ytr[idx]
        p, fit_s, pred_s = fit_predict(Xtr, ytr, d["Xte"])
        rec = {"species": sp, "kind": kind, "k": k, "model": model,
               "context": ctx, "draw": draw, "n_context": int(len(ytr)),
               "n_context_pres": int(ytr.sum()),
               **{key: v for key, v in d.items() if key.startswith("n_")},
               "fit_s": fit_s, "predict_s": pred_s, "total_s": fit_s + pred_s,
               "auc": auc(d["yte"], p), "n_features": len(preds),
               "y_test": d["yte"].astype(int).tolist(),
               "p_test": np.asarray(p, float).tolist()}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"{model:7s} {sp:30s} {kind:10s} k={k} ctx={str(ctx or ''):>5s} "
              f"n={len(ytr):>6,}  {fit_s + pred_s:7.1f} s  AUC {rec['auc']:.4f}", flush=True)
        new += 1
        if a.limit and new >= a.limit:
            break
    fh.close()
    print(f"[saved] {out_path} (+{new} records)")
