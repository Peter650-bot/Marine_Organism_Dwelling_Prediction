#!/usr/bin/env python3
"""
bench_context_sweep.py -- loggerhead: wall-clock time and ROC-AUC of TabICLv2
(no KV cache) and the nested-CV tuned Random Forest as the context grows.

    context_sweep.jsonl   one line per (fold, context size, draw, model),
                          with the per-cell test probabilities kept so
                          make_context_sweep_figure.py can bootstrap the AUC

Replays run_pipeline.py's folds and 300-cell balanced test sets (Caretta has 4
scoreable folds; fold 2 has no positives). Subsampled contexts are stratified
random draws of the training fold (3 draws per fold, seeds SEED + 100*fold +
draw, the same draws the earlier 2k/5k runs in time.csv used). Full context is
the whole training fold, one draw.

Both models get 4 CPU threads, and for each (fold, context, draw) they run
back to back on the same rows, so machine-load drift hits both equally.
  TabICLv2   fit + predict_proba on the 300 test cells, n_estimators=2
  tuned RF   tune_rf.py's nested leave-one-region-out search (32 configs x
             inner folds) + the AUC refit + the F1 refit, as in the paper
  fixed RF   run_pipeline.py's single RF config, fit + predict (reference)
At full context the tuned RF's inner subsamples come from tune_rf.py's single
INNER_SEED stream, so its AUC reproduces results.json (0.899); subsampled
contexts use a per-(fold, draw) inner stream.

Resumable: records already in the output file are skipped.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

SEED, N_TEST, THREADS = 42, 300, 4
CKPT = "models/tabicl-classifier-v2-20260212.ckpt"
SPECIES = "Caretta caretta"
CONTEXTS = [500, 1000, 2000, 5000, 0]          # 0 = full training fold
DRAWS = 3


def stratified_subsample(tr, y, n, rng):
    """Random subsample of the training rows that keeps the class ratio."""
    if n >= len(tr):
        return tr
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
    n_pos = max(1, int(round(n * len(pos) / len(tr))))
    return np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, n - n_pos, replace=False)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="context_sweep.jsonl")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(THREADS)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from tabicl import TabICLClassifier

    import label_cv as lcv
    import retention_core as rc
    import tabllm_prompt as tp
    import tune_rf as T
    from run_pipeline import RUNS

    def make_rf(cfg, seed=SEED):
        return RandomForestClassifier(n_estimators=200, random_state=seed,
                                      n_jobs=THREADS, **cfg)
    T.make_rf = make_rf          # tune_on_train builds its forests through this

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            for line in fh:
                r = json.loads(line)
                done.add((r["fold"], r["context"], r["draw"], r["model"]))
    out = open(a.out, "a")

    def emit(rec, p, kept):
        rec.update(species=SPECIES, n_test=int(len(kept)), threads=THREADS,
                   auc=float(roc_auc_score(y[kept], p)),
                   test_idx=kept.tolist(), y_test=y[kept].astype(int).tolist(),
                   p_test=np.round(p, 6).tolist())
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"fold {rec['fold']} ctx {rec['n_context']:>6,} draw {rec['draw']} "
              f"{rec['model']:9s} {rec['total_s']:8.1f} s  AUC {rec['auc']:.4f}",
              flush=True)

    vel_nc, csv, _ = RUNS[SPECIES]
    cells = tp.load_cells(SPECIES, vel_nc, csv)
    X, lats, lons, vel = cells["X"], cells["lats"], cells["lons"], cells["vel"]
    df = rc.load_sightings(csv, SPECIES, vel)
    y = lcv.core_area_label(lcv.kde_density(
        df["longitude"].values, df["latitude"].values,
        df["weighted_records"].values, lons, lats))
    splits, region = lcv.loro_kmeans_folds(lats, lons)

    # first-call overhead (checkpoint load, kernel warm-up) kept out of fold 0
    w = TabICLClassifier(n_estimators=2, device=None, random_state=SEED, model_path=CKPT)
    w.fit(X[:500], y[:500]); w.predict_proba(X[:50])

    rng = np.random.default_rng(SEED)                      # run_pipeline's test draws
    paper_inner = np.random.default_rng(T.INNER_SEED)      # tune_rf.py's inner stream
    for i, (tr, te) in enumerate(splits):
        kept = lcv.balanced_subsample(te, y, N_TEST, rng)
        if kept is None or len(np.unique(y[tr])) < 2:
            continue
        for ctx_n in CONTEXTS:
            for d in range(1 if ctx_n == 0 else DRAWS):
                ctx = tr if ctx_n == 0 else stratified_subsample(
                    tr, y, ctx_n, np.random.default_rng(SEED + 100 * i + d))
                base = {"fold": i, "context": ctx_n or "full", "draw": d,
                        "n_context": int(len(ctx)), "n_pos_context": int(y[ctx].sum())}
                # the paper stream must be consumed for every fold, even on resume
                inner = paper_inner if ctx_n == 0 else np.random.default_rng(
                    T.INNER_SEED + 100 * i + d)
                key = (i, base["context"], d)

                if key + ("tabicl",) not in done:
                    c = TabICLClassifier(n_estimators=2, device=None,
                                         random_state=SEED, model_path=CKPT)
                    t = time.perf_counter(); c.fit(X[ctx], y[ctx])
                    fit_s = time.perf_counter() - t
                    t = time.perf_counter(); p = c.predict_proba(X[kept])[:, 1]
                    pred_s = time.perf_counter() - t
                    emit(dict(base, model="tabicl", fit_s=fit_s, predict_s=pred_s,
                              total_s=fit_s + pred_s), p, kept)
                    del c

                if key + ("rf_tuned",) not in done:
                    t = time.perf_counter()
                    cfg_a, cfg_f, n_in = T.tune_on_train(X, y, ctx, region, inner)
                    p = make_rf(cfg_a).fit(X[ctx], y[ctx]).predict_proba(X[kept])[:, 1]
                    make_rf(cfg_f).fit(X[ctx], y[ctx]).predict_proba(X[kept])
                    emit(dict(base, model="rf_tuned", total_s=time.perf_counter() - t,
                              inner_folds=n_in, n_fits=32 * n_in + 2,
                              cfg_auc={k: str(v) for k, v in cfg_a.items()}), p, kept)
                elif ctx_n == 0:
                    # replay the inner draws so later folds see the paper's stream
                    _replay_inner(X, y, ctx, region, inner, lcv)

                if key + ("rf_fixed",) not in done:
                    t = time.perf_counter()
                    p = make_rf(T.BASELINE).fit(X[ctx], y[ctx]).predict_proba(X[kept])[:, 1]
                    emit(dict(base, model="rf_fixed", total_s=time.perf_counter() - t),
                         p, kept)
    out.close()
    print("[saved]", a.out)


def _replay_inner(X, y, tr, region, inner_rng, lcv):
    """Consume inner_rng exactly as tune_rf.tune_on_train does, without fitting."""
    for r in np.unique(region[tr]):
        itr, ite = tr[region[tr] != r], tr[region[tr] == r]
        if len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2:
            continue
        lcv.balanced_subsample(ite, y, N_TEST, inner_rng)


if __name__ == "__main__":
    main()
