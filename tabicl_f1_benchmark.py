#!/usr/bin/env python3
"""
tabicl_f1_benchmark.py — per-fold F1 for TabICL (full-context) vs the RF-full
ceiling under spatial-block CV, with a 95% confidence interval over the folds.

Why this exists separately from ``tabicl_benchmark.py``: that script records
only ROC-AUC, which is threshold-free and therefore never needs the hard
HIGH/LOW calls. F1 does. Rather than change the recorded run, this script
replays the *same* experiment and recovers the predictions.

**The reproduction check is the point.** It rebuilds the identical folds, seeds,
context rows and test subsamples used by ``tabicl_benchmark.run_benchmark`` in
``full`` mode (``seed_variant='full'``), re-derives each fold's AUC, and asserts
it against ``figures/<species>/tabicl_results.json``. If those AUCs match, the
F1 values are measured on the same predictions as the published AUCs rather than
on an independent draw. A mismatch means the seeds/folds have drifted and the
comparison is no longer paired — the script fails loudly instead.

Scoring: threshold P(HIGH) = 0.5, positive class = HIGH dwelling, no per-fold
threshold tuning (tuning on the held-out fold would leak). Test subsamples are
class-balanced at n=300/fold, so the trivial all-positive baseline is F1 ~ 0.667
— judge results against that, not against 0.

Outputs (under ``figures/<species>/``):
  tabicl_f1_spatial.json          per-fold F1/precision/recall/AUC + raw probs
  tabicl_f1_spatial_summary.json  means, SD, SEM, 95% t-CI, paired t/Wilcoxon

**Which weights.** The run is TabICLv2 (``--checkpoint-version``, default
``tabicl-classifier-v2-20260212.ckpt``). The generation is *verified from the
loaded checkpoint config*, not from the flag, because a local ``--model-path``
overrides ``checkpoint_version`` inside ``TabICLClassifier`` — so the flag alone
would prove nothing about the weights in memory. ``--want-arch`` aborts on a
mismatch and the fingerprint is written into the output JSON.

Runtime: ~10 min on 4 CPU cores for Caretta (5 forward passes over ~10.9k
context rows); ~3x that for Cetorhinus (~29.5k rows). Needs the local checkpoint
under ``models/`` — see ``--model-path``.

CLI:
    python tabicl_f1_benchmark.py                     # Caretta defaults
    python tabicl_f1_benchmark.py --threshold 0.5 --n-estimators 2
    python tabicl_f1_benchmark.py --species "Cetorhinus maximus"  # preset paths
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import tabicl_benchmark as tb   # numpy-only at import time; torch stays lazy

SEED = 42
DEFAULTS = dict(
    species="Caretta caretta",
    velocity="Datasets/Oceanic_data.nc",
    species_csv="Datasets/caretta_data.csv",
    model_path="models/tabicl-classifier-v2-20260212.ckpt",
    n_test=300,
    n_estimators=2,      # matches the recorded tabicl_results.json run
    seed_variant="full",
    scheme="spatial",
    threshold=0.5,
)

# The species this script was written around. A second species reaches the same
# code path through the CLI, so nothing here may assume Mediterranean geometry.
_OBIS = ("Datasets/obis_seamap_custom_6a21668089aef_20260604_075522_"
         "dist_sp_1deg_csv_97740/obis_seamap_custom_6a21668089aef_"
         "20260604_075522_dist_sp_1deg_csv.csv")

SPECIES_PRESETS = {
    "Cetorhinus maximus": dict(
        velocity="Datasets/NE_Atlantic.nc", species_csv=_OBIS),
    # California Current only. Blue whale is cosmopolitan, so this cube covers
    # one basin of its range by design -- alignment_check reports MISMATCH for
    # exactly that reason and it is expected, not a fault.
    "Balaenoptera musculus": dict(
        velocity="Datasets/California_Current.nc", species_csv=_OBIS),
}


def ci95(x):
    """Mean, half-width of the 95% t-CI, SD and SEM over the folds."""
    from scipy import stats
    x = np.asarray(x, float)
    n = x.size
    sd = x.std(ddof=1)
    sem = sd / np.sqrt(n)
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * sem), float(sd), float(sem)


def run(args):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                                 recall_score)
    from sklearn.ensemble import RandomForestClassifier
    from tabicl import TabICLClassifier
    from scipy import stats

    import tabllm_pipeline as tp
    from tabicl_benchmark import select_context, assert_checkpoint

    t0 = time.time()
    cells = tp.load_cells(args.species, args.velocity, args.species_csv,
                          fig_dir=args.fig_dir)
    X, y_cls, groups = cells["X"], cells["y_cls"], cells["groups"]
    out_dir = cells["out_dir"]
    print(f"[{time.time()-t0:.0f}s] n_cells={cells['n_cells']} "
          f"prevalence={cells['prevalence']:.4f} n_blocks={cells['n_blocks']}",
          flush=True)

    ng = len(np.unique(groups))
    folds_iter = list(GroupKFold(min(5, ng)).split(X, y_cls, groups))

    ref_path = os.path.join(out_dir, "tabicl_results.json")
    ref_pt = None
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        ref_pt = next((p for p in ref["points"]
                       if p["scheme"] == args.scheme and p["mode"] == "full"), None)
    if ref_pt is None:
        print(f"WARNING: no full-mode '{args.scheme}' point in {ref_path}; "
              f"the reproduction check will be skipped.", flush=True)

    # Per-fold checkpointing. A full-context fold costs minutes (~12 for
    # Cetorhinus), so a run killed at fold 4 used to throw away everything: the
    # results were only written after the loop. Each completed fold is now
    # flushed to a partial file and reused on the next invocation, but only when
    # every setting that could change a number matches -- otherwise the resumed
    # folds and the fresh ones would come from two different experiments.
    part_path = os.path.join(out_dir, f"tabicl_f1_{args.scheme}.partial.json")
    config = {"species": args.species, "scheme": args.scheme, "mode": "full",
              "threshold": args.threshold, "n_estimators": args.n_estimators,
              "n_test": args.n_test, "seed_variant": args.seed_variant,
              "checkpoint_version": args.checkpoint_version,
              "want_arch": args.want_arch, "n_cells": int(cells["n_cells"]),
              "n_folds": len(folds_iter)}

    rows, done, resumed_fingerprint = [], {}, None
    if os.path.exists(part_path) and not args.no_resume:
        part = json.load(open(part_path))
        if part.get("config") == config:
            done = {r["fold"]: r for r in part["folds"]}
            resumed_fingerprint = part.get("checkpoint")
            print(f"[{time.time()-t0:.0f}s] resuming: {sorted(done)} already done "
                  f"({part_path})", flush=True)
        else:
            print(f"[{time.time()-t0:.0f}s] ignoring stale {part_path} "
                  f"(settings changed)", flush=True)

    def flush_partial():
        os.makedirs(out_dir, exist_ok=True)
        with open(part_path, "w") as fh:
            json.dump({"config": config, "checkpoint": fingerprint,
                       "folds": rows}, fh)

    fingerprint, verified = resumed_fingerprint, False
    for fold, (tr, te) in enumerate(folds_iter):
        if fold in done:
            rows.append(done[fold])
            print(f"[{time.time()-t0:.0f}s] fold {fold}: reused from partial "
                  f"(AUC tabicl={done[fold]['auc_tabicl']:.4f})", flush=True)
            continue
        seed = tp._seed_for(fold, args.scheme, args.species, -1, args.seed_variant)
        ctx = select_context(X, y_cls, tr, "full", None, seed, tp.sample_exemplars)
        kept = tp.subsample_test(te, y_cls, n=args.n_test, mode="balanced", seed=seed)

        # same hard leakage assertions as tabicl_benchmark, every fold
        assert np.intersect1d(ctx, kept).size == 0, "context overlaps test cells"
        assert set(groups[ctx]).isdisjoint(set(groups[kept])), \
            "context shares a spatial block with test"

        yk = y_cls[kept]

        clf = TabICLClassifier(
            n_estimators=args.n_estimators, device=args.device, n_jobs=args.n_jobs,
            random_state=SEED, checkpoint_version=args.checkpoint_version,
            **({"model_path": args.model_path} if args.model_path else {}))
        clf.fit(X[ctx], y_cls[ctx])
        if not verified:
            # Verified from the loaded config, not from the flag: a local
            # --model-path overrides checkpoint_version inside the library.
            # Re-checked once per process, so a resumed run still proves the
            # weights it is adding folds with.
            fingerprint, verified = assert_checkpoint(clf, args.want_arch), True
            print(f"[{time.time()-t0:.0f}s] weights: {fingerprint['architecture']} "
                  f"<- {fingerprint['model_path_loaded']}", flush=True)
        p_tab = clf.predict_proba(X[kept])[:, list(clf.classes_).index(1)]

        # identical to tabllm_pipeline._baseline_scores["rf_full"]
        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    class_weight="balanced", random_state=seed)
        rf.fit(X[tr], y_cls[tr])
        p_rf = rf.predict_proba(X[kept])[:, 1]

        row = {"fold": fold, "n_context": int(len(ctx)), "n_test": int(len(kept)),
               "n_pos_test": int(yk.sum()), "seed": int(seed)}
        for name, p in (("tabicl", p_tab), ("rf_full", p_rf)):
            yh = (p >= args.threshold).astype(int)
            row[f"auc_{name}"] = float(roc_auc_score(yk, p))
            row[f"f1_{name}"] = float(f1_score(yk, yh, zero_division=0))
            row[f"precision_{name}"] = float(precision_score(yk, yh, zero_division=0))
            row[f"recall_{name}"] = float(recall_score(yk, yh, zero_division=0))
            row[f"pred_pos_rate_{name}"] = float(yh.mean())
            row[f"proba_{name}"] = [float(v) for v in p]
        row["y_true"] = [int(v) for v in yk]
        rows.append(row)
        flush_partial()

        ref_txt = ""
        if ref_pt:
            ref_txt = (f" (ref {ref_pt['auc_tabicl_folds'][fold]:.4f} / "
                       f"{ref_pt['auc_rf_full_folds'][fold]:.4f})")
        print(f"[{time.time()-t0:.0f}s] fold {fold}: "
              f"AUC tabicl={row['auc_tabicl']:.4f} rf={row['auc_rf_full']:.4f}"
              f"{ref_txt} | F1 tabicl={row['f1_tabicl']:.4f} "
              f"rf={row['f1_rf_full']:.4f}", flush=True)

    # --- reproduction check: are these the published predictions? ------------
    repro = {}
    if ref_pt:
        repro = {
            "auc_tabicl_max_abs_diff": float(np.max(np.abs(
                np.array([r["auc_tabicl"] for r in rows])
                - np.array(ref_pt["auc_tabicl_folds"])))),
            "auc_rf_full_max_abs_diff": float(np.max(np.abs(
                np.array([r["auc_rf_full"] for r in rows])
                - np.array(ref_pt["auc_rf_full_folds"])))),
        }
        print("reproduction check:", repro, flush=True)
        if max(repro.values()) > args.repro_tol:
            raise SystemExit(
                f"ABORT: per-fold AUCs deviate from {ref_path} by more than "
                f"{args.repro_tol}. The folds/seeds have drifted, so these F1 "
                f"values would NOT be paired with the published AUCs.")

    raw = {"species": args.species, "scheme": args.scheme, "mode": "full",
           "threshold": args.threshold, "n_estimators": args.n_estimators,
           "n_test": args.n_test, "seed_variant": args.seed_variant,
           "positive_class": "HIGH dwelling",
           "n_cells": int(cells["n_cells"]), "prevalence": float(cells["prevalence"]),
           "n_blocks": int(cells["n_blocks"]), "n_folds": len(folds_iter),
           "checkpoint": fingerprint,
           "reproduction_check_vs_tabicl_results_json": repro,
           "folds": rows}
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"tabicl_f1_{args.scheme}.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"[saved] {raw_path}")
    # The full result supersedes the checkpoint; leaving it would let a later
    # settings change resume from folds it can no longer be compared with.
    if os.path.exists(part_path):
        os.remove(part_path)

    # --- summary: mean, 95% CI, paired test ---------------------------------
    summary = {}
    print()
    for m in ("tabicl", "rf_full"):
        f1 = np.array([r[f"f1_{m}"] for r in rows])
        mean, half, sd, sem = ci95(f1)
        summary[m] = dict(folds=f1.tolist(), mean=mean, half=half, sd=sd, sem=sem,
                          lo=mean - half, hi=mean + half)
        print(f"{m:8s} F1 per fold: " + "  ".join(f"{v:.4f}" for v in f1))
        print(f"{'':8s} mean={mean:.4f}  SD={sd:.4f}  SEM={sem:.4f}  "
              f"95% CI [{mean-half:.4f}, {mean+half:.4f}]  (±{half:.4f})")
        print(f"{'':8s} precision={np.mean([r[f'precision_{m}'] for r in rows]):.4f}  "
              f"recall={np.mean([r[f'recall_{m}'] for r in rows]):.4f}  "
              f"pred-pos-rate={np.mean([r[f'pred_pos_rate_{m}'] for r in rows]):.4f}")
        print()

    a = np.array(summary["tabicl"]["folds"])
    b = np.array(summary["rf_full"]["folds"])
    delta = a - b
    dm, dh, _, _ = ci95(delta)
    t, p = stats.ttest_rel(a, b)
    try:
        _, pw = stats.wilcoxon(a, b)
    except Exception:
        pw = float("nan")
    summary["delta"] = {"folds": delta.tolist(), "mean": float(dm), "half": float(dh),
                        "lo": float(dm - dh), "hi": float(dm + dh),
                        "t": float(t), "p": float(p), "wilcoxon_p": float(pw),
                        "wins": int((delta > 0).sum())}
    print("paired delta (TabICL - RF-full): " + "  ".join(f"{v:+.4f}" for v in delta))
    print(f"  mean={dm:+.4f}  95% CI [{dm-dh:+.4f}, {dm+dh:+.4f}]  "
          f"t p={p:.4f}  wilcoxon p={pw:.4f}  wins {(delta>0).sum()}/{len(delta)}")
    print(f"  CI {'EXCLUDES' if (dm - dh > 0 or dm + dh < 0) else 'INCLUDES'} zero")

    sum_path = os.path.join(out_dir, f"tabicl_f1_{args.scheme}_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {sum_path}  ({time.time()-t0:.0f}s total)")
    return raw, summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default=DEFAULTS["species"])
    # Left as None so a preset can fill them in; resolved after parsing.
    ap.add_argument("--velocity", default=None)
    ap.add_argument("--species-csv", default=None)
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--scheme", default=DEFAULTS["scheme"],
                    choices=("spatial", "random"))
    ap.add_argument("--threshold", type=float, default=DEFAULTS["threshold"],
                    help="decision threshold on P(HIGH) for the F1 calls")
    ap.add_argument("--n-test", type=int, default=DEFAULTS["n_test"])
    ap.add_argument("--n-estimators", type=int, default=DEFAULTS["n_estimators"],
                    help="MUST match the recorded run or the AUC check fails")
    ap.add_argument("--seed-variant", default=DEFAULTS["seed_variant"],
                    help="variant tag fed to tabllm_pipeline._seed_for; must "
                         "match the recorded run to stay paired")
    ap.add_argument("--model-path", default=DEFAULTS["model_path"])
    ap.add_argument("--checkpoint-version", default=tb.CKPT_V2,
                    choices=(tb.CKPT_V2, tb.CKPT_V1_1, tb.CKPT_V1),
                    help="pretrained TabICL generation (default: TabICLv2)")
    ap.add_argument("--want-arch", default="v2", choices=("v1", "v2"),
                    help="architecture the loaded weights must match, or abort")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute every fold, ignoring any partial file")
    ap.add_argument("--repro-tol", type=float, default=1e-9,
                    help="max tolerated per-fold AUC deviation from "
                         "tabicl_results.json before aborting")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    preset = SPECIES_PRESETS.get(args.species, DEFAULTS)
    args.velocity = args.velocity or preset["velocity"]
    args.species_csv = args.species_csv or preset["species_csv"]
    for label, path in (("velocity", args.velocity), ("species csv", args.species_csv)):
        if not os.path.exists(path):
            ap.error(f"{label} not found: {path}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
