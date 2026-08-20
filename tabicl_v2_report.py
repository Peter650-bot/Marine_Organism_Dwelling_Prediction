#!/usr/bin/env python3
"""
tabicl_v2_report.py — one table for the TabICLv2 vs RF-full comparison across
every species that has been run, under spatial-block CV only.

Reads the artefacts written by ``tabicl_f1_benchmark.py``
(``figures/<species>/tabicl_f1_spatial.json``) and reports, per species and per
fold, ROC-AUC and F1 for both models plus the paired difference over folds.
Nothing is re-fitted here; if a species is missing, run:

    python tabicl_f1_benchmark.py --species "<name>"

Two numbers that are easy to misread, restated at the bottom of the output:
  * test subsamples are class-balanced, so the trivial all-HIGH classifier
    scores F1 ~ 0.667 — that, not 0, is the floor to judge F1 against;
  * with 5 spatial blocks a t-CI has 4 df, so it is indicative, not decisive,
    and can run past a bounded metric's maximum.

Writes a markdown table (``--out``) and the same numbers as JSON
(``--out-json``) so the paper and any downstream figure read one file rather
than re-deriving means from the per-fold artefacts.

CLI:
    python tabicl_v2_report.py
    python tabicl_v2_report.py --species "Caretta caretta" "Cetorhinus maximus"
    python tabicl_v2_report.py --out results/tabicl_v2.md --out-json results/tabicl_v2.json
    python tabicl_v2_report.py --no-write          # print only
"""

import argparse
import json
import os

import numpy as np

FIG_ROOT = "figures"
SPECIES = ["Caretta caretta", "Cetorhinus maximus", "Balaenoptera musculus"]
MODELS = [("tabicl", "TabICLv2"), ("rf_full", "RF-full")]
OUT_MD = os.path.join(FIG_ROOT, "tabicl_v2_spatial_report.md")
OUT_JSON = os.path.join(FIG_ROOT, "tabicl_v2_spatial_report.json")

NOTES = (
    "Test subsamples are class-balanced, so the trivial all-HIGH baseline is "
    "F1 ~ 0.667 (not 0). With 5 spatial blocks a t-CI has 4 df -- indicative "
    "only, and it can extend past a bounded metric's maximum."
)


def load(species, fig_root=FIG_ROOT, scheme="spatial"):
    path = os.path.join(fig_root, species.replace(" ", "_"),
                        f"tabicl_f1_{scheme}.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as fh:
        return json.load(fh), path


def paired(a, b):
    """Mean paired difference over folds with a 95% t-CI and a t-test p."""
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    sem = d.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * sem
    p = float(stats.ttest_rel(a, b).pvalue)
    return float(d.mean()), float(half), p, int((d > 0).sum()), n


def collect(species, d, path):
    """Everything the console, markdown and JSON views all need, derived once."""
    folds = d["folds"]
    rec = {
        "species": species,
        "source": path,
        "architecture": (d.get("checkpoint") or {}).get("architecture", "unrecorded"),
        "checkpoint": d.get("checkpoint"),
        "scheme": d.get("scheme", "spatial"),
        "n_folds": len(folds),
        "n_cells": d.get("n_cells"),
        "prevalence": d.get("prevalence"),
        "n_blocks": d.get("n_blocks"),
        "n_context_per_fold": folds[0]["n_context"],
        "n_test_per_fold": folds[0]["n_test"],
        "threshold": d.get("threshold"),
        "n_estimators": d.get("n_estimators"),
        "reproduction_check": d.get("reproduction_check_vs_tabicl_results_json") or {},
        "folds": [{"fold": f["fold"],
                   "auc_tabicl": f["auc_tabicl"], "auc_rf_full": f["auc_rf_full"],
                   "f1_tabicl": f["f1_tabicl"], "f1_rf_full": f["f1_rf_full"]}
                  for f in folds],
        "metrics": {},
    }
    for metric in ("auc", "f1"):
        vals = {key: np.array([f[f"{metric}_{key}"] for f in folds])
                for key, _ in MODELS}
        m, h, p, wins, n = paired(vals["tabicl"], vals["rf_full"])
        rec["metrics"][metric] = {
            key: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                  "worst_fold": float(v.min()), "best_fold": float(v.max()),
                  "range": float(v.max() - v.min())}
            for key, v in vals.items()
        }
        rec["metrics"][metric]["delta"] = {
            "mean": m, "half": h, "lo": m - h, "hi": m + h, "p": p,
            "wins": wins, "n": n, "significant": bool((m - h) * (m + h) > 0),
        }
    return rec


def _console(recs, missing, emit):
    for r in recs:
        emit(f"\n{'=' * 78}\n{r['species']}  —  spatial-block CV, "
             f"{r['n_folds']} held-out regions  (TabICL {r['architecture']})")
        if r["n_cells"] is not None:
            emit(f"  {r['n_cells']} ocean cells, prevalence {r['prevalence']:.3f}, "
                 f"{r['n_blocks']} spatial blocks, "
                 f"{r['n_context_per_fold']} context rows/fold, "
                 f"{r['n_test_per_fold']} balanced test cells/fold")
        if r["reproduction_check"]:
            emit("  reproduction check vs tabicl_results.json: "
                 + ", ".join(f"{k}={v:g}" for k, v in r["reproduction_check"].items()))
        emit("=" * 78)
        emit(f"  {'fold':>4}  {'AUC TabICLv2':>13}{'AUC RF-full':>13}"
             f"{'F1 TabICLv2':>13}{'F1 RF-full':>12}")
        for f in r["folds"]:
            emit(f"  {f['fold']:>4}  {f['auc_tabicl']:>13.4f}{f['auc_rf_full']:>13.4f}"
                 f"{f['f1_tabicl']:>13.4f}{f['f1_rf_full']:>12.4f}")
        for metric, label in (("auc", "ROC-AUC"), ("f1", "F1")):
            mm = r["metrics"][metric]
            emit(f"\n  {label}")
            for key, name in MODELS:
                s = mm[key]
                emit(f"    {name:<10} mean {s['mean']:.4f}   sd {s['sd']:.4f}"
                     f"   worst fold {s['worst_fold']:.4f}   range {s['range']:.4f}")
            d = mm["delta"]
            emit(f"    paired delta (TabICLv2 - RF-full): {d['mean']:+.4f}  "
                 f"95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p']:.4f}  "
                 f"wins {d['wins']}/{d['n']}  -> "
                 f"{'differs' if d['significant'] else 'no significant difference'}")
    if missing:
        emit("\nnot yet run:")
        for species, path in missing:
            emit(f"  {species}  (expected {path})\n"
                 f'    python tabicl_f1_benchmark.py --species "{species}"')
    emit(f"\nReading notes: {NOTES}")


def _markdown(recs, missing):
    L = ["# TabICLv2 vs RandomForest (full data) — spatial-block 5-fold CV", ""]
    L += ["Positive class: HIGH dwelling. Decision threshold P(HIGH) = 0.5, no "
          "per-fold tuning. TabICL is not fitted: the whole training fold is "
          "passed as in-context data in a single forward pass.", ""]

    L += ["## Summary", "",
          "| species | model | AUC mean | AUC sd | AUC worst | F1 mean | F1 sd | F1 worst |",
          "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        for key, name in MODELS:
            a, f = r["metrics"]["auc"][key], r["metrics"]["f1"][key]
            L.append(f"| {r['species']} | {name} | {a['mean']:.4f} | {a['sd']:.4f} | "
                     f"{a['worst_fold']:.4f} | {f['mean']:.4f} | {f['sd']:.4f} | "
                     f"{f['worst_fold']:.4f} |")
    L += ["", "## Paired difference (TabICLv2 − RF-full), over folds", "",
          "| species | metric | Δ mean | 95% CI | p | wins |", "|---|---|---|---|---|---|"]
    for r in recs:
        for metric, label in (("auc", "ROC-AUC"), ("f1", "F1")):
            d = r["metrics"][metric]["delta"]
            L.append(f"| {r['species']} | {label} | {d['mean']:+.4f} | "
                     f"[{d['lo']:+.4f}, {d['hi']:+.4f}] | {d['p']:.4f} | "
                     f"{d['wins']}/{d['n']} |")

    for r in recs:
        L += ["", f"## {r['species']} — per fold", ""]
        L.append(f"TabICL weights: {r['architecture']}. "
                 f"{r['n_cells']} ocean cells, prevalence {r['prevalence']:.3f}, "
                 f"{r['n_blocks']} spatial blocks, {r['n_context_per_fold']} context "
                 f"rows/fold, {r['n_test_per_fold']} balanced test cells/fold."
                 if r["n_cells"] is not None else
                 f"TabICL weights: {r['architecture']}.")
        if r["reproduction_check"]:
            L.append("")
            L.append("Reproduction check vs `tabicl_results.json`: "
                     + ", ".join(f"`{k}={v:g}`" for k, v in r["reproduction_check"].items()))
        L += ["", "| fold | AUC TabICLv2 | AUC RF-full | F1 TabICLv2 | F1 RF-full |",
              "|---|---|---|---|---|"]
        for f in r["folds"]:
            L.append(f"| {f['fold']} | {f['auc_tabicl']:.4f} | {f['auc_rf_full']:.4f} | "
                     f"{f['f1_tabicl']:.4f} | {f['f1_rf_full']:.4f} |")
        L.append(f"| **mean** | **{r['metrics']['auc']['tabicl']['mean']:.4f}** | "
                 f"**{r['metrics']['auc']['rf_full']['mean']:.4f}** | "
                 f"**{r['metrics']['f1']['tabicl']['mean']:.4f}** | "
                 f"**{r['metrics']['f1']['rf_full']['mean']:.4f}** |")
        L.append(f"| source | `{r['source']}` | | | |")

    if missing:
        L += ["", "## Not yet run", ""]
        for species, path in missing:
            L.append(f"- **{species}** — expected `{path}`; "
                     f'run `python tabicl_f1_benchmark.py --species "{species}"`')
    L += ["", "## Reading notes", "", NOTES, ""]
    return "\n".join(L)


def report(species_list, fig_root=FIG_ROOT, out_md=OUT_MD, out_json=OUT_JSON):
    recs, missing = [], []
    for species in species_list:
        d, path = load(species, fig_root)
        if d is None:
            missing.append((species, path))
        else:
            recs.append(collect(species, d, path))

    _console(recs, missing, print)

    if not recs:
        print("\nnothing to write: no species have been run yet.")
        return 1

    for path, payload in ((out_md, _markdown(recs, missing)),
                          (out_json, json.dumps(
                              {"scheme": "spatial", "n_folds_requested": 5,
                               "models": [n for _, n in MODELS],
                               "reading_notes": NOTES,
                               "species": recs,
                               "not_run": [s for s, _ in missing]},
                              indent=2))):
        if not path:
            continue
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(payload)
        print(f"\n[saved] {path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", nargs="+", default=SPECIES)
    ap.add_argument("--fig-root", default=FIG_ROOT)
    ap.add_argument("--out", default=OUT_MD,
                    help="markdown table destination (default: %(default)s)")
    ap.add_argument("--out-json", default=OUT_JSON,
                    help="machine-readable destination (default: %(default)s)")
    ap.add_argument("--no-write", action="store_true",
                    help="print to stdout only, write nothing")
    args = ap.parse_args(argv)
    return report(args.species, args.fig_root,
                  out_md=None if args.no_write else args.out,
                  out_json=None if args.no_write else args.out_json)


if __name__ == "__main__":
    raise SystemExit(main())
