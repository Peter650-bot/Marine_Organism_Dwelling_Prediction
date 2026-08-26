#!/usr/bin/env python3
"""
compare_rf.py -- does the TabICLv2 advantage survive a properly tuned Random
Forest? Paired per-region tests of TabICLv2 against RF before and after nested
tuning, for both AUC and F1.

Reads `figures/<species>/results.json` (needs `tune_rf.py` to have run).
Prints a markdown report; writes nothing.
"""
import json, os
import numpy as np
from scipy import stats

FIG_ROOT = "figures"
SPECIES = [("Caretta caretta", "Caretta_caretta"),
           ("Cetorhinus maximus", "Cetorhinus_maximus"),
           ("Balaenoptera musculus", "Balaenoptera_musculus")]
ALPHA_BONF = 0.05 / 6


def paired(a, b):
    """a = TabICL, b = RF. Positive delta favours TabICL."""
    d = a - b
    n = len(d)
    out = {"n": n, "mean": d.mean(), "median": float(np.median(d)),
           "sd": d.std(ddof=1) if n > 1 else np.nan, "deltas": d}
    out["t_p"] = stats.ttest_rel(a, b).pvalue if n > 1 else np.nan
    nz = d[d != 0]
    out["wins"] = int((d > 0).sum()); out["losses"] = int((d < 0).sum())
    out["ties"] = int((d == 0).sum())
    out["sign_p"] = (stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue
                     if len(nz) else np.nan)
    return out


def main():
    rows, pooled = [], {m: {"t": [], "u": [], "k": []} for m in ("auc", "f1")}
    have_tuned = True
    for name, slug in SPECIES:
        p = os.path.join(FIG_ROOT, slug, "results.json")
        d = json.load(open(p))
        F = d["folds"]
        if not all("auc_rf_tuned" in f for f in F):
            have_tuned = False
            print(f"!! {name}: tuning incomplete"); continue
        for m in ("auc", "f1"):
            t = np.array([f[f"{m}_tabicl"] for f in F], float)
            u = np.array([f[f"{m}_rf_full"] for f in F], float)
            k = np.array([f[f"{m}_rf_tuned"] for f in F], float)
            pooled[m]["t"] += list(t); pooled[m]["u"] += list(u); pooled[m]["k"] += list(k)
            rows.append((name, m, t, u, k))
    if not have_tuned:
        return 1

    for metric in ("auc", "f1"):
        print(f"\n## {metric.upper()}\n")
        print("| species | TabICL | RF untuned | RF tuned | Δ vs untuned | p | "
              "Δ vs tuned | p | W-L-T |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name, m, t, u, k in rows:
            if m != metric: continue
            su, sk = paired(t, u), paired(t, k)
            star = "**" if sk["t_p"] < 0.05 else ""
            print(f"| *{name}* | {t.mean():.4f} | {u.mean():.4f} | {k.mean():.4f} "
                  f"| {su['mean']:+.4f} | {su['t_p']:.4f} "
                  f"| {star}{sk['mean']:+.4f}{star} | {star}{sk['t_p']:.4f}{star} "
                  f"| {sk['wins']}-{sk['losses']}-{sk['ties']} |")
        P = pooled[metric]
        t, u, k = (np.array(P["t"]), np.array(P["u"]), np.array(P["k"]))
        su, sk = paired(t, u), paired(t, k)
        print(f"| **pooled ({len(t)} regions)** | {t.mean():.4f} | {u.mean():.4f} "
              f"| {k.mean():.4f} | {su['mean']:+.4f} | sign {su['sign_p']:.4f} "
              f"| {sk['mean']:+.4f} | sign {sk['sign_p']:.4f} "
              f"| {sk['wins']}-{sk['losses']}-{sk['ties']} |")
        print(f"\nBonferroni alpha (6 tests) = {ALPHA_BONF:.4f}")
        print(f"per-fold deltas vs tuned: "
              f"{[round(x,4) for x in sk['deltas']]}")

    print("\n## RF: tuning effect on the baseline itself\n")
    print("| species | metric | untuned | tuned | change |")
    print("|---|---|---|---|---|")
    for name, m, t, u, k in rows:
        print(f"| *{name}* | {m.upper()} | {u.mean():.4f} | {k.mean():.4f} "
              f"| {k.mean()-u.mean():+.4f} |")

    print("\n## selected hyperparameters\n")
    for name, slug in SPECIES:
        d = json.load(open(os.path.join(FIG_ROOT, slug, "results.json")))
        for f in d["folds"]:
            print(f"  {name:24s} fold {f['fold']} "
                  f"AUC-> {f.get('rf_tuned_params_auc')} "
                  f"F1-> {f.get('rf_tuned_params_f1')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
