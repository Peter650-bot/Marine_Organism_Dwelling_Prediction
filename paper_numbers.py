#!/usr/bin/env python3
"""
paper_numbers.py -- every quantity the paper reports, recomputed against the
NESTED-TUNED Random Forest. Prints a labelled dump; writes nothing.

Verifies itself against the published untuned values first, so a mismatch in the
reproduction shows up before any tuned number is trusted.
"""
import json, os
import numpy as np
from scipy import stats

FIG = "figures"
SP = [("Caretta caretta", "Caretta_caretta"),
      ("Cetorhinus maximus", "Cetorhinus_maximus"),
      ("Balaenoptera musculus", "Balaenoptera_musculus")]
FEATS = [("latitude", "Latitude"), ("longitude", "Longitude"),
         ("dist_to_coast", "Distance to coast"), ("mean_speed", "Speed"),
         ("retention", "Retention"), ("vorticity", "Vorticity")]


def v(d, k):
    return np.array([f[k] for f in d["folds"]], float)


def ci(a, b):
    """95% t-interval on the paired difference a-b."""
    d = a - b
    n = len(d)
    if n < 2:
        return d.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - h, d.mean() + h


def block(d, tab, rf):
    t, r = v(d, tab), v(d, rf)
    m, lo, hi = ci(t, r)
    return dict(t_mean=t.mean(), t_sd=t.std(ddof=1), t_worst=t.min(),
                t_range=t.max() - t.min(),
                r_mean=r.mean(), r_sd=r.std(ddof=1), r_worst=r.min(),
                r_range=r.max() - r.min(),
                delta=m, lo=lo, hi=hi,
                won=int((t > r).sum()), n=len(t),
                var_ratio=(r.std(ddof=1) ** 2) / (t.std(ddof=1) ** 2))


for name, slug in SP:
    d = json.load(open(os.path.join(FIG, slug, "results.json")))
    print(f"\n{'='*72}\n{name}   (n_folds={len(d['folds'])})\n{'='*72}")
    for metric in ("auc", "f1"):
        old = block(d, f"{metric}_tabicl", f"{metric}_rf_full")
        new = block(d, f"{metric}_tabicl", f"{metric}_rf_tuned")
        print(f"\n-- {metric.upper()}")
        print(f"   TabICLv2     mean {new['t_mean']:.4f}  sd {new['t_sd']:.4f}  "
              f"worst {new['t_worst']:.4f}  range {new['t_range']:.4f}")
        print(f"   RF UNTUNED   mean {old['r_mean']:.4f}  sd {old['r_sd']:.4f}  "
              f"worst {old['r_worst']:.4f}  range {old['r_range']:.4f}")
        print(f"   RF TUNED     mean {new['r_mean']:.4f}  sd {new['r_sd']:.4f}  "
              f"worst {new['r_worst']:.4f}  range {new['r_range']:.4f}")
        print(f"   delta vs untuned {old['delta']:+.4f}  CI [{old['lo']:+.3f}, {old['hi']:+.3f}]  won {old['won']}/{old['n']}")
        print(f"   delta vs TUNED   {new['delta']:+.4f}  CI [{new['lo']:+.3f}, {new['hi']:+.3f}]  won {new['won']}/{new['n']}")
        print(f"   var ratio RF_tuned/TabICL {new['var_ratio']:.2f}x   "
              f"(untuned was {old['var_ratio']:.2f}x)")
        print(f"   worst-fold gap TabICL - RF_tuned = {new['t_worst']-new['r_worst']:+.3f}"
              f"   (untuned {old['t_worst']-old['r_worst']:+.3f})")
        print(f"   range ratio RF_tuned/TabICL {new['r_range']/new['t_range']:.1f}x narrower"
              if new['t_range'] > 0 else "")

    ks = d.get("kshot", {}).get("32", {})
    print("\n-- k=32 (pooled over fold x draw, matching the published table)")
    for key in ("tabicl", "tabllm", "rf", "rf_tuned", "gbt", "knn", "logreg"):
        b = ks.get(key)
        if not b:
            print(f"   {key:9s} --"); continue
        a = np.array([r["auc"] for r in b], float)
        print(f"   {key:9s} n={len(a):2d}  AUC {a.mean():.3f} +/- {a.std(ddof=1):.3f}")

    print("\n-- permutation importance (RF-full)")
    for k, lab in FEATS:
        o = np.mean([f["perm_importance"][k] for f in d["folds"]])
        tt = [f.get("perm_importance_tuned", {}).get(k) for f in d["folds"]]
        n = np.mean(tt) if all(x is not None for x in tt) else float("nan")
        print(f"   {lab:20s} untuned {o:+.3f}   TUNED {n:+.3f}")
