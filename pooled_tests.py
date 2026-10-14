#!/usr/bin/env python3
"""
pooled_tests.py -- TabICLv2 vs the nested-CV tuned RF, pooled over every
scoreable held-out region of all three species (4 + 5 + 3 = 12).

Per-species paired tests have no power at n = 3..5 (an exact Wilcoxon on 4
pairs cannot go below p = 0.125). Pooling the 12 region-level differences gives

  sign-flip   exact paired permutation test: all 2^12 sign assignments of the
              differences, two-sided, statistic = mean difference
  Wilcoxon    exact signed-rank test on the same 12 differences

Bonferroni is applied for the two metrics (m = 2). Both tests treat the 12
regions as independent, which they are not (regions of one species share most
of their training data); bayes_hierarchical.py is the dependence-aware check.

Reads figures/<species>/results.json; prints only.
"""
import itertools
import json
import os

import numpy as np
from scipy import stats

FIG_ROOT = "figures"
SPECIES = ["Caretta_caretta", "Cetorhinus_maximus", "Balaenoptera_musculus"]
N_METRICS = 2


def diffs(metric):
    out = []
    for s in SPECIES:
        with open(os.path.join(FIG_ROOT, s, "results.json")) as fh:
            F = json.load(fh)["folds"]
        out += [f[f"{metric}_tabicl"] - f[f"{metric}_rf_tuned"] for f in F]
    return np.array(out)


def sign_flip_p(d):
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


if __name__ == "__main__":
    for m in ("auc", "f1"):
        d = diffs(m)
        p_sf, p_w = sign_flip_p(d), float(stats.wilcoxon(d).pvalue)
        print(f"{m.upper():3s} n={len(d)}  mean diff {d.mean():+.4f}  "
              f"won/lost/tied {(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()}  "
              f"sign-flip p={p_sf:.4f} (x{N_METRICS} = {min(1, N_METRICS * p_sf):.4f})  "
              f"Wilcoxon p={p_w:.4f} (x{N_METRICS} = {min(1, N_METRICS * p_w):.4f})")
