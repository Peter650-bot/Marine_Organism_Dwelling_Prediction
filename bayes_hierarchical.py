#!/usr/bin/env python3
"""
bayes_hierarchical.py -- Bayesian hierarchical correlated t-test (Corani,
Benavoli et al., Machine Learning 2017) for TabICLv2 vs nested-tuned RF across
the three species, from the per-region scores in `figures/<species>/results.json`.

Why: the sign-flip / Wilcoxon tests treat the 12 held-out regions as
independent. They are not: folds of one species share ~75% of their training
data, and regions are nested within species. This model handles both -- fold
correlation via a compound-symmetric covariance with rho = 1/k, and species as
the upper level of a hierarchy.

This is baycomp 1.0.3's `HierarchicalTest` (same Stan model, priors, bounds,
rope scaling and argmax summary) with ONE change: baycomp needs the same number
of folds for every dataset, but here 4 / 5 / 3 regions are scoreable. The
likelihood is therefore written per species with its own fold count, using the
closed-form inverse and determinant of the compound-symmetric matrix. rho stays
1/5 for every species because each used 5-region CV.

Output: P(TabICL better), P(practically equivalent, |diff| < rope),
P(RF better), for the difference on a new species/dataset.
"""
import json, os

import numpy as np
from scipy import stats

FIG_ROOT = "figures"
SPECIES = ["Caretta_caretta", "Cetorhinus_maximus", "Balaenoptera_musculus"]
K_FOLDS = 5
ROPE = 0.01
SEED = 1
CHAINS, NSAMPLES = 4, 10000

STAN = """
data {
  int<lower=1> q;
  int<lower=1> Nmax;
  array[q] int<lower=2> N;
  matrix[q, Nmax] x;              // padded with zeros beyond N[i]
  real rho;
  real deltaLow; real deltaHi;
  real std0Low;  real std0Hi;
  real stdLow;   real stdHi;
  real lowerAlpha; real upperAlpha;
  real lowerBeta;  real upperBeta;
}
parameters {
  real<lower=deltaLow, upper=deltaHi> delta0;
  real<lower=std0Low, upper=std0Hi> std0;
  vector[q] delta;
  vector<lower=stdLow, upper=stdHi>[q] sigma;
  real<lower=0> nuMinusOne;
  real<lower=lowerAlpha, upper=upperAlpha> gammaAlpha;
  real<lower=lowerBeta, upper=upperBeta> gammaBeta;
}
transformed parameters {
  real<lower=1> nu = nuMinusOne + 1;
}
model {
  nuMinusOne ~ gamma(gammaAlpha, gammaBeta);
  delta ~ student_t(nu, delta0, std0);
  for (i in 1:q) {
    int n = N[i];
    row_vector[n] d = x[i, 1:n] - delta[i];
    // M = (1-rho) I + rho 11'; closed-form inverse quadratic form and det
    real quad = (dot_self(d) - rho / (1 + (n - 1) * rho) * square(sum(d))) / (1 - rho);
    real logdetM = (n - 1) * log1m(rho) + log1p((n - 1) * rho);
    target += -0.5 * (2 * n * log(sigma[i]) + logdetM) - 0.5 * n * log(6.283)
              - 0.5 * quad / square(sigma[i]);
  }
}
"""


def load(metric):
    rows = []
    for s in SPECIES:
        F = json.load(open(os.path.join(FIG_ROOT, s, "results.json")))["folds"]
        rows.append(np.array([f[f"{metric}_tabicl"] - f[f"{metric}_rf_tuned"] for f in F]))
    return rows


def run(metric):
    import stan
    diffs = load(metric)
    # baycomp scaling: divide by the mean within-dataset std (rope scaled too)
    std_diff = np.mean([np.std(d) for d in diffs])
    diffs = [d / std_diff for d in diffs]
    rope = ROPE / std_diff
    N = [len(d) for d in diffs]
    Nmax = max(N)
    x = np.zeros((len(diffs), Nmax))
    for i, d in enumerate(diffs):
        x[i, :len(d)] = d
    std_within = np.mean([np.std(d) for d in diffs])
    std_among = np.std([d.mean() for d in diffs])
    maxdiff = max(np.abs(d).max() for d in diffs)
    data = dict(q=len(diffs), Nmax=Nmax, N=N, x=x, rho=1 / K_FOLDS,
                deltaLow=-maxdiff, deltaHi=maxdiff,
                std0Low=0, std0Hi=std_among * 1000,
                stdLow=0, stdHi=std_within * 1000,
                lowerAlpha=1, upperAlpha=2, lowerBeta=0.01, upperBeta=0.1)
    fit = stan.build(STAN, data=data, random_seed=SEED).sample(
        num_chains=CHAINS, num_samples=NSAMPLES)
    mu, sd, nu = fit["delta0"][0], fit["std0"][0], fit["nu"][0]
    # per-draw P(new-dataset diff < -rope / in rope / > rope), then argmax
    p_rf = stats.t.cdf(-rope, nu, mu, sd)
    p_tab = 1 - stats.t.cdf(rope, nu, mu, sd)
    p_eq = 1 - p_rf - p_tab
    win = np.argmax(np.column_stack([p_rf, p_eq, p_tab]), axis=1)
    p = np.bincount(win, minlength=3) / len(win)
    df = fit.to_frame()
    rhat_ok = None
    try:
        import arviz  # optional convergence check
        rhat_ok = float(arviz.rhat(fit)["delta0"])
    except Exception:
        pass
    return {"metric": metric, "n_regions": N, "mean_diff": float(np.mean(np.concatenate(load(metric)))),
            "delta0_mean": float(mu.mean() * std_diff),
            "delta0_95": [float(np.percentile(mu, 2.5) * std_diff),
                          float(np.percentile(mu, 97.5) * std_diff)],
            "p_tabicl_better": float(p[2]), "p_equivalent": float(p[1]),
            "p_rf_better": float(p[0]), "rhat_delta0": rhat_ok,
            "divergent": int(df["divergent__"].sum()) if "divergent__" in df else None}


if __name__ == "__main__":
    out = [run(m) for m in ("auc", "f1")]
    json.dump(out, open(os.path.join(FIG_ROOT, "bayes_hierarchical.json"), "w"), indent=2)
    for r in out:
        print(json.dumps(r))
