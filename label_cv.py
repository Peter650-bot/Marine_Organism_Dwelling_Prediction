#!/usr/bin/env python3
"""
label_cv.py -- the label definition and the cross-validation scheme.

Kept SEPARATE from retention_core.py on purpose: the canonical Caretta results
(Spearman rho ~= 0.362, figures 01-19) must stay reproducible, so nothing here
touches the published codepath. Everything downstream of this module writes
under `figures/`.

Correction 1 -- the label (`core_area_label`)
---------------------------------------------
The published label was `sightings > median(sightings[sightings > 0])` on a
surface built with `scipy.interpolate.griddata(method="linear")`. Two problems:

  * A linear interpolant is supported only on the CONVEX HULL of the records,
    so "non-zero" is hull membership -- a geometric artefact. Defining presence
    that way is predictable from latitude and longitude alone at AUC ~0.99.
  * Taking the median of the density VALUES selects 50% of the CELLS. A 50%
    utilization distribution contour is the smallest region holding 50% of the
    probability VOLUME, which is a much smaller area because use concentrates.
    The published label was an areal median, not a core area, and calling it a
    UD contour would be wrong.

This module replaces both: a fixed-bandwidth Gaussian kernel density estimate
(reference/Scott bandwidth, the `href` analogue used in home-range work) over
the occurrence points, weighted by `weighted_records`, evaluated on the ocean
cells and normalised to unit mass; then the genuine 50% volume contour -- sort
cells by density, accumulate mass, keep cells up to the 50% isopleth. KDE has
decaying support, so the hard hull edge disappears.

The caveat KDE does NOT fix, and which must stay in the writeup: these are
pooled, effort-biased, opportunistic sightings from many observers, not a
random sample of relocations of one animal. The estimator is now a kernel
density estimate of SIGHTING INTENSITY. It is not a utilization distribution in
Worton's sense, and should not be cited as one.

Correction 2 -- the folds (`loro_kmeans_folds`)
-----------------------------------------------
The published scheme was `GroupKFold(5)` over 3-degree blocks. GroupKFold
balances fold SIZES, so it scatters the ~100 blocks across folds; no block is
split, but every held-out block ends up ringed by training blocks. Measured
separation was ~0.5 degrees -- four grid cells. Latitude and longitude alone
then score 0.88-0.94 under a scheme described as "five held-out regions".

`loro_kmeans_folds` runs k-means on cell coordinates (longitude scaled by
cos(mean latitude)) to get five compact CONTIGUOUS regions, each held out in
turn. Measured separation rises to ~2.4 degrees, p90 5.5-7.5.
"""

import numpy as np

SEED = 42
N_SPLITS = 5
CORE_MASS = 0.50


def kde_density(lon_pts, lat_pts, weights, lons, lats, bw_scale=1.0):
    """Weighted Gaussian KDE of sighting intensity evaluated on the ocean cells.

    Bandwidth is the Scott reference rule (n**(-1/6) in 2-D) on the weighted
    covariance -- the `href` analogue standard in home-range estimation. Returns
    density normalised to sum to 1 over the supplied cells.
    """
    from scipy.stats import gaussian_kde
    pts = np.vstack([np.asarray(lon_pts, float), np.asarray(lat_pts, float)])
    w = np.asarray(weights, float)
    w = np.ones(pts.shape[1]) if not np.isfinite(w).any() or w.sum() <= 0 else w
    kde = gaussian_kde(pts, weights=w)
    kde.set_bandwidth(kde.factor * bw_scale)
    d = kde(np.vstack([np.asarray(lons, float), np.asarray(lats, float)]))
    d = np.clip(np.nan_to_num(d, nan=0.0), 0.0, None)
    return d / d.sum() if d.sum() > 0 else d


def core_area_label(density, mass=CORE_MASS):
    """The genuine `mass` volume contour: sort cells by density, accumulate
    probability mass, and keep the cells up to the isopleth.

    Unlike a median of the density values this selects 50% of the MASS, not 50%
    of the cells, so the positive class is the concentrated-use core and its
    prevalence is far below 0.5.
    """
    d = np.asarray(density, float)
    order = np.argsort(-d)
    cum = np.cumsum(d[order])
    total = cum[-1] if cum.size else 0.0
    if total <= 0:
        return np.zeros(len(d), dtype=int)
    k = int(np.searchsorted(cum, mass * total) + 1)
    y = np.zeros(len(d), dtype=int)
    y[order[:k]] = 1
    return y


def loro_kmeans_folds(lats, lons, n_splits=N_SPLITS, seed=SEED):
    """Five compact contiguous regions, each held out in turn."""
    from sklearn.cluster import KMeans
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    scale = np.cos(np.deg2rad(lats.mean()))
    P = np.column_stack([lats, lons * scale])
    lab = KMeans(n_clusters=n_splits, random_state=seed, n_init=10).fit_predict(P)
    idx = np.arange(len(lats))
    return [(idx[lab != k], idx[lab == k]) for k in range(n_splits)], lab


def separation_deg(coords_test, coords_train, chunk=200):
    """Distance from each held-out cell to the nearest training cell (degrees)."""
    out = []
    for s in range(0, len(coords_test), chunk):
        d = np.sqrt(((coords_test[s:s + chunk, None, :] - coords_train[None, :, :]) ** 2).sum(-1))
        out.append(d.min(1))
    return np.concatenate(out)


def balanced_subsample(pool_idx, y, n_target, rng):
    """Class-balanced test subsample, shrinking gracefully.

    The published runs fixed n=300 (150/150). Under a genuine core-area label
    the positive class is rare and contiguous regions are small, so a held-out
    region may not hold 150 positives. Rather than drop the fold we take
    min(150, n_pos, n_neg) per class and RECORD the size, so the comparison
    stays balanced and every model still sees identical cells.
    """
    pool_idx = np.asarray(pool_idx)
    pos = pool_idx[y[pool_idx] == 1]
    neg = pool_idx[y[pool_idx] == 0]
    per = min(n_target // 2, len(pos), len(neg))
    if per == 0:
        return None
    return np.concatenate([rng.choice(pos, per, replace=False),
                           rng.choice(neg, per, replace=False)])
