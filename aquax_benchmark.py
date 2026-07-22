#!/usr/bin/env python3
"""
aquax_benchmark.py — external SDM benchmark for the retention + TabLLM
predictions, using AquaX / AquaMaps habitat-suitability.

AquaX (PLOS One / bioRxiv 2025) is an enhanced AquaMaps framework: a 10-algorithm
SDM ensemble producing 0.05-degree habitat-suitability surfaces. Its openly
downloadable predecessor AquaMaps publishes per-species native-range CSVs
(`Center Lat`, `Center Long`, `Overall Probability` in [0,1]) at 0.5 degrees.
We ship against AquaMaps and keep AquaX behind the same loader (swap `source`).

CRITICAL FRAMING (do not overclaim):
  * AquaMaps native-range maps are built from occurrence databases that OVERLAP
    OBIS-SEAMAP — the same source as this project's sightings — so "AquaMaps
    predicting our y_cls" is partly CIRCULAR. Disclose it.
  * AquaMaps is a fixed 0.5-degree surface upsampled to our ~1/8-degree grid with
    NO per-fold training. So its AUC is computed ONCE over the whole grid and
    reported as a correlated published-SDM REFERENCE — never slotted into the
    trained-CV rows next to RF / TabLLM.

Reuses the velocity grid via the cell lon/lat returned by
tabllm_pipeline.load_cells / retention_core.build_features. Heavy imports
(scipy, sklearn, matplotlib, pandas) are lazy so this module imports and
unit-tests with numpy + stdlib only.
"""

import json
import os

import numpy as np


# Candidate column names across AquaMaps / AquaX CSV exports.
_LAT_COLS = ["Center Lat", "centerlat", "CenterLat", "latitude", "lat",
             "Latitude"]
_LON_COLS = ["Center Long", "centerlong", "CenterLong", "longitude", "lon",
             "Longitude", "Center Lon"]
_SUIT_COLS = ["Overall Probability", "probability", "Probability", "suitability",
              "Suitability", "OverallProbability", "csquarecode_probability"]


def _first_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"none of {candidates} found in columns {list(df.columns)}")


def load_external_suitability(path, source="aquamaps"):
    """Load a per-species habitat-suitability CSV. Returns
    {lon_pts, lat_pts, suit, source} as 1-D float arrays. Tolerant of the column
    naming differences between AquaMaps and AquaX exports."""
    import pandas as pd
    df = pd.read_csv(path)
    latc = _first_col(df, _LAT_COLS)
    lonc = _first_col(df, _LON_COLS)
    suitc = _first_col(df, _SUIT_COLS)
    lat = df[latc].astype(float).to_numpy()
    lon = df[lonc].astype(float).to_numpy()
    suit = df[suitc].astype(float).to_numpy()
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(suit)
    return {"lon_pts": lon[ok], "lat_pts": lat[ok], "suit": suit[ok],
            "source": source}


def regrid_to_cells(ext, lats, lons, lon_conv="pm180"):
    """Interpolate the external suitability surface onto OUR grid cells (the
    per-valid-cell lon/lat from build_features). Linear with nearest-neighbour
    fallback for cells outside the convex hull. Aligns longitude to the cube's
    convention exactly as retention_core.load_sightings."""
    from scipy.interpolate import griddata
    lon_pts = ext["lon_pts"].copy()
    cell_lon = np.asarray(lons, float).copy()
    if lon_conv == "0360":
        lon_pts = lon_pts % 360
        cell_lon = cell_lon % 360
    else:
        lon_pts = ((lon_pts + 180) % 360) - 180
        cell_lon = ((cell_lon + 180) % 360) - 180
    pts = np.column_stack([lon_pts, ext["lat_pts"]])
    targ = np.column_stack([cell_lon, np.asarray(lats, float)])
    suit = griddata(pts, ext["suit"], targ, method="linear")
    miss = ~np.isfinite(suit)
    if miss.any():
        suit[miss] = griddata(pts, ext["suit"], targ[miss], method="nearest")
    return np.clip(suit, 0.0, 1.0)


def _topq_overlap(a, b):
    """Jaccard + Cohen's kappa of the two top-quartile masks."""
    am = a >= np.nanpercentile(a, 75)
    bm = b >= np.nanpercentile(b, 75)
    inter = np.sum(am & bm)
    union = np.sum(am | bm)
    jacc = float(inter / union) if union else float("nan")
    # Cohen's kappa on the 2x2 agreement of the binary top-Q masks
    po = np.mean(am == bm)
    pa1 = np.mean(am); pb1 = np.mean(bm)
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = float((po - pe) / (1 - pe)) if (1 - pe) else float("nan")
    return {"jaccard_topq": jacc, "kappa_topq": kappa}


def aquax_agreement(suit, retention, y_cls, p_high=None, seed=42,
                    n_perm=1000, n_boot=1000):
    """Agreement metrics on the shared valid cells. AquaMaps AUC is computed ONCE
    over the whole grid (it is a fixed untrained surface) and labelled as a
    correlated reference, NOT a trained-CV competitor."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    suit = np.asarray(suit, float)
    retention = np.asarray(retention, float)
    y_cls = np.asarray(y_cls, int)

    def _spearman_with_ci(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if len(a) < 3:
            return {"rho": None, "perm_p": None, "ci": [None, None], "n": int(len(a))}
        rho, _ = spearmanr(a, b)
        perm = np.array([spearmanr(a, rng.permutation(b))[0] for _ in range(n_perm)])
        perm_p = float(np.mean(np.abs(perm) >= abs(rho)))
        boot = np.array([
            spearmanr(*(lambda i: (a[i], b[i]))(rng.integers(0, len(a), len(a))))[0]
            for _ in range(n_boot)])
        ci = np.nanpercentile(boot, [2.5, 97.5]).tolist()
        return {"rho": float(rho), "perm_p": perm_p, "ci": ci, "n": int(len(a))}

    out = {
        "source_note": ("AquaMaps shares OBIS-SEAMAP occurrences with this study "
                        "(partly circular) and is a fixed 0.5-deg surface upsampled "
                        "with no per-fold training; AUC is a whole-grid reference, "
                        "not a trained-CV competitor."),
        "spearman_suit_retention": _spearman_with_ci(suit, retention),
    }
    if p_high is not None:
        out["spearman_suit_phigh"] = _spearman_with_ci(suit, np.asarray(p_high, float))
    if len(np.unique(y_cls)) == 2:
        out["auc_suit_vs_ycls_wholegrid"] = float(roc_auc_score(y_cls, suit))
    out.update({"topq_suit_vs_retention": _topq_overlap(suit, retention)})
    if p_high is not None:
        out["topq_suit_vs_phigh"] = _topq_overlap(suit, np.asarray(p_high, float))
    return out


def make_fig23(out_dir, species, lats, lons, suit, retention, y_cls,
               p_high=None, metrics=None):
    """Figure 23: suitability / retention / p_high maps (scatter, no 2-D mask
    needed) + suit-vs-p_high scatter + top-Q overlap bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lats = np.asarray(lats); lons = np.asarray(lons)
    have_phi = p_high is not None
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    def _map(a, x, title, cmap):
        sc = ax[a].scatter(lons, lats, c=x, s=4, cmap=cmap)
        ax[a].set_title(title); plt.colorbar(sc, ax=ax[a])

    _map((0, 0), suit, f"AquaMaps suitability — {species}", "viridis")
    _map((0, 1), retention, "Retention", "YlOrRd")
    if have_phi:
        _map((0, 2), np.asarray(p_high), "TabLLM P(high)", "magma")
    else:
        ax[0, 2].axis("off")

    if have_phi:
        ax[1, 0].scatter(suit, p_high, s=4, alpha=0.4)
        ax[1, 0].set_xlabel("AquaMaps suitability"); ax[1, 0].set_ylabel("TabLLM P(high)")
        rho = (metrics or {}).get("spearman_suit_phigh", {}).get("rho")
        ax[1, 0].set_title(f"suit vs P(high)  rho={rho}")
    else:
        ax[1, 0].scatter(suit, retention, s=4, alpha=0.4)
        ax[1, 0].set_xlabel("suitability"); ax[1, 0].set_ylabel("retention")
        rho = (metrics or {}).get("spearman_suit_retention", {}).get("rho")
        ax[1, 0].set_title(f"suit vs retention  rho={rho}")

    # AUC reference (whole-grid) text
    ax[1, 1].axis("off")
    if metrics:
        auc = metrics.get("auc_suit_vs_ycls_wholegrid")
        ax[1, 1].text(0.05, 0.6,
                      f"AquaMaps suitability AUC vs y_cls\n(whole-grid reference): "
                      f"{auc}\n\n{metrics.get('source_note', '')}",
                      wrap=True, fontsize=9, va="top")

    # top-Q overlap bars
    bars, vals = [], []
    if metrics:
        for key, lab in [("topq_suit_vs_retention", "suit∩retention"),
                         ("topq_suit_vs_phigh", "suit∩P(high)")]:
            if key in metrics:
                bars.append(lab + "\nJaccard"); vals.append(metrics[key]["jaccard_topq"])
                bars.append(lab + "\nkappa"); vals.append(metrics[key]["kappa_topq"])
    if bars:
        ax[1, 2].bar(range(len(bars)), vals, color="teal")
        ax[1, 2].set_xticks(range(len(bars)))
        ax[1, 2].set_xticklabels(bars, fontsize=7)
        ax[1, 2].set_title("top-quartile overlap")
    else:
        ax[1, 2].axis("off")

    fig.tight_layout()
    out = os.path.join(out_dir, "23_aquax_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_benchmark(cells, suitability_csv, source="aquamaps", p_high=None):
    """End-to-end: load external suitability, regrid onto cells, compute
    agreement, write 23_aquax_metrics.json + figure 23."""
    ext = load_external_suitability(suitability_csv, source=source)
    suit = regrid_to_cells(ext, cells["lats"], cells["lons"],
                           cells["vel"]["lon_conv"])
    retention = cells["X"][:, 0]  # retention is FEATURES[0]
    metrics = aquax_agreement(suit, retention, cells["y_cls"], p_high=p_high)
    out_dir = cells["out_dir"]
    with open(os.path.join(out_dir, "23_aquax_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    make_fig23(out_dir, cells["species"], cells["lats"], cells["lons"], suit,
               retention, cells["y_cls"], p_high=p_high, metrics=metrics)
    return metrics


def _selftest():
    rng = np.random.default_rng(0)
    # synthetic external surface: suitability ~ gaussian in lon/lat
    glon = rng.uniform(0, 30, 500)
    glat = rng.uniform(30, 45, 500)
    gs = np.exp(-((glon - 15) ** 2 + (glat - 38) ** 2) / 50.0)
    ext = {"lon_pts": glon, "lat_pts": glat, "suit": gs, "source": "test"}
    # our cells
    clat = rng.uniform(31, 44, 200)
    clon = rng.uniform(1, 29, 200)
    suit = regrid_to_cells(ext, clat, clon, "pm180")
    assert suit.shape == (200,) and np.all((suit >= 0) & (suit <= 1))
    # retention correlated with suitability; y_cls from suit
    retention = suit + rng.normal(0, 0.05, 200)
    y = (suit > np.median(suit)).astype(int)
    m = aquax_agreement(suit, retention, y, p_high=suit, n_perm=50, n_boot=50)
    assert m["spearman_suit_retention"]["rho"] > 0.5
    assert m["auc_suit_vs_ycls_wholegrid"] > 0.9  # suit predicts y by construction
    assert 0 <= m["topq_suit_vs_retention"]["jaccard_topq"] <= 1
    # 0360 longitude alignment path
    suit2 = regrid_to_cells(ext, clat, clon, "0360")
    assert suit2.shape == (200,)
    print("aquax_benchmark self-test OK (regrid, agreement, topq, lon-conv)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.error("only --selftest is runnable standalone; "
                 "use run_benchmark(cells, csv) from the pipeline")
