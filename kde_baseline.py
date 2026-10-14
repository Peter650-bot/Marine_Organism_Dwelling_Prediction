#!/usr/bin/env python3
"""
kde_baseline.py -- geography-only baselines: can lon/lat alone, smoothed by a
KDE, predict the held-out region's core-area label? If they score like
TabICLv2, the model may only be interpolating the label surface.

Two baselines, both scored on the SAME folds and 300-cell balanced test sets
as run_pipeline.py (replayed with the same seed, checked against
results.json), with nothing from the held-out region used for fitting:

  kde_sightings   the label recipe itself (label_cv.kde_density: weighted
                  Gaussian KDE, Scott bandwidth) fit only on the sightings
                  whose nearest ocean cell lies in a training region; the
                  score is the density at the test cell.
  kde_classifier  a KDE Bayes classifier on the training cells' labels:
                  score = log p(lon, lat | y=1) - log p(lon, lat | y=0),
                  Gaussian KDEs with Scott bandwidth. Uses exactly the
                  lon/lat + label information the models see.

Output: figures/kde_baseline.json and figures/kde_baseline.png/.pdf, and a
table with TabICLv2 / RF from results.json on the same test cells.
"""

import json
import os

import numpy as np

FIG_ROOT = "figures"
SEED, N_TEST = 42, 300
MODELS = [("auc_tabicl", "TabICLv2"), ("auc_rf_tuned", "RF tuned"),
          ("auc_rf_full", "RF fixed"), ("kde_sightings", "KDE sightings"),
          ("kde_classifier", "KDE classifier")]


def log_kde(pts_lon, pts_lat, lons, lats):
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(np.vstack([pts_lon, pts_lat]))
    return kde.logpdf(np.vstack([lons, lats]))


def run_species(species, vel_nc, csv):
    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score
    import label_cv as lcv
    import retention_core as rc
    import tabllm_prompt as tp

    with open(os.path.join(FIG_ROOT, species.replace(" ", "_"), "results.json")) as fh:
        recorded = {f["fold"]: f for f in json.load(fh)["folds"]}

    cells = tp.load_cells(species, vel_nc, csv)
    lats, lons, vel = cells["lats"], cells["lons"], cells["vel"]
    df = rc.load_sightings(csv, species, vel)
    s_lon, s_lat = df["longitude"].values, df["latitude"].values
    s_w = df["weighted_records"].values
    y = lcv.core_area_label(lcv.kde_density(s_lon, s_lat, s_w, lons, lats))
    splits, region = lcv.loro_kmeans_folds(lats, lons)
    # region of each sighting = region of its nearest ocean cell
    _, nn = cKDTree(np.column_stack([lons, lats])).query(np.column_stack([s_lon, s_lat]))
    s_region = region[nn]

    rng = np.random.default_rng(SEED)
    folds = []
    for i, (tr, te) in enumerate(splits):
        kept = lcv.balanced_subsample(te, y, N_TEST, rng)
        if kept is None or len(np.unique(y[tr])) < 2:
            continue
        rec = recorded[i]
        assert len(kept) == rec["n_test"] and int(y[kept].sum()) == rec["n_pos_test"], \
            f"{species} fold {i}: test-set replay mismatch"

        in_train = s_region != i
        d = lcv.kde_density(s_lon[in_train], s_lat[in_train], s_w[in_train],
                            lons[kept], lats[kept])
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        ratio = (log_kde(lons[pos], lats[pos], lons[kept], lats[kept])
                 - log_kde(lons[neg], lats[neg], lons[kept], lats[kept]))
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=1e9, neginf=-1e9)

        out = {"fold": i, "n_test": int(len(kept)),
               "n_sightings_train": int(in_train.sum()),
               "n_sightings_heldout": int((~in_train).sum()),
               "kde_sightings": float(roc_auc_score(y[kept], d)),
               "kde_classifier": float(roc_auc_score(y[kept], ratio))}
        for k in ("auc_tabicl", "auc_rf_tuned", "auc_rf_full"):
            out[k] = rec[k]
        folds.append(out)
    return folds


def t_interval(d):
    from scipy import stats
    d = np.asarray(d)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def report(res):
    for sp, folds in res.items():
        print(f"\n{sp}  ({len(folds)} scored folds)")
        print(f"  {'fold':>4s} " + " ".join(f"{n:>14s}" for _, n in MODELS)
              + "   sightings train/held-out")
        for f in folds:
            print(f"  {f['fold']:4d} " + " ".join(f"{f[k]:14.4f}" for k, _ in MODELS)
                  + f"   {f['n_sightings_train']}/{f['n_sightings_heldout']}")
        print(f"  {'mean':>4s} " + " ".join(
            f"{np.mean([f[k] for f in folds]):14.4f}" for k, _ in MODELS))
        for base in ("kde_sightings", "kde_classifier"):
            for k, n in MODELS[:3]:
                d = [f[k] - f[base] for f in folds]
                m, lo, hi = t_interval(d)
                wins = sum(x > 0 for x in d)
                print(f"    {n:9s} - {base:14s} {m:+.4f}  [{lo:+.3f}, {hi:+.3f}]  "
                      f"wins {wins}/{len(d)}")


def plot(res, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_figures import COMMON, INK, INK_2, INK_3, SURFACE
    colours = {"auc_tabicl": "#2a78d6", "auc_rf_tuned": "#eb6834",
               "auc_rf_full": "#f2a07a", "kde_sightings": INK_2,
               "kde_classifier": INK_3}
    fig, axes = plt.subplots(1, len(res), figsize=(3.3 * len(res), 3.9),
                             sharey=True, facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for ax, (sp, folds) in zip(np.atleast_1d(axes), res.items()):
        for j, (k, n) in enumerate(MODELS):
            v = np.array([f[k] for f in folds])
            ax.scatter(j + rng.uniform(-0.08, 0.08, v.size), v, s=18,
                       facecolor="none", edgecolor=colours[k], lw=1.0, zorder=3)
            ax.hlines(v.mean(), j - 0.28, j + 0.28, color=colours[k], lw=2.4, zorder=4)
            ax.annotate(f"{v.mean():.3f}", (j, v.mean()), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK)
        ax.axhline(0.5, color=INK_3, lw=0.8, ls=":")
        ax.axvline(2.5, color="#e4e3df", lw=1.0)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([n.replace(" ", "\n") for _, n in MODELS], fontsize=7.8)
        ax.set_title(f"{COMMON[sp]} ({len(folds)} folds)", fontsize=9.5,
                     color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#e4e3df")
    np.atleast_1d(axes)[0].set_ylabel("ROC-AUC on held-out region\n"
                                      "(bar = mean, circles = folds)",
                                      fontsize=9.5, color=INK_2)
    fig.suptitle("Full models (left of divider) vs lon/lat-only KDE baselines (right)",
                 fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_ROOT, f"kde_baseline.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {path}")
    plt.close(fig)


def main():
    from run_pipeline import RUNS
    res = {sp: run_species(sp, vel_nc, csv) for sp, (vel_nc, csv, _) in RUNS.items()}
    with open(os.path.join(FIG_ROOT, "kde_baseline.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    report(res)
    plot(res)


if __name__ == "__main__":
    main()
