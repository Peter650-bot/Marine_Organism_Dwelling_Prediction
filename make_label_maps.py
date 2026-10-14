#!/usr/bin/env python3
"""
make_label_maps.py -- one map per species of the KDE core-area label.

    figures/label_map_<Species>.png / .pdf

Rebuilds the label exactly as `run_pipeline.run_species` does
(`label_cv.kde_density` -> `label_cv.core_area_label`): a weighted Gaussian KDE
of sighting intensity evaluated on the ocean cells, and label 1 for the cells
inside the 50% volume contour -- the smallest set of cells holding 50% of the
estimated intensity. Land is the NaN mask of the velocity cube, so no external
coastline data is needed.

Each map shows: label-1 cells (core area), label-0 cells, the 50% isopleth
drawn as a line, and the sightings that fed the KDE sized by their weight.
"""

import os

import numpy as np

import label_cv as lcv
import retention_core as rc
import tabllm_prompt as tp
from make_figures import COMMON, INK, INK_2, INK_3, SURFACE
from run_pipeline import RUNS

FIG_ROOT = "figures"
CORE, OUTSIDE, LAND = "#eb6834", "#dbe6ef", "#bdbbb4"


def species_label_grid(species):
    vel_nc, csv, basin = RUNS[species]
    cells = tp.load_cells(species, vel_nc, csv)
    lats, lons, vel = cells["lats"], cells["lons"], cells["vel"]
    df = rc.load_sightings(csv, species, vel)
    dens = lcv.kde_density(df["longitude"].values, df["latitude"].values,
                           df["weighted_records"].values, lons, lats)
    y = lcv.core_area_label(dens)

    Lon, Lat = cells["fields"]["Lon"], cells["fields"]["Lat"]
    ilat = np.searchsorted(vel["lat"], lats)
    ilon = np.searchsorted(vel["lon"], lons)
    D = np.full(Lon.shape, np.nan)
    Y = np.full(Lon.shape, np.nan)
    D[ilat, ilon] = dens
    Y[ilat, ilon] = y
    return {"basin": basin, "Lon": Lon, "Lat": Lat, "D": D, "Y": Y,
            "land": np.isnan(vel["u"][0]), "iso": dens[y == 1].min(),
            "df": df, "y": y, "lon": vel["lon"], "lat": vel["lat"]}


def plot(species, g, fig_root=FIG_ROOT, legend=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lon, lat = g["lon"], g["lat"]
    fig, ax = plt.subplots(figsize=(7.2, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    dlon, dlat = np.diff(lon).mean() / 2, np.diff(lat).mean() / 2
    ext = [lon.min() - dlon, lon.max() + dlon, lat.min() - dlat, lat.max() + dlat]

    ax.imshow(np.where(g["land"], 1.0, np.nan), origin="lower", extent=ext,
              cmap=ListedColormap([LAND]), interpolation="nearest", zorder=1)
    ax.imshow(g["Y"], origin="lower", extent=ext, vmin=0, vmax=1,
              cmap=ListedColormap([OUTSIDE, CORE]), interpolation="nearest", zorder=2)
    ax.contour(g["Lon"], g["Lat"], np.nan_to_num(g["D"], nan=0.0), levels=[g["iso"]],
               colors=INK, linewidths=1.3, zorder=3)

    df = g["df"]
    w = df["weighted_records"].values
    s = 6 + 60 * np.sqrt(w / w.max()) if w.max() > 0 else 8
    ax.scatter(df["longitude"], df["latitude"], s=s, facecolor="none",
               edgecolor=INK_2, linewidth=0.6, alpha=0.8, zorder=4)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.set_xlabel("Longitude (°)", color=INK_2)
    ax.set_ylabel("Latitude (°)", color=INK_2)
    ax.tick_params(colors=INK_3)
    for sp in ax.spines.values():
        sp.set_color(INK_3)

    y = g["y"]
    name = species.replace(" ", r"\ ")
    ax.set_title(f"{COMMON[species]} (${name}$) — {g['basin']}",
                 loc="left", color=INK, fontsize=11)
    if legend:
        handles = [
            Patch(facecolor=CORE, label=f"Label 1: inside 50% contour "
                                        f"({y.sum():,} cells, {100 * y.mean():.1f}%)"),
            Patch(facecolor=OUTSIDE, label=f"Label 0: outside ({(y == 0).sum():,} cells)"),
            Line2D([], [], color=INK, lw=1.3, label="50% volume contour of weighted KDE"),
            Line2D([], [], marker="o", ls="", mfc="none", mec=INK_2,
                   label=f"Sightings, sized by weight (n={len(df):,})"),
            Patch(facecolor=LAND, label="Land"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=2, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()

    base = os.path.join(fig_root, f"label_map_{species.replace(' ', '_')}")
    for ext_ in ("png", "pdf"):
        fig.savefig(f"{base}.{ext_}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return base


def main():
    for sp in RUNS:
        g = species_label_grid(sp)
        print(f"{sp}: {plot(sp, g)}.png  prevalence={g['y'].mean():.3f}")


if __name__ == "__main__":
    main()
