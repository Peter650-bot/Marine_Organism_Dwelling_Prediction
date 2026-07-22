#!/usr/bin/env python3
"""
alignment_check.py — spatial/temporal alignment diagnostic for the
Lagrangian-retention SDM tool.

Given a velocity NetCDF (Copernicus ugos/vgos product) and an OBIS-SEAMAP
species CSV, reports whether the occurrence records are actually covered by the
velocity field in space and time, and — where they are not — prints the exact
longitude / latitude / date bounds needed to download matching ugos/vgos data
from Copernicus (DUACS L4).

This operationalises the alignment step that must precede every per-species run.
It performs NO particle simulation, so it is cheap to run.

Usage
-----
    python alignment_check.py \
        --velocity Datasets/Oceanic_data.nc \
        --species-csv "Datasets/obis_seamap_custom_.../...dist_sp_1deg_csv.csv"

    # restrict to one species, and re-window:
    python alignment_check.py --velocity ocean.nc --species-csv sp.csv \
        --species "Cetorhinus maximus" --window-years 2

The time-axis decoding mirrors enhanced_analysis.py (raw h5py + manual CF
decode) so the reported model period matches the pipeline exactly.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import h5py

# Start of the satellite-altimetry / DUACS record: no geostrophic velocity
# product can ever cover occurrence records earlier than this.
ALTIMETRY_START = pd.Timestamp("1993-01-01")
# CMEMS global product recommended for the multi-region extension.
GLOBAL_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D"


def load_velocity_bounds(path):
    """Read lon/lat/time extent and resolution from the velocity NetCDF.

    Mirrors the h5py + manual CF-time decode used in enhanced_analysis.py so the
    reported model period is identical to the pipeline's.
    """
    with h5py.File(path, "r") as h5:
        lon = h5["longitude"][:]
        lat = h5["latitude"][:]
        t_raw = h5["time"][:]
        units = h5["time"].attrs.get("units", "seconds since 1970-01-01 00:00:00")
        if isinstance(units, bytes):
            units = units.decode()

    parts = str(units).split("since")
    unit_word = parts[0].strip().lower()
    origin = pd.Timestamp(parts[-1].strip())
    if unit_word.startswith("second"):
        time_arr = np.array([origin + pd.Timedelta(seconds=float(s)) for s in t_raw])
    else:
        time_arr = np.array([origin + pd.Timedelta(days=float(d)) for d in t_raw])

    return {
        "lon": lon, "lat": lat, "time": pd.to_datetime(time_arr),
        "lon_min": float(lon.min()), "lon_max": float(lon.max()),
        "lat_min": float(lat.min()), "lat_max": float(lat.max()),
        "lon_res": float(abs(lon[1] - lon[0])), "lat_res": float(abs(lat[1] - lat[0])),
        "t_start": pd.Timestamp(time_arr[0]), "t_end": pd.Timestamp(time_arr[-1]),
        "n_lon": len(lon), "n_lat": len(lat), "n_t": len(time_arr),
    }


def load_species(path):
    """Load an OBIS-SEAMAP gridded-summary CSV and derive midpoint dates."""
    df = pd.read_csv(path)
    df["dmin"] = pd.to_datetime(df["date_min"], errors="coerce")
    df["dmax"] = pd.to_datetime(df["date_max"], errors="coerce")
    df["dmid"] = df["dmin"] + (df["dmax"] - df["dmin"]) / 2
    # gridded-summary artefact: cells where several species were collapsed with ';'
    df["is_multi"] = df["species"].astype(str).str.contains(";", na=False)
    return df


def in_box(df, v):
    return df[(df.longitude >= v["lon_min"]) & (df.longitude <= v["lon_max"]) &
              (df.latitude >= v["lat_min"]) & (df.latitude <= v["lat_max"])]


def basin_of(lon, lat):
    """Coarse ocean-basin label for download tiling (matches plan §2)."""
    if -100 <= lon <= 20 and lat >= 0:
        return "N_Atlantic"
    if -70 <= lon <= 20 and lat < 0:
        return "S_Atlantic"
    if 20 < lon <= 147 and lat >= 0:
        return "N_IndoPacific_W"
    if 20 < lon <= 147 and lat < 0:
        return "Indian_SWPacific"
    if lon > 147 or lon < -100:
        return "Pacific_dateline"
    return "other"


def fmt_cmd(dataset_id, lo0, lo1, la0, la1, t0, t1, name):
    return (
        "copernicusmarine subset \\\n"
        f"  --dataset-id {dataset_id} \\\n"
        "  --variable ugos --variable vgos \\\n"
        f"  --minimum-longitude {lo0:.2f} --maximum-longitude {lo1:.2f} \\\n"
        f"  --minimum-latitude {la0:.2f} --maximum-latitude {la1:.2f} \\\n"
        f"  --start-datetime {t0} --end-datetime {t1} \\\n"
        f"  --output-filename ocean_{name}.nc"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--velocity", required=True, help="velocity NetCDF (ugos/vgos)")
    ap.add_argument("--species-csv", required=True, help="OBIS-SEAMAP species CSV")
    ap.add_argument("--species", default=None,
                    help="restrict to one species name (exact match)")
    ap.add_argument("--window-years", type=int, default=2,
                    help="+/- N years around the model period for the 'window' count")
    ap.add_argument("--min-year", type=int, default=ALTIMETRY_START.year,
                    help="earliest downloadable year (default 1993, altimetry era)")
    ap.add_argument("--equator-band", type=float, default=5.0,
                    help="|lat| below which geostrophy is unreliable (default 5)")
    ap.add_argument("--top", type=int, default=20,
                    help="how many species to list in the per-species table")
    ap.add_argument("--dataset-id", default=GLOBAL_DATASET_ID,
                    help="CMEMS dataset id for the download command templates")
    args = ap.parse_args(argv)

    v = load_velocity_bounds(args.velocity)
    df = load_species(args.species_csv)
    if args.species:
        df = df[df["species"] == args.species].copy()
        if df.empty:
            sys.exit(f"No records for species '{args.species}'.")

    min_year = pd.Timestamp(f"{args.min_year}-01-01")
    t0, t1 = v["t_start"], v["t_end"]
    w0 = t0 - pd.DateOffset(years=args.window_years)
    w1 = t1 + pd.DateOffset(years=args.window_years)

    print("=" * 74)
    print("VELOCITY FIELD")
    print("=" * 74)
    print(f"  file:        {args.velocity}")
    print(f"  lon:         {v['lon_min']:.4f} .. {v['lon_max']:.4f}  "
          f"(n={v['n_lon']}, res={v['lon_res']:.4f} deg)")
    print(f"  lat:         {v['lat_min']:.4f} .. {v['lat_max']:.4f}  "
          f"(n={v['n_lat']}, res={v['lat_res']:.4f} deg)")
    print(f"  time:        {t0.date()} .. {t1.date()}  (n={v['n_t']} steps)")
    print(f"  window +-{args.window_years}y: {w0.date()} .. {w1.date()}")

    print("\n" + "=" * 74)
    print("SPECIES FILE")
    print("=" * 74)
    print(f"  file:        {args.species_csv}")
    print(f"  rows:        {len(df)}  | unique species: {df['species'].nunique()}"
          f"  | multi-species (';') cells: {int(df['is_multi'].sum())} "
          f"({100*df['is_multi'].mean():.0f}%)")
    print(f"  lon extent:  {df.longitude.min():.2f} .. {df.longitude.max():.2f}")
    print(f"  lat extent:  {df.latitude.min():.2f} .. {df.latitude.max():.2f}")
    print(f"  date extent: {df.dmin.min()} .. {df.dmax.max()}")

    # ---- coverage ----
    box = in_box(df, v)
    box_period = box[(box.dmid >= t0) & (box.dmid <= t1)]
    box_window = box[(box.dmid >= w0) & (box.dmid <= w1)]
    pre = df[df.dmid < min_year]
    eq = df[df.latitude.abs() < args.equator_band]
    usable = df[df.dmid >= min_year]

    print("\n" + "=" * 74)
    print("COVERAGE  (does the velocity field cover the records?)")
    print("=" * 74)
    print(f"  in velocity box:                    {len(box):6d} / {len(df)}")
    print(f"  in box AND model period:            {len(box_period):6d}")
    print(f"  in box AND +-{args.window_years}y window:           {len(box_window):6d}")
    print(f"  pre-{args.min_year} (NO altimetry possible):   {len(pre):6d}  -> unrecoverable")
    print(f"  in equatorial band |lat|<{args.equator_band:g} (weak): {len(eq):6d}  -> flag/down-weight")
    print(f"  downloadable (>= {args.min_year}):              {len(usable):6d}")

    spatial_ok = len(box) >= 0.5 * len(df)
    temporal_ok = len(box_period) >= max(20, 0.1 * len(box))
    print()
    if spatial_ok and temporal_ok:
        print("  VERDICT: records are adequately covered by this velocity field.")
    else:
        print("  VERDICT: MISMATCH — the velocity field does not cover the records.")
        if not spatial_ok:
            print("    * spatial: most records fall outside the velocity bounding box.")
        if not temporal_ok:
            print("    * temporal: too few records inside the model period.")
        print("    -> use the Copernicus download bounds below.")

    # ---- per-species table ----
    print("\n" + "=" * 74)
    print(f"PER-SPECIES (single-species rows, >= {args.min_year}, top {args.top} by count)")
    print("=" * 74)
    single = usable[~usable.is_multi].copy()
    print(f"  {'species':<32}{'n':>5}{'inbox':>6}  "
          f"{'lon_min..max':>16}  {'lat_min..max':>14}  date_min..max")
    for sp, c in single["species"].value_counts().head(args.top).items():
        s = single[single.species == sp]
        sb = len(in_box(s, v))
        print(f"  {sp[:32]:<32}{c:>5}{sb:>6}  "
              f"[{s.longitude.min():7.1f},{s.longitude.max():7.1f}]  "
              f"[{s.latitude.min():6.1f},{s.latitude.max():5.1f}]  "
              f"{s.dmid.min().date()}..{s.dmid.max().date()}")

    # ---- Copernicus download bounds ----
    print("\n" + "=" * 74)
    print(f"COPERNICUS DOWNLOAD BOUNDS  (records >= {args.min_year}, outside current field)")
    print("=" * 74)
    if usable.empty:
        print("  No downloadable records.")
        return 0

    print(f"  Overall: lon [{usable.longitude.min():.2f}, {usable.longitude.max():.2f}]  "
          f"lat [{usable.latitude.min():.2f}, {usable.latitude.max():.2f}]  "
          f"time {usable.dmid.min().date()} .. {usable.dmid.max().date()}")

    usable = usable.copy()
    usable["basin"] = [basin_of(lo, la)
                       for lo, la in zip(usable.longitude, usable.latitude)]
    print("\n  Regional tiles (download one cube each):")
    for b, g in usable.groupby("basin"):
        print(f"    {b:<18} n={len(g):4d}  "
              f"lon[{g.longitude.min():7.1f},{g.longitude.max():7.1f}] "
              f"lat[{g.latitude.min():6.1f},{g.latitude.max():6.1f}] "
              f"{g.dmid.min().date()}..{g.dmid.max().date()}")
        if b == "Pacific_dateline" and g.longitude.min() < -100 and g.longitude.max() > 147:
            print("      ^ straddles the dateline: split into 120..180 and -180..-100, "
                  "or pull the global cube.")

    # one ready-to-run command for the largest tile
    big = usable.groupby("basin").size().idxmax()
    g = usable[usable.basin == big]
    print(f"\n  Example command (largest tile = {big}):\n")
    print(fmt_cmd(args.dataset_id, g.longitude.min(), g.longitude.max(),
                  g.latitude.min(), g.latitude.max(),
                  max(g.dmid.min(), min_year).date(), g.dmid.max().date(), big))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
