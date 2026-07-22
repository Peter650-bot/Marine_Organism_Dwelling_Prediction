#!/usr/bin/env python3
"""
retention_core.py — shared, validated building blocks for the Lagrangian
retention SDM, generalised to any species / any Copernicus velocity cube.

These functions are the single source of truth reused by:
  * ablation_study.py   (feature ablation)
  * species_pipeline.py (per-species run + pooled model)

They reproduce enhanced_analysis.py Sections 1-2 (and 4a) exactly when given the
Mediterranean cube + Caretta CSV with the default parameters and seeds, so the
generalised tooling is consistent with the canonical figures 01-19.

Generalisations beyond the Caretta-specific script:
  * load_velocity reads ANY ugos/vgos cube and reports its longitude convention;
  * load_sightings takes a species name, drops ';'-merged gridded-summary cells,
    applies a min-year (altimetry-era) cutoff, optionally down-weights the
    equatorial band, and aligns occurrence longitudes to the cube's convention;
  * nothing assumes the Mediterranean extent or a 1/8 deg grid — resolution and
    bounds are always read from the file.
"""

import numpy as np
import pandas as pd
import h5py

# --- defaults: MUST match enhanced_analysis.py Section 2 for reproducibility ---
DT = 24 * 3600
K_DIFF = 10.0
N_PARTICLES = 500
N_ENSEMBLE = 10
MAX_STEPS = 365
SEED = 42
ALTIMETRY_START_YEAR = 1993

FEATURES = ["retention", "mean_speed", "vorticity", "latitude", "longitude", "dist_to_coast"]


# ============================================================================
#  Velocity
# ============================================================================
def load_velocity(path):
    """Load a ugos/vgos NetCDF. Returns a dict with grid, time, fields, and the
    longitude convention ('pm180' for [-180,180] or '0360' for [0,360]).

    Mirrors enhanced_analysis.py's raw-h5py + manual CF-time decode so the model
    period is identical to the pipeline's.
    """
    h5 = h5py.File(path, "r")
    lon = h5["longitude"][:].astype(np.float64)
    lat = h5["latitude"][:].astype(np.float64)
    t_raw = h5["time"][:]
    units = h5["time"].attrs.get("units", "seconds since 1970-01-01 00:00:00")
    if isinstance(units, bytes):
        units = units.decode()
    parts = str(units).split("since")
    uw = parts[0].strip().lower()
    origin = pd.Timestamp(parts[-1].strip())
    if uw.startswith("second"):
        ta = np.array([origin + pd.Timedelta(seconds=float(s)) for s in t_raw], dtype="datetime64[ns]")
    else:
        ta = np.array([origin + pd.Timedelta(days=float(d)) for d in t_raw], dtype="datetime64[ns]")
    u = np.where(np.abs(h5["ugos"][:]) > 1e10, np.nan, h5["ugos"][:])
    v = np.where(np.abs(h5["vgos"][:]) > 1e10, np.nan, h5["vgos"][:])
    h5.close()
    return {
        "lon": lon, "lat": lat, "time": pd.to_datetime(ta), "u": u, "v": v,
        "lon_conv": "0360" if lon.max() > 180 else "pm180",
        "t_start": pd.Timestamp(ta[0]), "t_end": pd.Timestamp(ta[-1]),
    }


def make_interp(u_data, v_data, lon, lat):
    """Fast bilinear interpolator on raw numpy arrays (the Section-2 hot path)."""
    lon0, lat0 = float(lon[0]), float(lat[0])
    dlon, dlat = float(lon[1] - lon[0]), float(lat[1] - lat[0])
    nlon, nlat = len(lon), len(lat)

    def interp(t, lon_pt, lat_pt):
        fi = (lon_pt - lon0) / dlon
        fj = (lat_pt - lat0) / dlat
        i0, j0 = int(np.floor(fi)), int(np.floor(fj))
        if i0 < 0 or i0 >= nlon - 1 or j0 < 0 or j0 >= nlat - 1:
            return np.nan, np.nan
        di, dj = fi - i0, fj - j0
        uc = u_data[t, j0:j0 + 2, i0:i0 + 2]
        vc = v_data[t, j0:j0 + 2, i0:i0 + 2]
        if np.isnan(uc).any() or np.isnan(vc).any():
            return np.nan, np.nan
        w = np.array([[(1 - di) * (1 - dj), di * (1 - dj)], [(1 - di) * dj, di * dj]])
        return float((uc.T * w.T).sum()), float((vc.T * w.T).sum())
    return interp, dlon, dlat


def simulate_retention(lon, lat, n_time, interp, n_particles=N_PARTICLES,
                       n_ensemble=N_ENSEMBLE, max_steps=MAX_STEPS, k_diff=K_DIFF, seed=SEED):
    """Ensemble visit-count retention field, normalised to [0,1].

    With the default args + Mediterranean cube this reproduces the canonical
    retention_norm of enhanced_analysis.py (same seeds 42, 42+r).
    """
    rng = np.random.default_rng(seed)
    init_lons = rng.uniform(lon.min() + 1, lon.max() - 1, n_particles)
    init_lats = rng.uniform(lat.min() + 1, lat.max() - 1, n_particles)
    noise_scale = np.sqrt(2 * k_diff * DT)
    ens = np.zeros((n_ensemble, len(lat), len(lon)))
    n_steps = min(max_steps, n_time - 1)
    for r in range(n_ensemble):
        rl = np.random.default_rng(seed + r)
        for p in range(n_particles):
            lo, la = float(init_lons[p]), float(init_lats[p])
            traj = [(lo, la)]
            for t in range(n_steps):
                up, vp = interp(t, lo, la)
                if np.isnan(up):
                    break
                la += (vp * DT + rl.normal(0, noise_scale)) / 111_000
                lo += (up * DT + rl.normal(0, noise_scale)) / (111_000 * np.cos(np.deg2rad(la)))
                if lo < lon[0] or lo > lon[-1] or la < lat[0] or la > lat[-1]:
                    break
                traj.append((lo, la))
            if len(traj) > 10:
                for lo_p, la_p in traj:
                    i = np.argmin(np.abs(lat - la_p))
                    j = np.argmin(np.abs(lon - lo_p))
                    ens[r, i, j] += 1
    mean = ens.mean(axis=0)
    std = ens.std(axis=0)
    norm = mean / mean.max() if mean.max() > 0 else mean
    return norm, mean, std


# ============================================================================
#  Sightings
# ============================================================================
def load_sightings(csv, species, vel, window_years=2, min_year=ALTIMETRY_START_YEAR,
                   drop_multi=True, equator_band=0.0):
    """Load + filter OBIS-SEAMAP records for one species against a velocity cube.

    Reuses the enhanced_analysis.py temporal logic verbatim (drop future dates,
    midpoint date, Gaussian temporal weight sigma=365 d, +/-window_years window,
    box filter) and adds the generalisations: species filter, ';'-cell drop,
    min-year cutoff, equatorial down-weight, longitude-convention alignment.
    Returns the in-domain dataframe with a `weighted_records` column.
    """
    df = pd.read_csv(csv)
    df = df[[c for c in df.columns if not df[c].isna().all()]]
    if species:
        df = df[df["species"].astype(str) == species].copy()
    if drop_multi:
        df = df[~df["species"].astype(str).str.contains(";", na=False)].copy()
    df["date_min"] = pd.to_datetime(df["date_min"], errors="coerce")
    df["date_max"] = pd.to_datetime(df["date_max"], errors="coerce")
    df = df[df["date_min"] <= pd.Timestamp("2024-12-31")].copy()
    if "num_animals" in df:
        df["num_animals"] = df["num_animals"].fillna(df["num_animals"].median())
    df["date_mid"] = df["date_min"] + (df["date_max"] - df["date_min"]) / 2
    df = df[df["date_mid"] >= pd.Timestamp(f"{min_year}-01-01")].copy()

    # align occurrence longitudes to the cube's convention before box-filtering
    if vel["lon_conv"] == "0360":
        df["longitude"] = df["longitude"] % 360

    t0, t1 = vel["t_start"], vel["t_end"]
    model_mid = t0 + (t1 - t0) / 2
    w0 = t0 - pd.DateOffset(years=window_years)
    w1 = t1 + pd.DateOffset(years=window_years)
    df = df[(df["date_mid"] >= w0) & (df["date_mid"] <= w1)].copy()
    days = (df["date_mid"] - model_mid).dt.total_seconds() / 86400
    df["temporal_weight"] = np.exp(-0.5 * (days / 365) ** 2)
    if equator_band > 0:
        df.loc[df["latitude"].abs() < equator_band, "temporal_weight"] *= 0.5
    df["weighted_records"] = df["num_records"] * df["temporal_weight"]

    lon, lat = vel["lon"], vel["lat"]
    df_med = df[(df.longitude >= lon.min()) & (df.longitude <= lon.max()) &
                (df.latitude >= lat.min()) & (df.latitude <= lat.max())].copy()
    return df_med


# ============================================================================
#  Features
# ============================================================================
def build_features(vel, retention_norm, df_med):
    """Build the 6-feature per-cell matrix + normalised sighting target.

    Identical feature definitions to enhanced_analysis.py Section 4a.
    Returns X (valid cells), y_reg, lats, lons, and the 2-D fields for plotting.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt
    lon, lat, u, v = vel["lon"], vel["lat"], vel["u"], vel["v"]
    Lon, Lat = np.meshgrid(lon, lat)
    u_mean = np.nanmean(u, axis=0)
    v_mean = np.nanmean(v, axis=0)
    speed = np.sqrt(u_mean ** 2 + v_mean ** 2)
    dv_dx = np.gradient(v_mean, lon * 111_000 * np.cos(np.deg2rad(lat.mean())), axis=1)
    du_dy = np.gradient(u_mean, lat * 111_000, axis=0)
    vort = dv_dx - du_dy
    if len(df_med) > 3:
        sg = griddata(df_med[["longitude", "latitude"]].values,
                      df_med["weighted_records"].values, (Lon, Lat),
                      method="linear", fill_value=0)
    else:
        sg = np.zeros_like(retention_norm)
    sg = np.clip(sg, 0, None)
    sights = sg / sg.max() if sg.max() > 0 else sg
    land = np.isnan(u[0])
    if land.any():
        dist = distance_transform_edt(~land) * np.mean([abs(lat[1] - lat[0]), abs(lon[1] - lon[0])])
    else:
        dist = np.full_like(retention_norm, 999.0)
    X = np.column_stack([retention_norm.ravel(), speed.ravel(), vort.ravel(),
                         Lat.ravel(), Lon.ravel(), dist.ravel()])
    y = sights.ravel()
    valid = ~np.isnan(X).any(axis=1)
    fields = {"retention": retention_norm, "sights": sights, "speed": speed,
              "vort": vort, "Lon": Lon, "Lat": Lat}
    return X[valid], y[valid], Lat.ravel()[valid], Lon.ravel()[valid], fields


def spatial_groups(lats, lons, block_deg=3.0):
    """Spatial-block labels for honest leave-block-out CV (spatial autocorrelation)."""
    return (np.floor(lats / block_deg).astype(int) * 100000 +
            np.floor(lons / block_deg).astype(int))
