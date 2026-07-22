#!/usr/bin/env python3
"""
Enhanced Analysis: Correlation Between Modeled Particle Retention Areas
and Caretta caretta Sightings in the Mediterranean Sea.

Master's Statistics Final Project — Enhanced Version
=====================================================
Sections:
  1. Data Preprocessing & Quality
  2. Improved Particle Simulation (stochastic, ensemble)
  3. Advanced Statistical Analysis (Moran's I, GWR, permutation, KDE)
  4. Machine Learning Models (RF, XGBoost+SHAP, DBSCAN, GMM, NN)
  5. Enhanced Visualizations (Plotly, animated GIF, 3D, SHAP plots)
  6. Validation & Robustness (sensitivity, bootstrap, seasonal)
  7. Summary Report
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, gaussian_kde
from scipy.interpolate import griddata

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (10, 6),
})

# Output directory for figures
import os
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name, fig=None):
    """Save current or given figure to FIG_DIR."""
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {FIG_DIR}/{name}")


# =============================================================================
#  SECTION 1 — DATA PREPROCESSING & QUALITY
# =============================================================================
print("=" * 70)
print("SECTION 1: DATA PREPROCESSING & QUALITY")
print("=" * 70)

# --- 1a. Load ocean current data ---
# Use h5py to read the HDF5/NetCDF4 file, then wrap in xarray
import h5py

_h5 = h5py.File("Oceanic_data.nc", "r")
lon_grid = _h5["longitude"][:]
lat_grid = _h5["latitude"][:]
_time_raw = _h5["time"][:]

# Decode time: CF convention — read units attribute
_time_units_str = _h5["time"].attrs.get("units", "seconds since 1970-01-01 00:00:00")
if isinstance(_time_units_str, bytes):
    _time_units_str = _time_units_str.decode()
# Parse "seconds since ..." or "days since ..."
_parts = _time_units_str.split("since")
_unit_word = _parts[0].strip().lower()   # "seconds" or "days"
_origin = pd.Timestamp(_parts[-1].strip())
if _unit_word.startswith("second"):
    time_arr = np.array([_origin + pd.Timedelta(seconds=float(s)) for s in _time_raw],
                         dtype="datetime64[ns]")
else:
    time_arr = np.array([_origin + pd.Timedelta(days=float(d)) for d in _time_raw],
                         dtype="datetime64[ns]")

_ugos = _h5["ugos"][:]  # shape (732, 125, 299)
_vgos = _h5["vgos"][:]
_h5.close()

# Replace NaN fill values (very large/small floats) with np.nan
_ugos = np.where(np.abs(_ugos) > 1e10, np.nan, _ugos)
_vgos = np.where(np.abs(_vgos) > 1e10, np.nan, _vgos)

# Build xarray Dataset so the rest of the code (.isel/.interp) works unchanged
ds = xr.Dataset(
    {
        "ugos": (["time", "latitude", "longitude"], _ugos),
        "vgos": (["time", "latitude", "longitude"], _vgos),
    },
    coords={"time": time_arr, "latitude": lat_grid, "longitude": lon_grid},
)
u_field = ds["ugos"]   # eastward geostrophic velocity (m/s)
v_field = ds["vgos"]   # northward geostrophic velocity (m/s)

model_start = pd.to_datetime(time_arr[0])
model_end = pd.to_datetime(time_arr[-1])

print(f"Ocean data: {len(lon_grid)} lon × {len(lat_grid)} lat × {len(time_arr)} time")
print(f"Model period: {model_start.date()} → {model_end.date()}")

# Pre-compute time-mean fields (used later as features)
u_mean = u_field.mean(dim="time").values
v_mean = v_field.mean(dim="time").values
speed_mean = np.sqrt(u_mean**2 + v_mean**2)

# Vorticity: ζ = ∂v/∂x − ∂u/∂y  (on the grid)
dv_dx = np.gradient(v_mean, lon_grid * 111_000 * np.cos(np.deg2rad(lat_grid.mean())), axis=1)
du_dy = np.gradient(u_mean, lat_grid * 111_000, axis=0)
vorticity_mean = dv_dx - du_dy

# --- 1b. Load & clean caretta data ---
df_raw = pd.read_csv("caretta_data.csv")
print(f"\nRaw caretta records: {len(df_raw)}")

# Drop fully-empty columns
empty_cols = [c for c in df_raw.columns if df_raw[c].isna().all()]
df_raw.drop(columns=empty_cols, inplace=True)
print(f"Dropped empty columns: {empty_cols}")

# Parse dates
df_raw["date_min"] = pd.to_datetime(df_raw["date_min"], errors="coerce")
df_raw["date_max"] = pd.to_datetime(df_raw["date_max"], errors="coerce")

# Remove impossible future dates (beyond 2024)
cutoff = pd.Timestamp("2024-12-31")
before = len(df_raw)
df_raw = df_raw[
    (df_raw["date_min"] <= cutoff) | df_raw["date_min"].isna()
].copy()
print(f"Removed {before - len(df_raw)} records with future dates (> 2024)")

# Median imputation for num_animals
median_animals = df_raw["num_animals"].median()
df_raw["num_animals"].fillna(median_animals, inplace=True)
print(f"Imputed {df_raw['num_animals'].isna().sum()} missing num_animals with median={median_animals:.0f}")

# Compute midpoint date
df_raw["date_mid"] = df_raw["date_min"] + (df_raw["date_max"] - df_raw["date_min"]) / 2

# --- 1c. Expanded temporal matching (±2 years) with proximity weighting ---
window_start = model_start - pd.DateOffset(years=2)  # 2010-01-01
window_end = model_end + pd.DateOffset(years=2)       # 2015-12-31

df = df_raw[
    (df_raw["date_mid"] >= window_start) &
    (df_raw["date_mid"] <= window_end)
].copy()

# Temporal proximity weight: Gaussian centered on model midpoint
model_mid = model_start + (model_end - model_start) / 2
df["days_from_center"] = (df["date_mid"] - model_mid).dt.total_seconds() / 86400
sigma_days = 365  # 1-year std deviation
df["temporal_weight"] = np.exp(-0.5 * (df["days_from_center"] / sigma_days) ** 2)

# Weighted sighting count
df["weighted_records"] = df["num_records"] * df["temporal_weight"]

print(f"\nSightings in ±2yr window: {len(df)}  (was {len(df_raw)} total)")
print(f"Temporal weight range: {df['temporal_weight'].min():.3f} – {df['temporal_weight'].max():.3f}")

# Filter to Mediterranean domain
lon_min, lon_max = lon_grid.min(), lon_grid.max()
lat_min, lat_max = lat_grid.min(), lat_grid.max()
df_med = df[
    (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max) &
    (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
].copy()
print(f"Sightings within ocean model domain: {len(df_med)}")

print("\nSection 1 complete.\n")


# =============================================================================
#  SECTION 2 — IMPROVED PARTICLE SIMULATION
# =============================================================================
print("=" * 70)
print("SECTION 2: IMPROVED PARTICLE SIMULATION")
print("=" * 70)

# --- 2a. Simulation parameters ---
dt = 24 * 3600          # 1 day (seconds)
K_diff = 10.0           # diffusivity (m²/s)
N_particles = 500
N_ensemble = 10
max_steps = min(365, len(time_arr))
alpha_wind = 0.01
u_wind, v_wind = 0.0, 0.0

# Random initialization across the full domain
rng = np.random.default_rng(42)
init_lons = rng.uniform(lon_grid.min() + 1, lon_grid.max() - 1, N_particles)
init_lats = rng.uniform(lat_grid.min() + 1, lat_grid.max() - 1, N_particles)

print(f"Particles: {N_particles} | Ensemble runs: {N_ensemble}")
print(f"Diffusivity K = {K_diff} m²/s | Max steps = {max_steps}")


# Pre-extract raw numpy arrays for fast interpolation (avoid xarray overhead)
_u_data = ds["ugos"].values   # (T, nlat, nlon) float32
_v_data = ds["vgos"].values
_lon_arr = lon_grid.astype(np.float64)
_lat_arr = lat_grid.astype(np.float64)
_dlon = float(_lon_arr[1] - _lon_arr[0])
_dlat = float(_lat_arr[1] - _lat_arr[0])
_lon0_grid = float(_lon_arr[0])
_lat0_grid = float(_lat_arr[0])
_nlon = len(_lon_arr)
_nlat = len(_lat_arr)


def _interp_uv(t_idx, lon_pt, lat_pt):
    """Fast bilinear interpolation of u, v from raw numpy arrays."""
    fi = (lon_pt - _lon0_grid) / _dlon
    fj = (lat_pt - _lat0_grid) / _dlat
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))
    if i0 < 0 or i0 >= _nlon - 1 or j0 < 0 or j0 >= _nlat - 1:
        return np.nan, np.nan
    di = fi - i0
    dj = fj - j0
    # Bilinear weights
    u00 = _u_data[t_idx, j0, i0]
    u10 = _u_data[t_idx, j0, i0 + 1]
    u01 = _u_data[t_idx, j0 + 1, i0]
    u11 = _u_data[t_idx, j0 + 1, i0 + 1]
    v00 = _v_data[t_idx, j0, i0]
    v10 = _v_data[t_idx, j0, i0 + 1]
    v01 = _v_data[t_idx, j0 + 1, i0]
    v11 = _v_data[t_idx, j0 + 1, i0 + 1]
    # Check for NaN in any corner
    if (np.isnan(u00) or np.isnan(u10) or np.isnan(u01) or np.isnan(u11) or
            np.isnan(v00) or np.isnan(v10) or np.isnan(v01) or np.isnan(v11)):
        return np.nan, np.nan
    u_val = (u00 * (1 - di) * (1 - dj) + u10 * di * (1 - dj) +
             u01 * (1 - di) * dj + u11 * di * dj)
    v_val = (v00 * (1 - di) * (1 - dj) + v10 * di * (1 - dj) +
             v01 * (1 - di) * dj + v11 * di * dj)
    return float(u_val), float(v_val)


def simulate_trajectory_stochastic(lon0, lat0, rng_local, max_steps=365):
    """Lagrangian advection with stochastic diffusion (fast numpy interp)."""
    lon, lat = float(lon0), float(lat0)
    traj = [(lon, lat)]
    n_steps = min(max_steps, len(time_arr) - 1)
    noise_scale = np.sqrt(2 * K_diff * dt)

    for t in range(n_steps):
        u_p, v_p = _interp_uv(t, lon, lat)

        if np.isnan(u_p):
            break

        # Deterministic advection + wind
        u_tot = u_p + alpha_wind * u_wind
        v_tot = v_p + alpha_wind * v_wind

        # Stochastic diffusion (random walk)
        noise_x = rng_local.normal(0, noise_scale)  # meters
        noise_y = rng_local.normal(0, noise_scale)

        dlat = (v_tot * dt + noise_y) / 111_000
        dlon = (u_tot * dt + noise_x) / (111_000 * np.cos(np.deg2rad(lat)))

        lat += dlat
        lon += dlon

        # Boundary check
        if lon < _lon_arr[0] or lon > _lon_arr[-1]:
            break
        if lat < _lat_arr[0] or lat > _lat_arr[-1]:
            break

        traj.append((lon, lat))

    return traj


# --- 2b. Run ensemble simulations ---
print("Running ensemble particle simulations...")
ensemble_trajectories = []   # list of N_ensemble lists of trajectories
ensemble_retention = np.zeros((N_ensemble, len(lat_grid), len(lon_grid)))

for run_idx in range(N_ensemble):
    rng_run = np.random.default_rng(42 + run_idx)
    run_trajs = []

    for p in range(N_particles):
        traj = simulate_trajectory_stochastic(
            init_lons[p], init_lats[p], rng_run, max_steps=max_steps)
        if len(traj) > 10:
            run_trajs.append(traj)

    # Build retention map for this run
    for traj in run_trajs:
        for lon_pt, lat_pt in traj:
            i = np.argmin(np.abs(lat_grid - lat_pt))
            j = np.argmin(np.abs(lon_grid - lon_pt))
            ensemble_retention[run_idx, i, j] += 1

    ensemble_trajectories.append(run_trajs)
    print(f"  Run {run_idx + 1}/{N_ensemble}: {len(run_trajs)} valid particles")

# Mean & std retention across ensemble
retention_mean = ensemble_retention.mean(axis=0)
retention_std = ensemble_retention.std(axis=0)
if retention_mean.max() > 0:
    retention_norm = retention_mean / retention_mean.max()
else:
    retention_norm = retention_mean

print(f"Mean retention max = {retention_mean.max():.1f}, "
      f"Normalized retention range = [0, 1]")
print("Section 2 complete.\n")

# Free xarray dataset (we have raw numpy arrays _u_data/_v_data)
ds.close()
del ds
import gc; gc.collect()

# =============================================================================
#  SECTION 3 — ADVANCED STATISTICAL ANALYSIS
# =============================================================================
print("=" * 70)
print("SECTION 3: ADVANCED STATISTICAL ANALYSIS")
print("=" * 70)

# --- 3a. Map sightings to model grid (weighted) ---
Lon_mesh, Lat_mesh = np.meshgrid(lon_grid, lat_grid)

if len(df_med) > 3:
    points = df_med[["longitude", "latitude"]].values
    values = df_med["weighted_records"].values
    sightings_grid = griddata(points, values, (Lon_mesh, Lat_mesh),
                              method="linear", fill_value=0)
else:
    sightings_grid = np.zeros_like(retention_norm)

sightings_grid = np.clip(sightings_grid, 0, None)
if sightings_grid.max() > 0:
    sightings_norm = sightings_grid / sightings_grid.max()
else:
    sightings_norm = sightings_grid

# --- 3b. Spearman correlation ---
mask = (retention_norm > 0) | (sightings_norm > 0)
ret_vals = retention_norm[mask]
sight_vals = sightings_norm[mask]

if len(ret_vals) > 2:
    corr_spearman, pval_spearman = spearmanr(ret_vals, sight_vals)
else:
    corr_spearman, pval_spearman = np.nan, np.nan
print(f"Spearman ρ = {corr_spearman:.4f},  p = {pval_spearman:.3e}")

# --- 3c. Permutation test (N=1000) ---
N_perm = 1000
perm_corrs = np.zeros(N_perm)
for i in range(N_perm):
    shuffled = rng.permutation(sight_vals)
    perm_corrs[i], _ = spearmanr(ret_vals, shuffled)

perm_pval = np.mean(np.abs(perm_corrs) >= np.abs(corr_spearman))
print(f"Permutation test p-value (N={N_perm}): {perm_pval:.4f}")

# --- 3d. Bootstrap confidence intervals ---
N_boot = 1000
boot_corrs = np.zeros(N_boot)
n_pts = len(ret_vals)
for i in range(N_boot):
    idx = rng.integers(0, n_pts, n_pts)
    boot_corrs[i], _ = spearmanr(ret_vals[idx], sight_vals[idx])

ci_low, ci_high = np.percentile(boot_corrs, [2.5, 97.5])
print(f"Bootstrap 95% CI for Spearman ρ: [{ci_low:.4f}, {ci_high:.4f}]")

# --- 3e. Moran's I for spatial autocorrelation ---
try:
    from libpysal.weights import lat2W
    from esda.moran import Moran

    # Use a coarsened grid for computational feasibility
    step = 4
    ret_coarse = retention_norm[::step, ::step]
    sight_coarse = sightings_norm[::step, ::step]
    nr, nc = ret_coarse.shape

    w = lat2W(nr, nc)

    moran_ret = Moran(ret_coarse.flatten(), w)
    moran_sight = Moran(sight_coarse.flatten(), w)

    print(f"Moran's I (retention):  I = {moran_ret.I:.4f}, p = {moran_ret.p_sim:.4f}")
    print(f"Moran's I (sightings): I = {moran_sight.I:.4f}, p = {moran_sight.p_sim:.4f}")
    MORANS_AVAILABLE = True
except ImportError:
    print("libpysal/esda not installed — skipping Moran's I. "
          "Install with: pip install libpysal esda")
    moran_ret = moran_sight = None
    MORANS_AVAILABLE = False

# --- 3f. Kernel Density Estimation ---
# KDE for sightings (point locations)
if len(df_med) >= 2:
    sight_xy = df_med[["longitude", "latitude"]].values.T
    kde_sight = gaussian_kde(sight_xy, bw_method=0.3)
    kde_sight_vals = kde_sight(np.vstack([Lon_mesh.ravel(), Lat_mesh.ravel()]))
    kde_sight_grid = kde_sight_vals.reshape(Lon_mesh.shape)
else:
    kde_sight_grid = np.zeros_like(Lon_mesh)

# KDE for particle endpoints (last positions of first ensemble run)
endpoints = []
for traj in ensemble_trajectories[0]:
    if len(traj) > 0:
        endpoints.append(traj[-1])
if len(endpoints) >= 2:
    ep_arr = np.array(endpoints).T
    kde_part = gaussian_kde(ep_arr, bw_method=0.3)
    kde_part_vals = kde_part(np.vstack([Lon_mesh.ravel(), Lat_mesh.ravel()]))
    kde_part_grid = kde_part_vals.reshape(Lon_mesh.shape)
else:
    kde_part_grid = np.zeros_like(Lon_mesh)

# --- 3g. Geographically Weighted Regression (lightweight version) ---
# Using a local weighted regression approach
print("\nLocal spatial regression (GWR-lite):")

# Flatten grid to points
flat_lon = Lon_mesh[mask].ravel()
flat_lat = Lat_mesh[mask].ravel()
coords = np.column_stack([flat_lon, flat_lat])

bw = 2.0  # bandwidth in degrees
n_sample = min(len(ret_vals), 2000)  # subsample for speed
idx_sample = rng.choice(len(ret_vals), n_sample, replace=False)

local_coeffs = np.zeros(n_sample)
for ii, si in enumerate(idx_sample):
    dists = np.sqrt((coords[:, 0] - coords[si, 0])**2 +
                    (coords[:, 1] - coords[si, 1])**2)
    weights = np.exp(-0.5 * (dists / bw) ** 2)
    weights /= weights.sum()

    x_w = ret_vals * weights
    y_w = sight_vals * weights
    x_mean = np.sum(x_w)
    y_mean = np.sum(y_w)
    cov_xy = np.sum(weights * (ret_vals - x_mean) * (sight_vals - y_mean))
    var_x = np.sum(weights * (ret_vals - x_mean) ** 2)
    local_coeffs[ii] = cov_xy / var_x if var_x > 1e-12 else 0.0

print(f"  Local regression coefficients: "
      f"mean = {local_coeffs.mean():.4f}, "
      f"std = {local_coeffs.std():.4f}, "
      f"range = [{local_coeffs.min():.4f}, {local_coeffs.max():.4f}]")

print("Section 3 complete.\n")


# =============================================================================
#  SECTION 4 — MACHINE LEARNING MODELS
# =============================================================================
print("=" * 70)
print("SECTION 4: MACHINE LEARNING MODELS")
print("=" * 70)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# --- 4a. Build feature matrix from grid ---
# Features per grid cell: retention, mean speed, vorticity, lat, lon
flat_retention = retention_norm.ravel()
flat_sightings = sightings_norm.ravel()
flat_speed = speed_mean.ravel()
flat_vorticity = vorticity_mean.ravel()
flat_lons = Lon_mesh.ravel()
flat_lats = Lat_mesh.ravel()

# Distance to nearest coast (proxy: NaN mask from ocean data)
# Cells where current data is NaN are "land"; distance to nearest NaN cell
u_snapshot = u_field.isel(time=0).values
land_mask = np.isnan(u_snapshot)

# Compute distance to coast using scipy.ndimage (memory-efficient)
from scipy.ndimage import distance_transform_edt

if land_mask.any():
    # distance_transform_edt computes distance from each non-zero cell
    # to nearest zero cell. We want distance from ocean (True) to land (False=0).
    # Invert: land=0, ocean=1 → distance gives ocean-to-land in grid units
    dist_grid = distance_transform_edt(~land_mask)  # grid-cell units
    # Convert to approximate degrees (average spacing)
    dist_grid = dist_grid * np.mean([abs(_dlat), abs(_dlon)])
    dist_to_coast = dist_grid.ravel()
else:
    dist_to_coast = np.ones(len(flat_lons)) * 999

feature_names = ["retention", "mean_speed", "vorticity",
                 "latitude", "longitude", "dist_to_coast"]
X_all = np.column_stack([
    flat_retention, flat_speed, flat_vorticity,
    flat_lats, flat_lons, dist_to_coast
])

# Target: sightings density (continuous) & binary (high/low)
y_reg = flat_sightings
threshold = np.percentile(flat_sightings[flat_sightings > 0], 50) if (flat_sightings > 0).any() else 0.01
y_cls = (flat_sightings > threshold).astype(int)

# Filter to cells with valid ocean data
valid_mask = ~np.isnan(X_all).any(axis=1)
X = X_all[valid_mask]
y_r = y_reg[valid_mask]
y_c = y_cls[valid_mask]

print(f"Feature matrix: {X.shape[0]} cells × {X.shape[1]} features")
print(f"Classification target: {y_c.sum()} positive / {(1 - y_c).sum()} negative")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
    X_scaled, y_c, y_r, test_size=0.25, random_state=42, stratify=y_c
)

# --- 4b. Random Forest Classifier ---
print("\n--- Random Forest Classifier ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                            class_weight="balanced")
rf.fit(X_train, y_train_c)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(y_test_c, rf_prob)
print(f"AUC-ROC: {rf_auc:.4f}")
print(classification_report(y_test_c, rf_pred, target_names=["Low", "High"]))

# Cross-validation
cv_scores = cross_val_score(rf, X_scaled, y_c, cv=5, scoring="roc_auc")
print(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
rf_importance = rf.feature_importances_

# --- 4c. Gradient Boosting Regressor (sklearn) + SHAP ---
print("--- Gradient Boosting Regressor ---")
gbr = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                random_state=42)
gbr.fit(X_train, y_train_r)
gbr_pred = gbr.predict(X_test)

gbr_rmse = np.sqrt(mean_squared_error(y_test_r, gbr_pred))
gbr_r2 = r2_score(y_test_r, gbr_pred)
print(f"RMSE: {gbr_rmse:.4f}")
print(f"R²:   {gbr_r2:.4f}")

# SHAP analysis
try:
    import shap
    explainer = shap.TreeExplainer(gbr)
    shap_values = explainer.shap_values(X_test[:500])  # subsample for speed
    SHAP_AVAILABLE = True
    print("SHAP values computed successfully.")
except ImportError:
    SHAP_AVAILABLE = False
    shap_values = None
    print("shap not installed — skipping SHAP. Install with: pip install shap")

# --- 4d. DBSCAN Spatial Clustering ---
print("\n--- DBSCAN Spatial Clustering ---")

# Cluster sightings
if len(df_med) >= 5:
    sight_coords = df_med[["longitude", "latitude"]].values
    db_sight = DBSCAN(eps=1.0, min_samples=3).fit(sight_coords)
    n_sight_clusters = len(set(db_sight.labels_)) - (1 if -1 in db_sight.labels_ else 0)
    print(f"Sighting clusters: {n_sight_clusters} "
          f"(noise points: {(db_sight.labels_ == -1).sum()})")
else:
    db_sight = None
    n_sight_clusters = 0
    print("Too few sightings for DBSCAN.")

# Cluster particle endpoints
if len(endpoints) >= 5:
    ep_coords = np.array(endpoints)
    db_part = DBSCAN(eps=0.8, min_samples=5).fit(ep_coords)
    n_part_clusters = len(set(db_part.labels_)) - (1 if -1 in db_part.labels_ else 0)
    print(f"Particle endpoint clusters: {n_part_clusters} "
          f"(noise points: {(db_part.labels_ == -1).sum()})")
else:
    db_part = None
    n_part_clusters = 0

# --- 4e. Gaussian Mixture Model for retention zones ---
print("\n--- Gaussian Mixture Model ---")
# Extract high-retention points
high_ret_mask = retention_norm > 0.1
if high_ret_mask.sum() > 10:
    ret_points = np.column_stack([
        Lon_mesh[high_ret_mask], Lat_mesh[high_ret_mask]
    ])

    best_bic = np.inf
    best_n = 2
    for n_comp in range(2, 8):
        gmm_test = GaussianMixture(n_components=n_comp, random_state=42)
        gmm_test.fit(ret_points)
        bic = gmm_test.bic(ret_points)
        if bic < best_bic:
            best_bic = bic
            best_n = n_comp

    gmm = GaussianMixture(n_components=best_n, random_state=42)
    gmm.fit(ret_points)
    gmm_labels = gmm.predict(ret_points)
    print(f"Optimal GMM components (BIC): {best_n}")
    print(f"GMM means:\n{gmm.means_}")
else:
    gmm = None
    best_n = 0
    print("Insufficient high-retention cells for GMM.")

# --- 4f. Simple Neural Network ---
print("\n--- Neural Network ---")
try:
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                       random_state=42, early_stopping=True,
                       validation_fraction=0.15)
    nn.fit(X_train, y_train_c)
    nn_pred = nn.predict(X_test)
    nn_prob = nn.predict_proba(X_test)[:, 1]
    nn_auc = roc_auc_score(y_test_c, nn_prob)
    print(f"MLP AUC-ROC: {nn_auc:.4f}")
    print(classification_report(y_test_c, nn_pred, target_names=["Low", "High"]))
    NN_AVAILABLE = True
except Exception as e:
    print(f"Neural network failed: {e}")
    nn_auc = np.nan
    nn_prob = None
    NN_AVAILABLE = False

print("Section 4 complete.\n")


# =============================================================================
#  SECTION 5 — ENHANCED VISUALIZATIONS
# =============================================================================
print("=" * 70)
print("SECTION 5: ENHANCED VISUALIZATIONS")
print("=" * 70)

# --- 5a. Retention & sightings heatmaps (publication quality) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].pcolormesh(lon_grid, lat_grid, retention_norm,
                          shading="auto", cmap="YlOrRd")
axes[0].set_title("Particle Retention (Ensemble Mean)")
axes[0].set_xlabel("Longitude (°E)")
axes[0].set_ylabel("Latitude (°N)")
plt.colorbar(im0, ax=axes[0], label="Normalized retention")

im1 = axes[1].pcolormesh(lon_grid, lat_grid, sightings_norm,
                          shading="auto", cmap="YlGnBu")
axes[1].set_title("Caretta Sightings Density (Weighted)")
axes[1].set_xlabel("Longitude (°E)")
axes[1].set_ylabel("Latitude (°N)")
plt.colorbar(im1, ax=axes[1], label="Normalized sightings")

axes[2].scatter(ret_vals, sight_vals, s=3, alpha=0.4, c="steelblue")
axes[2].set_xlabel("Retention")
axes[2].set_ylabel("Sightings")
axes[2].set_title(f"Retention vs Sightings (ρ={corr_spearman:.3f})")

plt.tight_layout()
savefig("01_retention_vs_sightings.png")

# --- 5b. Ensemble retention uncertainty ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].pcolormesh(lon_grid, lat_grid, retention_mean,
                          shading="auto", cmap="hot")
axes[0].set_title("Mean Retention (Ensemble)")
axes[0].set_xlabel("Longitude (°E)")
axes[0].set_ylabel("Latitude (°N)")
plt.colorbar(im0, ax=axes[0], label="Visit count")

im1 = axes[1].pcolormesh(lon_grid, lat_grid, retention_std,
                          shading="auto", cmap="Purples")
axes[1].set_title("Retention Std Dev (Ensemble Uncertainty)")
axes[1].set_xlabel("Longitude (°E)")
axes[1].set_ylabel("Latitude (°N)")
plt.colorbar(im1, ax=axes[1], label="Std dev")

plt.tight_layout()
savefig("02_ensemble_uncertainty.png")

# --- 5c. KDE contour maps side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

vmax = max(kde_part_grid.max(), kde_sight_grid.max())
if vmax == 0:
    vmax = 1

c0 = axes[0].contourf(Lon_mesh, Lat_mesh, kde_part_grid, levels=20,
                        cmap="Reds", vmin=0, vmax=vmax)
axes[0].set_title("KDE: Particle Endpoints")
axes[0].set_xlabel("Longitude (°E)")
axes[0].set_ylabel("Latitude (°N)")

axes[1].contourf(Lon_mesh, Lat_mesh, kde_sight_grid, levels=20,
                 cmap="Blues", vmin=0, vmax=vmax)
axes[1].set_title("KDE: Caretta Sightings")
axes[1].set_xlabel("Longitude (°E)")
axes[1].set_ylabel("Latitude (°N)")

fig.colorbar(c0, ax=axes, label="Density", shrink=0.8)
plt.tight_layout()
savefig("03_kde_contours.png")

# --- 5d. Permutation test histogram ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(perm_corrs, bins=50, color="skyblue", edgecolor="white", label="Null distribution")
ax.axvline(corr_spearman, color="red", linewidth=2, linestyle="--",
           label=f"Observed ρ = {corr_spearman:.3f}")
ax.set_xlabel("Spearman ρ")
ax.set_ylabel("Frequency")
ax.set_title(f"Permutation Test (N={N_perm}, p={perm_pval:.4f})")
ax.legend()
savefig("04_permutation_test.png")

# --- 5e. Bootstrap CI distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(boot_corrs, bins=50, color="salmon", edgecolor="white")
ax.axvline(ci_low, color="black", linestyle="--", label=f"2.5% = {ci_low:.3f}")
ax.axvline(ci_high, color="black", linestyle="--", label=f"97.5% = {ci_high:.3f}")
ax.axvline(corr_spearman, color="red", linewidth=2, label=f"ρ = {corr_spearman:.3f}")
ax.set_xlabel("Spearman ρ")
ax.set_ylabel("Frequency")
ax.set_title(f"Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
ax.legend()
savefig("05_bootstrap_ci.png")

# --- 5f. Random Forest feature importance ---
fig, ax = plt.subplots(figsize=(8, 5))
sorted_idx = np.argsort(rf_importance)
ax.barh([feature_names[i] for i in sorted_idx], rf_importance[sorted_idx],
        color="teal")
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
savefig("06_rf_feature_importance.png")

# --- 5g. ROC curves ---
fig, ax = plt.subplots(figsize=(7, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test_c, rf_prob)
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})", linewidth=2)

if NN_AVAILABLE and nn_prob is not None:
    fpr_nn, tpr_nn, _ = roc_curve(y_test_c, nn_prob)
    ax.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC={nn_auc:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Sighting Zone Classification")
ax.legend()
savefig("07_roc_curves.png")

# --- 5h. Confusion matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test_c, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Random Forest Confusion Matrix")
savefig("08_confusion_matrix.png")

# --- 5i. GBR residual plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

residuals = y_test_r - gbr_pred
axes[0].scatter(gbr_pred, residuals, s=3, alpha=0.4, color="coral")
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residuals")
axes[0].set_title("GBR Residual Plot")

axes[1].scatter(y_test_r, gbr_pred, s=3, alpha=0.4, color="steelblue")
lims = [0, max(y_test_r.max(), gbr_pred.max())]
axes[1].plot(lims, lims, "k--", alpha=0.5)
axes[1].set_xlabel("Actual Sightings")
axes[1].set_ylabel("Predicted Sightings")
axes[1].set_title(f"Actual vs Predicted (R²={gbr_r2:.3f})")

plt.tight_layout()
savefig("09_gbr_residuals.png")

# --- 5j. Correlation matrix of oceanographic features ---
fig, ax = plt.subplots(figsize=(8, 7))
feat_df = pd.DataFrame(X[valid_mask[:X.shape[0]] if len(valid_mask) > X.shape[0] else slice(None)],
                        columns=feature_names)
corr_matrix = feat_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, square=True)
ax.set_title("Feature Correlation Matrix")
savefig("10_feature_correlation.png")

# --- 5k. SHAP summary plot ---
if SHAP_AVAILABLE and shap_values is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names,
                      show=False)
    savefig("11_shap_summary.png")
    print("  SHAP summary plot saved.")

# --- 5l. DBSCAN cluster visualization ---
if db_sight is not None and n_sight_clusters > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(sight_coords[:, 0], sight_coords[:, 1],
                    c=db_sight.labels_, cmap="tab10", s=15, alpha=0.7)
    axes[0].set_title(f"DBSCAN: Sighting Clusters ({n_sight_clusters})")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")

    if db_part is not None:
        axes[1].scatter(ep_coords[:, 0], ep_coords[:, 1],
                        c=db_part.labels_, cmap="tab10", s=10, alpha=0.7)
        axes[1].set_title(f"DBSCAN: Particle Endpoint Clusters ({n_part_clusters})")
    else:
        axes[1].text(0.5, 0.5, "Insufficient data", transform=axes[1].transAxes,
                     ha="center")
    axes[1].set_xlabel("Longitude (°E)")
    axes[1].set_ylabel("Latitude (°N)")

    plt.tight_layout()
    savefig("12_dbscan_clusters.png")

# --- 5m. GMM retention zones ---
if gmm is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(lon_grid, lat_grid, retention_norm, shading="auto",
                  cmap="YlOrRd", alpha=0.6)
    scatter = ax.scatter(ret_points[:, 0], ret_points[:, 1],
                         c=gmm_labels, cmap="Set1", s=8, alpha=0.8)
    for k in range(best_n):
        ax.plot(gmm.means_[k, 0], gmm.means_[k, 1], "k*", markersize=15)
    ax.set_title(f"GMM Retention Zones ({best_n} components)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.colorbar(plt.cm.ScalarMappable(cmap="YlOrRd"), ax=ax, label="Retention")
    savefig("13_gmm_retention.png")

# --- 5n. 3D surface plot of retention ---
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Lon_mesh, Lat_mesh, retention_norm,
                cmap="YlOrRd", alpha=0.9, edgecolor="none")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_zlabel("Retention (norm)")
ax.set_title("3D Retention Surface")
ax.view_init(elev=35, azim=225)
savefig("14_3d_retention.png")

# --- 5o. Animated particle trajectory GIF (memory-efficient) ---
print("Generating animated particle GIF (first ensemble run)...")
sample_trajs = ensemble_trajectories[0][:30]  # 30 particles for speed/memory
max_anim_len = max(len(t) for t in sample_trajs)
n_frames = 40  # keep frame count low for memory
frame_step = max(1, max_anim_len // n_frames)
colors_anim = plt.cm.viridis(np.linspace(0, 1, len(sample_trajs)))

try:
    from PIL import Image
    import io
    pil_frames = []
    for frame_idx in range(n_frames):
        step = frame_idx * frame_step
        fig_f, ax_f = plt.subplots(figsize=(8, 6))
        ax_f.set_xlim(lon_grid.min(), lon_grid.max())
        ax_f.set_ylim(lat_grid.min(), lat_grid.max())
        ax_f.set_xlabel("Longitude (°E)")
        ax_f.set_ylabel("Latitude (°N)")
        ax_f.set_title(f"Particle Drift — Day {step}")
        for k, traj in enumerate(sample_trajs):
            idx = min(step, len(traj) - 1)
            lons_t = [p[0] for p in traj[:idx + 1]]
            lats_t = [p[1] for p in traj[:idx + 1]]
            ax_f.plot(lons_t, lats_t, "-", color=colors_anim[k],
                      linewidth=0.8, alpha=0.6)
            ax_f.plot(lons_t[-1], lats_t[-1], "o", color=colors_anim[k],
                      markersize=3)
        buf = io.BytesIO()
        fig_f.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig_f)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        buf.close()

    gif_path = os.path.join(FIG_DIR, "15_particle_animation.gif")
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                        duration=120, loop=0)
    print(f"  [saved] {gif_path}")
except Exception as e:
    print(f"  GIF generation failed ({e}). Saving last-frame PNG instead.")
    fig_f, ax_f = plt.subplots(figsize=(8, 6))
    for k, traj in enumerate(sample_trajs):
        lons_t = [p[0] for p in traj]
        lats_t = [p[1] for p in traj]
        ax_f.plot(lons_t, lats_t, "-", color=colors_anim[k], linewidth=0.8, alpha=0.6)
    ax_f.set_xlabel("Longitude (°E)")
    ax_f.set_ylabel("Latitude (°N)")
    ax_f.set_title("Particle Trajectories (Final)")
    savefig("15_particle_animation_lastframe.png", fig_f)

# --- 5p. Interactive Plotly map ---
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig_plotly = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Particle Retention", "Caretta Sightings"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
    )

    fig_plotly.add_trace(
        go.Heatmap(z=retention_norm, x=lon_grid, y=lat_grid,
                    colorscale="YlOrRd", name="Retention"),
        row=1, col=1
    )
    fig_plotly.add_trace(
        go.Heatmap(z=sightings_norm, x=lon_grid, y=lat_grid,
                    colorscale="YlGnBu", name="Sightings"),
        row=1, col=2
    )

    # Add sighting points
    if len(df_med) > 0:
        fig_plotly.add_trace(
            go.Scatter(x=df_med["longitude"], y=df_med["latitude"],
                       mode="markers", marker=dict(size=5, color="red"),
                       name="Sightings"),
            row=1, col=2
        )

    fig_plotly.update_layout(
        title="Interactive: Retention vs Sightings",
        height=500, width=1100
    )
    fig_plotly.write_html(os.path.join(FIG_DIR, "16_interactive_map.html"))
    print(f"  [saved] {FIG_DIR}/16_interactive_map.html")
except ImportError:
    print("plotly not installed — skipping interactive map. "
          "Install with: pip install plotly")

# --- 5q. Sample particle trajectories ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.pcolormesh(lon_grid, lat_grid, retention_norm, shading="auto",
              cmap="YlOrRd", alpha=0.5)
for k, traj in enumerate(ensemble_trajectories[0][:30]):
    lons_t = [p[0] for p in traj]
    lats_t = [p[1] for p in traj]
    ax.plot(lons_t, lats_t, linewidth=0.7, alpha=0.6)
    ax.plot(lons_t[0], lats_t[0], "k*", markersize=5)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Sample Particle Trajectories over Retention Map")
savefig("17_sample_trajectories.png")

print("Section 5 complete.\n")


# =============================================================================
#  SECTION 6 — VALIDATION & ROBUSTNESS
# =============================================================================
print("=" * 70)
print("SECTION 6: VALIDATION & ROBUSTNESS")
print("=" * 70)

# --- 6a. Sensitivity analysis ---
print("\n--- Sensitivity Analysis ---")

sensitivity_results = {}

# Vary diffusivity K
print("  Varying diffusivity K...")
for K_test in [0, 5, 10, 50, 100]:
    rng_sa = np.random.default_rng(99)
    sa_retention = np.zeros((len(lat_grid), len(lon_grid)))
    # Use 100 particles for speed
    for p in range(100):
        lon0_sa = init_lons[p % N_particles]
        lat0_sa = init_lats[p % N_particles]
        lon_c, lat_c = float(lon0_sa), float(lat0_sa)
        for t in range(min(180, len(time_arr) - 1)):
            u_p = float(u_field.isel(time=t).interp(
                longitude=lon_c, latitude=lat_c).values)
            v_p = float(v_field.isel(time=t).interp(
                longitude=lon_c, latitude=lat_c).values)
            if np.isnan(u_p) or np.isnan(v_p):
                break
            noise_x = rng_sa.normal(0, np.sqrt(2 * K_test * dt)) if K_test > 0 else 0
            noise_y = rng_sa.normal(0, np.sqrt(2 * K_test * dt)) if K_test > 0 else 0
            dlat_sa = (v_p * dt + noise_y) / 111_000
            dlon_sa = (u_p * dt + noise_x) / (111_000 * np.cos(np.deg2rad(lat_c)))
            lat_c += dlat_sa
            lon_c += dlon_sa
            if (lon_c < lon_grid.min() or lon_c > lon_grid.max() or
                    lat_c < lat_grid.min() or lat_c > lat_grid.max()):
                break
            i = np.argmin(np.abs(lat_grid - lat_c))
            j = np.argmin(np.abs(lon_grid - lon_c))
            sa_retention[i, j] += 1

    if sa_retention.max() > 0:
        sa_norm = sa_retention / sa_retention.max()
    else:
        sa_norm = sa_retention
    sa_mask = (sa_norm > 0) | (sightings_norm > 0)
    if sa_mask.sum() > 2:
        sa_corr, _ = spearmanr(sa_norm[sa_mask], sightings_norm[sa_mask])
    else:
        sa_corr = np.nan
    sensitivity_results[f"K={K_test}"] = sa_corr
    print(f"    K={K_test:>4} m²/s → ρ = {sa_corr:.4f}")

# --- 6b. Seasonal comparison ---
print("\n--- Seasonal Comparison ---")
# Winter: months 11,12,1,2,3 vs Summer: 5,6,7,8,9
seasons = {
    "Winter": [11, 12, 1, 2, 3],
    "Summer": [5, 6, 7, 8, 9]
}

seasonal_retention = {}
for season_name, months in seasons.items():
    time_pd = pd.to_datetime(time_arr)
    season_mask = np.isin(time_pd.month, months)
    season_indices = np.where(season_mask)[0]

    if len(season_indices) < 30:
        print(f"  {season_name}: insufficient timesteps ({len(season_indices)})")
        continue

    s_retention = np.zeros((len(lat_grid), len(lon_grid)))
    rng_s = np.random.default_rng(77)

    for p in range(100):
        lon_c = float(init_lons[p % N_particles])
        lat_c = float(init_lats[p % N_particles])
        for t_idx in season_indices[:180]:
            u_p = float(u_field.isel(time=int(t_idx)).interp(
                longitude=lon_c, latitude=lat_c).values)
            v_p = float(v_field.isel(time=int(t_idx)).interp(
                longitude=lon_c, latitude=lat_c).values)
            if np.isnan(u_p) or np.isnan(v_p):
                break
            noise_x = rng_s.normal(0, np.sqrt(2 * K_diff * dt))
            noise_y = rng_s.normal(0, np.sqrt(2 * K_diff * dt))
            dlat_s = (v_p * dt + noise_y) / 111_000
            dlon_s = (u_p * dt + noise_x) / (111_000 * np.cos(np.deg2rad(lat_c)))
            lat_c += dlat_s
            lon_c += dlon_s
            if (lon_c < lon_grid.min() or lon_c > lon_grid.max() or
                    lat_c < lat_grid.min() or lat_c > lat_grid.max()):
                break
            i = np.argmin(np.abs(lat_grid - lat_c))
            j = np.argmin(np.abs(lon_grid - lon_c))
            s_retention[i, j] += 1

    if s_retention.max() > 0:
        s_norm = s_retention / s_retention.max()
    else:
        s_norm = s_retention

    seasonal_retention[season_name] = s_norm
    s_mask = (s_norm > 0) | (sightings_norm > 0)
    if s_mask.sum() > 2:
        s_corr, s_pval = spearmanr(s_norm[s_mask], sightings_norm[s_mask])
    else:
        s_corr, s_pval = np.nan, np.nan
    print(f"  {season_name}: ρ = {s_corr:.4f}, p = {s_pval:.3e}")

# Plot seasonal comparison
if len(seasonal_retention) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (sname, sret) in zip(axes, seasonal_retention.items()):
        im = ax.pcolormesh(lon_grid, lat_grid, sret, shading="auto", cmap="YlOrRd")
        ax.set_title(f"Retention — {sname}")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        plt.colorbar(im, ax=ax, label="Normalized")
    plt.tight_layout()
    savefig("18_seasonal_retention.png")

# --- 6c. Sensitivity bar chart ---
fig, ax = plt.subplots(figsize=(8, 5))
k_labels = list(sensitivity_results.keys())
k_corrs = list(sensitivity_results.values())
bars = ax.bar(k_labels, k_corrs, color="teal", edgecolor="white")
ax.axhline(corr_spearman, color="red", linestyle="--",
           label=f"Baseline (K={K_diff})")
ax.set_xlabel("Diffusivity Setting")
ax.set_ylabel("Spearman ρ")
ax.set_title("Sensitivity: Correlation vs Diffusivity")
ax.legend()
for bar, val in zip(bars, k_corrs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontsize=9)
savefig("19_sensitivity_diffusivity.png")

print("Section 6 complete.\n")


# =============================================================================
#  SECTION 7 — SUMMARY REPORT
# =============================================================================
print("=" * 70)
print("SECTION 7: SUMMARY REPORT")
print("=" * 70)

print(f"""
{'='*60}
  ENHANCED ANALYSIS — RESULTS SUMMARY
{'='*60}

DATA
  Ocean grid:          {len(lon_grid)} × {len(lat_grid)} × {len(time_arr)}
  Model period:        {model_start.date()} → {model_end.date()}
  Caretta records:     {len(df_raw)} total, {len(df)} in ±2yr window
  Mediterranean:       {len(df_med)} sightings in model domain
  Empty cols dropped:  {empty_cols}

PARTICLE SIMULATION
  Particles:           {N_particles}
  Ensemble runs:       {N_ensemble}
  Diffusivity K:       {K_diff} m²/s
  Max steps:           {max_steps} days

CORRELATION ANALYSIS
  Spearman ρ:          {corr_spearman:.4f}
  Spearman p-value:    {pval_spearman:.3e}
  Permutation p:       {perm_pval:.4f}  (N={N_perm})
  Bootstrap 95% CI:    [{ci_low:.4f}, {ci_high:.4f}]  (N={N_boot})

SPATIAL AUTOCORRELATION (Moran's I)
  Retention:           {"I = {:.4f}, p = {:.4f}".format(moran_ret.I, moran_ret.p_sim) if MORANS_AVAILABLE and moran_ret else "not computed"}
  Sightings:           {"I = {:.4f}, p = {:.4f}".format(moran_sight.I, moran_sight.p_sim) if MORANS_AVAILABLE and moran_sight else "not computed"}

LOCAL SPATIAL REGRESSION (GWR-lite)
  Mean coefficient:    {local_coeffs.mean():.4f}
  Coefficient range:   [{local_coeffs.min():.4f}, {local_coeffs.max():.4f}]

MACHINE LEARNING
  Random Forest:       AUC = {rf_auc:.4f}, CV AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  Gradient Boosting:   RMSE = {gbr_rmse:.4f}, R² = {gbr_r2:.4f}
  Neural Network:      AUC = {nn_auc:.4f}
  DBSCAN clusters:     {n_sight_clusters} sighting, {n_part_clusters} particle
  GMM components:      {best_n}

SENSITIVITY (Spearman ρ vs diffusivity)
""")
for k, v in sensitivity_results.items():
    print(f"  {k:>12s}:  ρ = {v:.4f}")

print(f"""
FIGURES SAVED TO: {os.path.abspath(FIG_DIR)}/
  01  Retention vs Sightings (3-panel)
  02  Ensemble Uncertainty (mean + std)
  03  KDE Contour Maps (particles vs sightings)
  04  Permutation Test Histogram
  05  Bootstrap CI Distribution
  06  Random Forest Feature Importance
  07  ROC Curves
  08  Confusion Matrix
  09  GBR Residuals & Actual vs Predicted
  10  Feature Correlation Matrix
  11  SHAP Summary Plot {"✓" if SHAP_AVAILABLE else "(skipped — install shap)"}
  12  DBSCAN Clusters
  13  GMM Retention Zones
  14  3D Retention Surface
  15  Particle Animation GIF
  16  Interactive Plotly Map {"✓" if os.path.exists(os.path.join(FIG_DIR, '16_interactive_map.html')) else "(skipped — install plotly)"}
  17  Sample Trajectories over Retention
  18  Seasonal Retention Comparison
  19  Sensitivity Analysis Bar Chart

{'='*60}
  ANALYSIS COMPLETE
{'='*60}
""")
