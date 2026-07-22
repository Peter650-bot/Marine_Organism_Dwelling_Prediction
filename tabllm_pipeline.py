#!/usr/bin/env python3
"""
tabllm_pipeline.py — in-context TabLLM dwelling prediction, reusing the existing
retention pipeline.

Each per-grid-cell feature row (retention_core.FEATURES) is serialized to a fixed
natural-language "Text Template" string (Hegselmann et al. 2210.10723), Claude
classifies the cell HIGH/LOW dwelling, and the elicited probability is the
continuous score for ROC-AUC — the prompt-only analogue of the RF/GBT baselines
in species_pipeline.cv_scores.

This module reuses retention_core (rc) for everything physical:
  load_velocity / load_sightings / build_features / spatial_groups
so the Caretta/Mediterranean case is identical to the canonical pipeline
(same retention .npy cache, same median-of-nonzero y_cls threshold, same
3-degree spatial blocks).

The serialization, exemplar-sampling, subsampling and leakage-assertion helpers
depend only on numpy + stdlib, so they are importable and unit-testable without
pandas / scikit-learn / the anthropic SDK / an API key. Heavy and networked
pieces (rc, sklearn, matplotlib, anthropic_tabllm) are imported lazily.

Headline configuration (see plan jiggly-nibbling-hopcroft):
  spatial-block CV + prior-free system prompt + the `georegion_blind` ablation —
  the only honest analogues of the ablation_study.py "retention adds ~nothing net
  of geography" finding. Always report random AND spatial-block CV side by side.

CLI:
    python tabllm_pipeline.py --selftest          # offline, free
    python tabllm_pipeline.py --species "Caretta caretta" \
        --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv \
        --smoke                                   # cheap real run (~$0.30, hits API)
"""

import argparse
import json
import os

import numpy as np

# Mirror of retention_core.FEATURES — duplicated so this file imports without rc.
FEATURES = ["retention", "mean_speed", "vorticity", "latitude", "longitude",
            "dist_to_coast"]
IDX = {f: i for i, f in enumerate(FEATURES)}
SHOTS = [0, 4, 8, 16, 32, 64]
VARIANTS = ["full", "geo_blind", "georegion_blind", "ret_ablated"]
SEED = 42

# species -> (common name, latin name, region label)
SPECIES_META = {
    "Caretta caretta": ("loggerhead sea turtle", "Caretta caretta",
                        "Mediterranean Sea"),
    "Cetorhinus maximus": ("basking shark", "Cetorhinus maximus",
                           "Northeast Atlantic"),
}

# A-priori qualitative bins — fixed constants, documented as analyst priors,
# NOT derived from labels (deriving from labels would be a soft target leak).
RETENTION_BINS = [(0.10, "very low"), (0.30, "low"), (0.55, "moderate"),
                  (0.80, "high"), (1.01, "very high")]
SPEED_BINS_CMS = [(5, "weak"), (20, "moderate flow"), (50, "strong"),
                  (1e9, "very strong")]
COAST_BINS_KM = [(20, "coastal"), (80, "continental shelf"),
                 (1e9, "open/offshore")]
VORT_SCALE = 1e-6  # vorticity shown as multiples of 1e-6 s^-1


# ----------------------------------------------------------------------------
#  Serialization
# ----------------------------------------------------------------------------
def _bin(value, table):
    for thresh, label in table:
        if value < thresh:
            return label
    return table[-1][1]


def _signed_lon(lon):
    """Map any longitude convention to [-180, 180] for legible E/W rendering."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def serialize_row(x, variant="full", land_present=True):
    """Serialize one 6-feature cell vector to the Text-Template string.

    `variant` drops lines to mirror ablation_study.py:
      full            : all features
      geo_blind       : drop latitude + longitude lines
      georegion_blind : same lines as geo_blind (system prompt also strips region)
      ret_ablated     : drop the retention line

    If `land_present` is False the velocity cube has no land cells, so
    build_features set dist_to_coast=999 everywhere -> the coast line is a dead
    constant and is omitted.
    """
    ret = float(x[IDX["retention"]])
    spd_cms = round(float(x[IDX["mean_speed"]]) * 100)
    vort = float(x[IDX["vorticity"]])
    lat = float(x[IDX["latitude"]])
    lon = _signed_lon(x[IDX["longitude"]])
    coast_km = round(float(x[IDX["dist_to_coast"]]) * 111.0)

    drop_geo = variant in ("geo_blind", "georegion_blind")
    drop_ret = variant == "ret_ablated"

    vs = vort / VORT_SCALE
    if abs(vort) < 0.5 * VORT_SCALE:
        sense = "near-zero (weak rotation)"
    elif vort > 0:
        sense = "cyclonic/counterclockwise"
    else:
        sense = "anticyclonic/clockwise"

    lines = []
    if not drop_ret:
        lines.append(
            f"The ocean particle-retention score is {ret:.2f} on a 0 to 1 scale "
            f"({_bin(ret, RETENTION_BINS)})."
        )
    lines.append(
        f"The mean surface current speed is {spd_cms} cm/s "
        f"({_bin(spd_cms, SPEED_BINS_CMS)})."
    )
    lines.append(
        f"The local current rotation is {sense} "
        f"(relative vorticity {vs:+.1f}e-6 per second)."
    )
    if not drop_geo:
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        lines.append(f"The latitude is {abs(lat):.1f} degrees {ns}.")
        lines.append(f"The longitude is {abs(lon):.1f} degrees {ew}.")
    if land_present:
        lines.append(
            f"The distance to the nearest coast is {coast_km} km "
            f"({_bin(coast_km, COAST_BINS_KM)})."
        )
    return "\n".join(lines)


def build_system(species, variant, prevalence, prior=False):
    """System prompt. The hypothesis-injecting prior is EMPTY by default
    (prior-free is the headline). `georegion_blind` strips region/place names so
    the model cannot localize from region identity — the only fair analogue of
    RF's no-geography ablation."""
    common, latin, region = SPECIES_META.get(
        species, (species, species, "this ocean region")
    )
    blind = variant == "georegion_blind"
    if blind:
        who = "a marine megafauna species"
        where = "this ocean region"
    else:
        who = f"the {common} ({latin})"
        where = f"the {region}"

    prior_txt = ""
    if prior:
        # ablation-only; must NOT mention retention (would prime the hypothesis)
        prior_txt = (" Habitat use concentrates where physical ocean conditions"
                     " favour the species.")

    feats = []
    if variant != "ret_ablated":
        feats.append("a particle-retention score")
    feats += ["mean surface current speed", "current rotation (vorticity)"]
    if variant not in ("geo_blind", "georegion_blind"):
        feats.append("latitude and longitude")
    feats.append("distance to the nearest coast")
    feat_list = ", ".join(feats[:-1]) + ", and " + feats[-1]

    return (
        "You are a marine spatial-ecology model that predicts megafauna habitat "
        "use from oceanographic conditions at a single ocean grid cell. "
        f"The target species is {who} in {where}.{prior_txt}\n\n"
        f"Each location is described by physical ocean features: {feat_list}.\n\n"
        "Task: decide whether this location has HIGH or LOW habitat use for this "
        "species. \"HIGH\" means the location is in the upper half of suitable "
        "habitat; \"LOW\" the lower half. The base rate of HIGH locations in this "
        f"region is approximately {prevalence:.0%}. Reason from the ocean "
        "features, then give a probability that the location is HIGH."
    )


_QUERY_SUFFIX = "\n\nClassify this location (HIGH or LOW habitat use)."


def _user_turn(x, variant, land_present):
    return serialize_row(x, variant, land_present) + _QUERY_SUFFIX


def build_messages(X, exemplar_idx, query_i, variant, land_present):
    """Few-shot message list: label-only assistant demos (no fabricated p_high,
    which would anchor the model's output) followed by the query cell."""
    msgs = []
    for ei in exemplar_idx:
        msgs.append({"role": "user",
                     "content": _user_turn(X[ei], variant, land_present)})
        # label-only demonstration; the label is provided by the caller via y_cls
        msgs.append({"role": "assistant",
                     "content": json.dumps({"label": exemplar_idx[ei]})})
    msgs.append({"role": "user",
                 "content": _user_turn(X[query_i], variant, land_present)})
    return msgs


# ----------------------------------------------------------------------------
#  Leakage-safe sampling
# ----------------------------------------------------------------------------
def _seed_for(fold, scheme, species, k, variant):
    import hashlib
    h = hashlib.sha256(
        f"{fold}|{scheme}|{species}|{k}|{variant}".encode()
    ).hexdigest()[:8]
    return int(h, 16)


def sample_exemplars(X, y_cls, train_idx, k, seed):
    """Class-balanced exemplars drawn ONLY from this fold's train_idx. Returns an
    ordered dict {cell_index: "high"|"low"} (label-only demos). Under spatial-block
    CV, train_idx excludes every cell in held-out blocks, so no exemplar shares a
    block with any test cell."""
    if k == 0:
        return {}
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    pos = train_idx[y_cls[train_idx] == 1]
    neg = train_idx[y_cls[train_idx] == 0]
    kp, kn = k // 2, k - k // 2
    pick_pos = rng.choice(pos, min(kp, len(pos)), replace=False) if len(pos) else np.array([], int)
    pick_neg = rng.choice(neg, min(kn, len(neg)), replace=False) if len(neg) else np.array([], int)
    pick = np.concatenate([pick_pos, pick_neg]).astype(int)
    rng.shuffle(pick)  # interleave H/L to suppress position/recency bias
    return {int(i): ("high" if y_cls[i] == 1 else "low") for i in pick}


def subsample_test(test_idx, y_cls, n=800, mode="balanced", seed=SEED):
    """Subsample test cells AFTER the fold split, from test_idx only.
    mode="balanced": n/2 HIGH + n/2 LOW (AUC is prevalence-insensitive).
    mode="prevalence": preserve the fold's class balance (for F1 / ECE).
    No silent truncation — caller logs n_dropped."""
    rng = np.random.default_rng(seed)
    test_idx = np.asarray(test_idx)
    pos = test_idx[y_cls[test_idx] == 1]
    neg = test_idx[y_cls[test_idx] == 0]
    if mode == "balanced":
        half = n // 2
        kp = rng.choice(pos, min(half, len(pos)), replace=False) if len(pos) else np.array([], int)
        kn = rng.choice(neg, min(half, len(neg)), replace=False) if len(neg) else np.array([], int)
        kept = np.concatenate([kp, kn]).astype(int)
    else:  # prevalence-preserving
        if len(test_idx) <= n:
            kept = test_idx.copy()
        else:
            kept = rng.choice(test_idx, n, replace=False)
    rng.shuffle(kept)
    return kept


def assert_no_leak(exemplar_idx, test_idx, groups=None):
    """Hard leakage assertions, run every fold."""
    ex = np.asarray(list(exemplar_idx), dtype=int) if len(exemplar_idx) else np.array([], int)
    te = np.asarray(test_idx, dtype=int)
    assert np.intersect1d(ex, te).size == 0, "exemplar/test index overlap"
    if groups is not None and ex.size:
        assert set(groups[ex]).isdisjoint(set(groups[te])), \
            "exemplar and test share a spatial block"


# ----------------------------------------------------------------------------
#  Data loading (reuses retention_core; lazy heavy imports)
# ----------------------------------------------------------------------------
def load_cells(species, velocity_nc, species_csv, fig_dir="figures",
               window_years=2, min_year=1993, max_steps=365):
    """Reuse retention_core to get the per-cell matrix, the SAME y_cls threshold
    and spatial groups as species_pipeline. Reuses the retention .npy cache that
    species_pipeline writes (no resimulation)."""
    import retention_core as rc
    tag = species.replace(" ", "_").replace("/", "_")
    out_dir = os.path.join(fig_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    vel = rc.load_velocity(velocity_nc)
    cache = os.path.join(
        out_dir,
        f".retention_{os.path.basename(velocity_nc)}_{vel['u'].shape}_{max_steps}.npy",
    )
    if os.path.exists(cache):
        retention = np.load(cache)
    else:
        interp, _, _ = rc.make_interp(vel["u"], vel["v"], vel["lon"], vel["lat"])
        retention, _, _ = rc.simulate_retention(
            vel["lon"], vel["lat"], len(vel["time"]), interp, max_steps=max_steps
        )
        np.save(cache, retention)

    df_med = rc.load_sightings(species_csv, species, vel,
                               window_years=window_years, min_year=min_year)
    X, y_reg, lats, lons, fields = rc.build_features(vel, retention, df_med)
    thr = np.percentile(y_reg[y_reg > 0], 50) if (y_reg > 0).any() else 0.01
    y_cls = (y_reg > thr).astype(int)
    groups = rc.spatial_groups(lats, lons)
    land_present = bool(np.isnan(vel["u"][0]).any())
    prevalence = float(y_cls.mean())
    return {
        "species": species, "out_dir": out_dir, "vel": vel, "fields": fields,
        "X": X, "y_reg": y_reg, "y_cls": y_cls, "lats": lats, "lons": lons,
        "groups": groups, "n_sightings": int(len(df_med)),
        "land_present": land_present, "prevalence": prevalence,
        "n_cells": int(len(X)), "n_blocks": int(len(np.unique(groups))),
    }


# ----------------------------------------------------------------------------
#  Tabular baselines on the SAME k exemplars + identical kept indices
# ----------------------------------------------------------------------------
def _baseline_scores(X, y_cls, exemplar_idx, full_train_idx, kept_idx, seed=SEED):
    """Fit RF / GBT / LR / kNN on the k exemplars and a full-data RF ceiling,
    all scored on the identical kept indices. Scale-sensitive models (LR/kNN) get
    a StandardScaler fit on exemplar rows only."""
    from sklearn.ensemble import (RandomForestClassifier,
                                   GradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    out = {}
    ex = np.asarray(list(exemplar_idx), dtype=int) if len(exemplar_idx) else np.array([], int)
    Xk, yk = X[kept_idx], y_cls[kept_idx]

    def _proba(model, Xtr, ytr, Xte):
        model.fit(Xtr, ytr)
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xte), float(ytr[0]))
        return model.predict_proba(Xte)[:, 1]

    if ex.size and len(np.unique(y_cls[ex])) == 2:
        Xtr, ytr = X[ex], y_cls[ex]
        out["rf_kshot"] = _proba(
            RandomForestClassifier(n_estimators=200, max_depth=10,
                                   class_weight="balanced", random_state=seed),
            Xtr, ytr, Xk)
        out["gbt_kshot"] = _proba(
            GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, random_state=seed),
            Xtr, ytr, Xk)
        sc = StandardScaler().fit(Xtr)  # exemplar-only scaler (no leak)
        out["lr_kshot"] = _proba(
            LogisticRegression(class_weight="balanced", max_iter=1000),
            sc.transform(Xtr), ytr, sc.transform(Xk))
        out["knn_kshot"] = _proba(
            KNeighborsClassifier(n_neighbors=min(5, len(ex))),
            sc.transform(Xtr), ytr, sc.transform(Xk))

    # full-data RF ceiling, scored on the SAME kept indices
    ft = np.asarray(full_train_idx, dtype=int)
    if len(np.unique(y_cls[ft])) == 2:
        out["rf_full"] = _proba(
            RandomForestClassifier(n_estimators=200, max_depth=10,
                                   class_weight="balanced", random_state=seed),
            X[ft], y_cls[ft], Xk)
    return out, yk


# ----------------------------------------------------------------------------
#  Driver
# ----------------------------------------------------------------------------
def run_curve(cells, model=None, shots=None, variants=("full", "georegion_blind"),
              schemes=("random", "spatial"), n_test=800, n_vote=1,
              use_batch=True, cache_path=None, verbose=True,
              base_url=None, api_key=None, max_retries=4, error_abort=25):
    """Full few-shot learning curve. Writes <out_dir>/tabllm_results.json and
    figures 21/22. Hits an LLM backend: the Anthropic API by default, or a free
    self-hosted vLLM server when base_url (or $VLLM_BASE_URL) is set."""
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import roc_auc_score, f1_score
    import anthropic_tabllm as at

    X, y_cls, groups = cells["X"], cells["y_cls"], cells["groups"]
    species, out_dir = cells["species"], cells["out_dir"]
    land_present, prevalence = cells["land_present"], cells["prevalence"]
    shots = SHOTS if shots is None else shots
    model = model or at.HAIKU
    cache_path = cache_path or os.path.join(out_dir, "tabllm_cache.sqlite")
    client = at.TabLLMClient(model=model, cache_path=cache_path, n_vote=n_vote,
                             base_url=base_url, api_key=api_key,
                             max_retries=max_retries, error_abort=error_abort)

    def log(*a):
        if verbose:
            print(*a)

    def folds(scheme):
        if scheme == "random":
            skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
            return list(skf.split(X, y_cls))
        ng = len(np.unique(groups))
        gkf = GroupKFold(min(5, ng))
        return list(gkf.split(X, y_cls, groups))

    results = {"species": species, "model": model, "prevalence": prevalence,
               "n_cells": cells["n_cells"], "land_present": land_present,
               "n_vote": n_vote, "points": [], "dropped_log": []}

    for scheme in schemes:
        for variant in variants:
            for k in shots:
                auc_runs, base_runs, f1_runs = [], {}, []
                for fold, (tr, te) in enumerate(folds(scheme)):
                    seed = _seed_for(fold, scheme, species, k, variant)
                    ex = sample_exemplars(X, y_cls, tr, k, seed)
                    assert_no_leak(ex, te, groups if scheme == "spatial" else None)
                    kept = subsample_test(te, y_cls, n=n_test, mode="balanced",
                                          seed=seed)
                    results["dropped_log"].append(
                        {"scheme": scheme, "variant": variant, "k": k,
                         "fold": fold, "n_test_full": int(len(te)),
                         "n_test_used": int(len(kept)),
                         "n_dropped": int(len(te) - len(kept))})
                    # LLM scores for kept cells
                    jobs = [{"id": f"{scheme}:{variant}:{k}:{fold}:{i}",
                             "system": build_system(species, variant, prevalence),
                             "messages": build_messages(X, ex, i, variant,
                                                        land_present)}
                            for i in kept]
                    if use_batch:
                        res = client.classify_batch(jobs)
                    else:
                        res = {j["id"]: client.classify_one(j["system"],
                                                            j["messages"])
                               for j in jobs}
                    scores = np.array([res[j["id"]]["p_high"] for j in jobs])
                    yk = y_cls[kept]
                    if len(np.unique(yk)) == 2:
                        auc_runs.append(roc_auc_score(yk, scores))
                        f1_runs.append(f1_score(yk, (scores > 0.5).astype(int),
                                                zero_division=0))
                    # tabular baselines on identical kept indices
                    bsc, _ = _baseline_scores(X, y_cls, ex, tr, kept, seed)
                    for name, sc in bsc.items():
                        if len(np.unique(yk)) == 2:
                            base_runs.setdefault(name, []).append(
                                roc_auc_score(yk, sc))
                point = {
                    "scheme": scheme, "variant": variant, "k": k,
                    "auc_llm": float(np.mean(auc_runs)) if auc_runs else None,
                    "f1_llm": float(np.mean(f1_runs)) if f1_runs else None,
                    "n_folds": len(auc_runs),
                }
                for name, runs in base_runs.items():
                    point[f"auc_{name}"] = float(np.mean(runs)) if runs else None
                results["points"].append(point)
                log(f"[{scheme}/{variant}] k={k:>2}  AUC_llm="
                    f"{point['auc_llm']}  calls={client.calls_made} "
                    f"hits={client.cache_hits}")

    results["api_calls"] = client.calls_made
    results["cache_hits"] = client.cache_hits
    with open(os.path.join(out_dir, "tabllm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"[saved] {out_dir}/tabllm_results.json")
    try:
        _make_figures(results, out_dir)
    except Exception as e:  # figures are best-effort
        log(f"figure generation skipped: {e}")
    return results


def _make_figures(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = results["points"]
    # Figure 21: learning curve (per scheme panel), variant 'full'
    schemes = sorted({p["scheme"] for p in pts})
    fig, axes = plt.subplots(1, len(schemes), figsize=(7 * len(schemes), 5),
                             squeeze=False)
    series_keys = ["auc_llm", "auc_rf_kshot", "auc_gbt_kshot", "auc_lr_kshot",
                   "auc_knn_kshot"]
    for ax, scheme in zip(axes[0], schemes):
        sub = [p for p in pts if p["scheme"] == scheme and p["variant"] == "full"]
        sub = sorted(sub, key=lambda p: p["k"])
        ks = [p["k"] for p in sub]
        for key in series_keys:
            ys = [p.get(key) for p in sub]
            if any(y is not None for y in ys):
                ax.plot(ks, ys, marker="o", label=key.replace("auc_", ""))
        ceil = [p.get("auc_rf_full") for p in sub]
        ceil = [c for c in ceil if c is not None]
        if ceil:
            ax.axhline(float(np.mean(ceil)), ls="--", color="k",
                       label="rf_full (ceiling)")
        ax.axhline(0.5, ls=":", color="grey")
        ax.set_xlabel("shots (k)"); ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{results['species']} — {scheme} CV"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "21_tabllm_learning_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 22: geography ablation (spatial CV, AUC vs variant at each k)
    spat = [p for p in pts if p["scheme"] == "spatial"]
    variants = [v for v in VARIANTS if any(p["variant"] == v for p in spat)]
    if variants:
        fig, ax = plt.subplots(figsize=(8, 5))
        ks = sorted({p["k"] for p in spat})
        for v in variants:
            ys = [next((p["auc_llm"] for p in spat
                        if p["variant"] == v and p["k"] == k), None) for k in ks]
            ax.plot(ks, ys, marker="o", label=v)
        ax.axhline(0.5, ls=":", color="grey")
        ax.set_xlabel("shots (k)"); ax.set_ylabel("ROC-AUC (spatial CV)")
        ax.set_title(f"{results['species']} — geography ablation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "22_tabllm_geography_ablation.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------
#  Offline self-test (numpy + stdlib only)
# ----------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    # synthetic cell: retention .42, speed .18 m/s, vort -3.1e-6, lat 36.5, lon 18.2, dist 47/111
    x = np.array([0.42, 0.18, -3.1e-6, 36.5, 18.2, 47 / 111.0])
    s_full = serialize_row(x, "full", True)
    assert "particle-retention score is 0.42" in s_full
    assert "18 cm/s (moderate flow)" in s_full
    assert "anticyclonic/clockwise" in s_full
    assert "36.5 degrees N" in s_full and "18.2 degrees E" in s_full
    assert "47 km (continental shelf)" in s_full
    # geo_blind drops lat/lon; ret_ablated drops retention; no-land drops coast
    assert "latitude" not in serialize_row(x, "geo_blind", True)
    assert "retention" not in serialize_row(x, "ret_ablated", True)
    assert "coast" not in serialize_row(x, "full", False)
    # line order matches FEATURES order (auditability)
    assert s_full.index("retention") < s_full.index("speed") \
        < s_full.index("rotation") < s_full.index("latitude") \
        < s_full.index("longitude") < s_full.index("coast")

    # system prompt: georegion_blind strips region/species
    sys_full = build_system("Caretta caretta", "full", 0.18)
    assert "loggerhead" in sys_full and "Mediterranean" in sys_full
    assert "18%" in sys_full  # base-rate stated
    sys_blind = build_system("Caretta caretta", "georegion_blind", 0.18)
    assert "loggerhead" not in sys_blind and "Mediterranean" not in sys_blind

    # exemplar sampling: train-only, balanced, label-only, leakage-safe
    N = 300
    X = rng.normal(size=(N, 6))
    y = (rng.random(N) < 0.3).astype(int)
    groups = (np.arange(N) // 30)  # 10 blocks of 30
    tr = np.arange(0, 210)          # blocks 0-6
    te = np.arange(210, 300)        # blocks 7-9 (disjoint)
    ex = sample_exemplars(X, y, tr, 16, seed=123)
    assert len(ex) <= 16 and set(ex).issubset(set(tr.tolist()))
    assert set(ex.values()).issubset({"high", "low"})
    assert_no_leak(ex, te, groups)  # must not raise
    # leak detection fires when exemplars overlap test
    try:
        assert_no_leak({int(te[0]): "high"}, te, groups)
        raise RuntimeError("leak assertion failed to fire")
    except AssertionError:
        pass

    # subsample: balanced returns ~equal classes
    kept = subsample_test(np.arange(N), y, n=40, mode="balanced", seed=1)
    nb = int(y[kept].sum())
    assert abs(nb - (len(kept) - nb)) <= 1

    # messages: label-only assistant turns, query last
    msgs = build_messages(X, ex, query_i=215, variant="full", land_present=True)
    assert msgs[-1]["role"] == "user"
    assistant = [m for m in msgs if m["role"] == "assistant"]
    assert all("p_high" not in m["content"] for m in assistant)  # no anchoring
    assert len(assistant) == len(ex)
    print("tabllm_pipeline self-test OK "
          "(serialize, system, exemplars, leakage, subsample, messages)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run offline unit checks (no SDK / no API)")
    ap.add_argument("--species")
    ap.add_argument("--velocity")
    ap.add_argument("--species-csv")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="LLM id. For vLLM, the served model name "
                         "(e.g. unsloth/Llama-3.2-3B-Instruct)")
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"),
                    help="OpenAI-compatible endpoint of a self-hosted vLLM server "
                         "(e.g. https://<tunnel>/v1). Default $VLLM_BASE_URL. When "
                         "set, inference is FREE via vLLM instead of the paid API.")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"),
                    help="API key for the vLLM/OpenAI endpoint (default $VLLM_API_KEY "
                         "or 'EMPTY'; vLLM ignores it unless started with one)")
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--n-vote", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=4,
                    help="per-request retries before a call is treated as failed. "
                         "Lower it (e.g. 1) so a dead vLLM server fails fast "
                         "instead of grinding retry-backoff through the whole grid.")
    ap.add_argument("--error-abort", type=int, default=25,
                    help="circuit-breaker: abort a batched run after this many "
                         "failed requests (server likely down), leaving the cache "
                         "clean. 0 disables.")
    ap.add_argument("--probe", action="store_true",
                    help="load cells and report prevalence/land/counts, then stop "
                         "(no API calls, free)")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap real run: full variant, random CV, shots {0,8}, "
                         "50 cells, n_vote=1 (~$0.30, HITS API)")
    ap.add_argument("--shots",
                    help="comma-separated shot counts to override the default "
                         f"{SHOTS} (e.g. '4,8,16,32,64' to drop the 0-shot point)")
    ap.add_argument("--schemes",
                    help="comma-separated CV schemes to run, subset of "
                         "{random,spatial} (default: both)")
    ap.add_argument("--variants",
                    help=f"comma-separated serialization variants, subset of "
                         f"{list(VARIANTS)} (default: full,georegion_blind)")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if not (args.species and args.velocity and args.species_csv):
        ap.error("provide --selftest, or --species/--velocity/--species-csv")

    cells = load_cells(args.species, args.velocity, args.species_csv,
                       fig_dir=args.fig_dir)
    print(f"species={args.species} n_cells={cells['n_cells']} "
          f"prevalence={cells['prevalence']:.3f} land_present={cells['land_present']} "
          f"n_blocks={cells['n_blocks']} n_sightings={cells['n_sightings']}")
    if args.probe:
        # Free validation of the serialization on a real cell (no API).
        ex_row = cells["X"][int(np.argmax(cells["y_cls"]))]
        print("--- sample serialization (full) ---")
        print(serialize_row(ex_row, "full", cells["land_present"]))
        print("--- system prompt (full, prior-free) ---")
        print(build_system(args.species, "full", cells["prevalence"]))
        return 0
    if args.smoke:
        run_curve(cells, model=args.model, shots=[0, 8], variants=("full",),
                  schemes=("random",), n_test=50, n_vote=1, use_batch=False,
                  base_url=args.base_url, api_key=args.api_key,
                  max_retries=args.max_retries, error_abort=args.error_abort)
    else:
        kw = {}
        if args.shots:
            kw["shots"] = [int(s) for s in args.shots.split(",") if s.strip() != ""]
        if args.schemes:
            kw["schemes"] = tuple(s.strip() for s in args.schemes.split(",")
                                  if s.strip())
        if args.variants:
            kw["variants"] = tuple(v.strip() for v in args.variants.split(",")
                                   if v.strip())
        run_curve(cells, model=args.model, n_test=args.n_test, n_vote=args.n_vote,
                  base_url=args.base_url, api_key=args.api_key,
                  max_retries=args.max_retries, error_abort=args.error_abort, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
