#!/usr/bin/env python3
"""
tabllm_prompt.py — TabLLM serialization and prompting for dwelling prediction.

Each per-grid-cell feature row (retention_core.FEATURES) is serialized to a fixed
natural-language "Text Template" string (Hegselmann et al. 2210.10723); a language
model classifies the cell HIGH/LOW dwelling, and the elicited probability is the
continuous score for ROC-AUC — the prompt-only analogue of the tabular baselines.

This module owns the *prompt*: serialization variants, the system prompt, few-shot
message construction, exemplar sampling and the leakage assertions. `run_tabllm.py`
drives it; `label_cv.py` supplies labels and folds.

`load_cells` reuses retention_core for everything physical (load_velocity /
load_sightings / build_features) and caches the retention field, so cells are
built once and shared by every model in the pipeline. It deliberately does NOT
define a label — the label is the KDE core area in `label_cv.py`.

The serialization, exemplar-sampling, subsampling and leakage-assertion helpers
depend only on numpy + stdlib, so they are importable and unit-testable without
pandas / scikit-learn / an SDK / an API key. Heavy imports are lazy.

CLI (both free, no API calls):
    python tabllm_prompt.py --selftest
    python tabllm_prompt.py --probe --species "Caretta caretta" \
        --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv
"""


import argparse
import json
import os

import numpy as np

# Mirror of retention_core.FEATURES — duplicated so this file imports without rc.
FEATURES = ["retention", "mean_speed", "vorticity", "latitude", "longitude",
            "dist_to_coast"]
IDX = {f: i for i, f in enumerate(FEATURES)}
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


def build_system(species, variant, prevalence, prior=False,
                 scoring="verbalized"):
    """System prompt. The hypothesis-injecting prior is EMPTY by default
    (prior-free is the headline). `georegion_blind` strips region/place names so
    the model cannot localize from region identity — the only fair analogue of
    RF's no-geography ablation.

    `scoring` changes only the ANSWER FORMAT instruction, never the evidence:
      verbalized : model writes {label, p_high}  (JSON, guided decoding)
      logprob    : model answers with the bare word HIGH or LOW, and the score
                   is read from the token distribution (TabLLM's actual method).
                   Asking for prose here would push the verbalizer away from the
                   first token position, so the instruction is emphatic."""
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

    if scoring == "logprob":
        answer = ("Answer with exactly one word — either HIGH or LOW — and "
                  "nothing else. Do not explain, do not add punctuation, and do "
                  "not restate the features. Your entire reply must be that one "
                  "word.")
    else:
        answer = ("Reason from the ocean features, then give a probability that "
                  "the location is HIGH.")

    return (
        "You are a marine spatial-ecology model that predicts megafauna habitat "
        "use from oceanographic conditions at a single ocean grid cell. "
        f"The target species is {who} in {where}.{prior_txt}\n\n"
        f"Each location is described by physical ocean features: {feat_list}.\n\n"
        "Task: decide whether this location has HIGH or LOW habitat use for this "
        "species. \"HIGH\" means the location is in the upper half of suitable "
        "habitat; \"LOW\" the lower half. The base rate of HIGH locations in this "
        f"region is approximately {prevalence:.0%}. {answer}"
    )


_QUERY_SUFFIX = "\n\nClassify this location (HIGH or LOW habitat use)."


def _user_turn(x, variant, land_present):
    return serialize_row(x, variant, land_present) + _QUERY_SUFFIX


def build_messages(X, exemplar_idx, query_i, variant, land_present,
                   scoring="verbalized"):
    """Few-shot message list: label-only assistant demos (no fabricated p_high,
    which would anchor the model's output) followed by the query cell.

    Under logprob scoring the demos are the bare verbalizer word, so the
    few-shot pattern the model continues is exactly the single token whose
    distribution we read."""
    msgs = []
    for ei in exemplar_idx:
        msgs.append({"role": "user",
                     "content": _user_turn(X[ei], variant, land_present)})
        # label-only demonstration; the label is provided by the caller via y_cls
        demo = (exemplar_idx[ei].upper() if scoring == "logprob"
                else json.dumps({"label": exemplar_idx[ei]}))
        msgs.append({"role": "assistant", "content": demo})
    msgs.append({"role": "user",
                 "content": _user_turn(X[query_i], variant, land_present)})
    return msgs


# ----------------------------------------------------------------------------
#  Leakage-safe sampling
# ----------------------------------------------------------------------------
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
    """Build the per-cell feature matrix, reusing the retention .npy cache.

    Returns the physical quantities only. The label is NOT defined here — the
    pipeline derives it from the KDE core area (`label_cv.core_area_label`), so
    every model is scored against the same ground truth.
    """
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
    land_present = bool(np.isnan(vel["u"][0]).any())
    return {
        "species": species, "out_dir": out_dir, "vel": vel, "fields": fields,
        "X": X, "y_reg": y_reg, "lats": lats, "lons": lons,
        "n_sightings": int(len(df_med)), "land_present": land_present,
        "n_cells": int(len(X)),
    }



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

    # --- logprob scoring: answer-format instruction + bare-word demos --------
    sys_lp = build_system("Caretta caretta", "full", 0.18, scoring="logprob")
    assert "exactly one word" in sys_lp
    assert "give a probability" not in sys_lp   # prose request must be gone
    assert "18%" in sys_lp                      # evidence/base rate unchanged
    # the serialized EVIDENCE must be identical across scoring modes — only the
    # answer format may differ, else the two runs aren't comparable
    assert (build_system("Caretta caretta", "full", 0.18).split("Task:")[0]
            == sys_lp.split("Task:")[0])
    msgs_lp = build_messages(X, ex, 215, "full", True, scoring="logprob")
    demos = [m["content"] for m in msgs_lp if m["role"] == "assistant"]
    assert demos and all(d in ("HIGH", "LOW") for d in demos), demos
    assert msgs_lp[-1]["role"] == "user"
    # user turns carry the same evidence in both modes
    assert ([m["content"] for m in msgs_lp if m["role"] == "user"]
            == [m["content"] for m in msgs if m["role"] == "user"])
    print("tabllm_prompt self-test OK "
          "(serialize, system, exemplars, leakage, subsample, messages, logprob)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run offline unit checks (no SDK / no API)")
    ap.add_argument("--probe", action="store_true",
                    help="load cells and report counts/land, then stop (free)")
    ap.add_argument("--species")
    ap.add_argument("--velocity")
    ap.add_argument("--species-csv")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if not (args.species and args.velocity and args.species_csv):
        ap.error("provide --selftest, or --species/--velocity/--species-csv")

    cells = load_cells(args.species, args.velocity, args.species_csv, args.fig_dir)
    print(f"{cells['species']}: {cells['n_cells']} ocean cells, "
          f"{cells['n_sightings']} occurrence rows, "
          f"land_present={cells['land_present']}")
    if args.probe:
        i = int(np.argmax(cells["y_reg"]))
        print("\nsample serialization (densest cell):\n")
        print(serialize_row(cells["X"][i], "full", cells["land_present"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
