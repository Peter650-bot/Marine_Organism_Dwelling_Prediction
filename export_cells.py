#!/usr/bin/env python3
"""
export_cells.py — freeze one species' per-cell matrix and its exact CV split
into a small .npz, so a GPU run elsewhere scores the SAME cells as the
published CPU benchmark.

WHY
    `deploy/tabicl_finetune_modal.py` runs on Modal, which has no access to the
    600 MB velocity cubes and cannot rebuild the retention field. Re-deriving
    the folds remotely would risk a split that merely resembles the published
    one. Instead the split is computed here, verified against the committed
    artefact, and shipped as data (< 1 MB).

WHAT IT VERIFIES
    Before writing, every fold is checked against
    `figures/<species>/tabicl_f1_spatial.json`:
      * the per-fold seed matches `tabllm_pipeline._seed_for`,
      * the training-fold size matches `n_context`,
      * all 300 balanced test labels match `y_true`, element for element.
    Any mismatch aborts. So a successful export is proof that the exported
    split IS the published split, not an approximation of it.

CONTENTS
    X (n_cells, 6) float32, y (n_cells,) int64, groups (n_cells,)  -- 3-deg blocks
    tr{0..4}   training-fold cell indices    (in-context rows)
    kept{0..4} the 300 balanced test cells scored that fold

USAGE
    python export_cells.py --species "Caretta caretta"
    python export_cells.py --all
"""

import argparse
import json
import os

import numpy as np

OUT_DIR = "figures"
N_TEST = 300


def export(species, velocity, species_csv, fig_root=OUT_DIR, n_test=N_TEST):
    from sklearn.model_selection import GroupKFold
    import tabllm_pipeline as tp

    tag = species.replace(" ", "_")
    cells = tp.load_cells(species, velocity, species_csv, fig_dir=fig_root)
    X, y, groups = cells["X"], cells["y_cls"], cells["groups"]
    n_blocks = len(np.unique(groups))
    folds = list(GroupKFold(min(5, n_blocks)).split(X, y, groups))

    ref_path = os.path.join(fig_root, tag, "tabicl_f1_spatial.json")
    if not os.path.exists(ref_path):
        raise SystemExit(
            f"no published artefact at {ref_path} to verify against; run\n"
            f'    python tabicl_f1_benchmark.py --species "{species}"\n'
            "first, so the exported split can be proven identical to it.")
    ref = json.load(open(ref_path))["folds"]

    tr_list, kept_list = [], []
    for i, (tr, te) in enumerate(folds):
        seed = tp._seed_for(i, "spatial", species, -1, "full")
        kept = tp.subsample_test(te, y, n=n_test, mode="balanced", seed=seed)
        if seed != ref[i]["seed"]:
            raise SystemExit(f"fold {i}: seed {seed} != published {ref[i]['seed']}")
        if len(tr) != ref[i]["n_context"]:
            raise SystemExit(f"fold {i}: {len(tr)} context rows != "
                             f"published {ref[i]['n_context']}")
        if list(y[kept]) != ref[i]["y_true"]:
            raise SystemExit(f"fold {i}: exported test labels differ from published")
        tr_list.append(tr)
        kept_list.append(kept)

    out = os.path.join(fig_root, tag, "cells_export.npz")
    np.savez_compressed(
        out, X=X.astype(np.float32), y=y.astype(np.int64), groups=groups,
        **{f"tr{i}": a for i, a in enumerate(tr_list)},
        **{f"kept{i}": a for i, a in enumerate(kept_list)})
    print(f"[saved] {out}  ({os.path.getsize(out) / 1e6:.2f} MB)")
    print(f"  {X.shape[0]} cells, prevalence {y.mean():.4f}, {n_blocks} blocks")
    print(f"  context rows/fold {[len(a) for a in tr_list]}")
    print(f"  verified identical to {ref_path}")
    return out


def main(argv=None):
    import tabicl_f1_benchmark as tf

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default=tf.DEFAULTS["species"])
    ap.add_argument("--velocity")
    ap.add_argument("--species-csv")
    ap.add_argument("--fig-root", default=OUT_DIR)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--all", action="store_true",
                    help="export every species that has a preset")
    args = ap.parse_args(argv)

    todo = ([tf.DEFAULTS["species"]] + list(tf.SPECIES_PRESETS)
            if args.all else [args.species])
    for sp in todo:
        pre = tf.SPECIES_PRESETS.get(sp, {})
        vel = args.velocity or pre.get("velocity", tf.DEFAULTS["velocity"])
        csv = args.species_csv or pre.get("species_csv", tf.DEFAULTS["species_csv"])
        for path, what in ((vel, "velocity cube"), (csv, "occurrence CSV")):
            if not os.path.exists(path):
                raise SystemExit(f"{what} not found: {path}")
        print(f"\n=== {sp} ===")
        export(sp, vel, csv, args.fig_root, args.n_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
