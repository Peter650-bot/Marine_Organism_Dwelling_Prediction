"""Fine-tune TabICLv2 on Caretta caretta, on a Modal GPU, under the published
spatial-block split.

WHY THIS EXISTS
    The published benchmark runs TabICL zero-shot: the training fold is passed
    as in-context data in a single forward pass, nothing is fitted. This app
    asks the separate question of what a *fine-tuned* TabICL scores on exactly
    the same folds -- so it is a ceiling measurement, not a replacement for the
    zero-shot number. Report the two side by side; fine-tuning forfeits the
    zero-training claim the paper's throughput argument rests on.

COMPARABILITY
    The fold indices and the 300 balanced test cells are NOT recomputed here.
    They are exported from the local run (`caretta_cells.npz`) after asserting
    that the seeds, train sizes and every test label match
    `figures/Caretta_caretta/tabicl_f1_spatial.json`. So the fine-tuned and
    zero-shot numbers differ only in the model, never in the data.

    Each container also re-scores the fold zero-shot, so every fold yields a
    paired (finetuned, zero-shot) result measured on the same GPU with the same
    weights loaded -- the delta is then free of any hardware or version drift
    against the CPU-run published numbers.

VALIDATION SPLIT
    Fine-tuning needs a validation set for early stopping. The built-in
    `validation_split_ratio` splits RANDOMLY, which in this project would leak
    across spatial autocorrelation and inflate the result -- the exact failure
    the paper is about. Instead the inner split holds out whole 3-degree blocks
    from the training fold, so validation is spatially disjoint from training,
    and the outer test fold is never touched.

COST
    A10G is Modal's strongest card-free tier (A100/H100 want a payment method).
    `TIME_LIMIT_S` caps each fold's fine-tuning wall-clock, so the worst case is
    bounded: 5 folds * (TIME_LIMIT_S + inference) at the A10G rate. Run the
    pilot (`--folds 0`) first and read the printed cost before the full sweep.

USAGE
    python export_cells.py --all                                   # once, locally
    modal run deploy/tabicl_finetune_modal.py --folds 0            # pilot, ~$0.01
    modal run deploy/tabicl_finetune_modal.py --folds 0,1,2,3,4 \
        --species "Cetorhinus maximus"                             # full, ~$0.15
"""
import json
import os

import modal

# Cells come from `export_cells.py`, which verifies the split against the
# committed `tabicl_f1_spatial.json` before writing -- so these files are the
# published split, not a remote re-derivation of it. Run first:
#     python export_cells.py --all
# Every species' cells ride along in the image (all three total < 2 MB), so one
# build serves every run and --species just picks which to read.
SPECIES = ("Caretta caretta", "Cetorhinus maximus", "Balaenoptera musculus")
SPECIES_CELLS = {
    sp: os.path.join("figures", sp.replace(" ", "_"), "cells_export.npz")
    for sp in SPECIES
}

_missing = [p for p in SPECIES_CELLS.values() if not os.path.exists(p)]
if _missing:
    raise SystemExit(
        "missing exported cells:\n  " + "\n  ".join(_missing)
        + "\n\nRun `python export_cells.py --all` from the repo root first "
          "(it needs the velocity cubes locally; Modal never sees them).")

GPU = "A10G"
GPU_HOURLY_USD = 1.10          # published A10G rate; used only for the estimate
TIME_LIMIT_S = 1500            # per-fold fine-tuning cap (early stopping may end sooner)
CONTAINER_TIMEOUT_S = 3600
EPOCHS = 30
LR = 1e-5
CKPT_V2 = "tabicl-classifier-v2-20260212.ckpt"
V2_MARKERS = ("col_target_aware", "col_ssmax", "icl_ssmax")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("tabicl[finetune]", "scikit-learn", "numpy<2")
    .add_local_file(SPECIES_CELLS["Caretta caretta"], "/data/Caretta caretta.npz")
    .add_local_file(SPECIES_CELLS["Cetorhinus maximus"], "/data/Cetorhinus maximus.npz")
    .add_local_file(SPECIES_CELLS["Balaenoptera musculus"],
                    "/data/Balaenoptera musculus.npz")  # noqa: E501
)

app = modal.App("tabicl-finetune-caretta")


def _inner_split(groups, train_idx, seed, val_frac=0.2):
    """Hold out whole spatial blocks from the training fold for validation."""
    import numpy as np
    rng = np.random.default_rng(seed)
    blocks = np.unique(groups[train_idx])
    n_val = max(1, int(round(val_frac * len(blocks))))
    val_blocks = set(rng.choice(blocks, n_val, replace=False).tolist())
    is_val = np.array([g in val_blocks for g in groups[train_idx]])
    return train_idx[~is_val], train_idx[is_val]


def _fingerprint(est):
    cfg = getattr(est, "model_config_", {}) or {}
    arch = "v2" if all(k in cfg for k in V2_MARKERS) else "v1"
    return {"architecture": arch,
            "config": {k: cfg.get(k) for k in
                       ("embed_dim", "icl_num_blocks", "max_classes") + V2_MARKERS}}


@app.function(image=image, gpu=GPU, timeout=CONTAINER_TIMEOUT_S)
def run_fold(args: tuple):
    import time
    import numpy as np
    import torch
    from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                                 recall_score)
    from tabicl import TabICLClassifier, FinetunedTabICLClassifier

    fold, species = args
    t0 = time.time()
    d = np.load(f"/data/{species}.npz")
    X, y, groups = d["X"].astype(np.float32), d["y"], d["groups"]
    tr, kept = d[f"tr{fold}"], d[f"kept{fold}"]
    yk = y[kept]

    inner_tr, inner_val = _inner_split(groups, tr, seed=1000 + fold)
    print(f"[{species} fold {fold}] gpu={torch.cuda.get_device_name(0)} "
          f"train={len(inner_tr)} val={len(inner_val)} test={len(kept)} "
          f"(val blocks disjoint from train)", flush=True)

    # ---- zero-shot control, same weights, same GPU -------------------------
    zs = TabICLClassifier(n_estimators=2, device="cuda", random_state=42,
                          checkpoint_version=CKPT_V2)
    zs.fit(X[tr], y[tr])
    fp = _fingerprint(zs)
    if fp["architecture"] != "v2":
        raise RuntimeError(f"expected TabICLv2 weights, got {fp}")
    p_zs = zs.predict_proba(X[kept])[:, list(zs.classes_).index(1)]
    t_zs = time.time() - t0
    print(f"[fold {fold}] zero-shot done in {t_zs:.0f}s  "
          f"AUC={roc_auc_score(yk, p_zs):.4f}", flush=True)

    # ---- fine-tune ---------------------------------------------------------
    t1 = time.time()
    ckpt_dir = f"/tmp/ft_fold{fold}"
    ft = FinetunedTabICLClassifier(
        epochs=EPOCHS, learning_rate=LR, amp=True,
        early_stopping=True, patience=8, time_limit=TIME_LIMIT_S,
        n_estimators_inference=2,          # match the zero-shot control
        checkpoint_version=CKPT_V2, device="cuda", random_state=42,
        verbose=True)
    ft.fit(X[inner_tr], y[inner_tr], X_val=X[inner_val], y_val=y[inner_val],
           output_dir=ckpt_dir)
    # Arm 2: tuned weights, but only the inner-train rows as context (8,390).
    p_ft = ft.predict_proba(X[kept])[:, list(ft.classes_).index(1)]
    t_ft = time.time() - t1

    # ---- Arm 3: tuned weights, FULL training fold as context ---------------
    # The confound in arm 2 is that fine-tuning had to carve a validation split
    # out of the training fold, so it also inferred from 23% fewer context rows
    # than zero-shot did. Reloading best.ckpt into the plain estimator and
    # fitting on the whole fold puts the context back, so arm 3 vs zero-shot
    # isolates the weights and arm 2 vs arm 3 isolates the context size.
    t2 = time.time()
    best = os.path.join(ckpt_dir, "best.ckpt")
    if not os.path.exists(best):
        raise RuntimeError(f"fine-tuning wrote no checkpoint at {best}")
    fc = TabICLClassifier(n_estimators=2, device="cuda", random_state=42,
                          model_path=best)
    fc.fit(X[tr], y[tr])
    fp_fc = _fingerprint(fc)
    if fp_fc["architecture"] != "v2":
        raise RuntimeError(f"fine-tuned ckpt is not v2: {fp_fc}")
    p_fc = fc.predict_proba(X[kept])[:, list(fc.classes_).index(1)]
    t_fc = time.time() - t2
    print(f"[fold {fold}] full-context reload {t_fc:.0f}s  "
          f"AUC={roc_auc_score(yk, p_fc):.4f}  (context {len(tr)} rows)",
          flush=True)

    row = {"fold": int(fold), "species": species, "n_train": int(len(inner_tr)),
           "n_val": int(len(inner_val)), "n_test": int(len(kept)),
           "n_pos_test": int(yk.sum()), "checkpoint": fp,
           "n_context_zeroshot": int(len(tr)), "n_context_finetuned": int(len(inner_tr)),
           "seconds_zeroshot": round(t_zs, 1), "seconds_finetune": round(t_ft, 1),
           "seconds_fullctx": round(t_fc, 1)}
    for name, p in (("finetuned", p_ft), ("finetuned_fullctx", p_fc),
                    ("zeroshot", p_zs)):
        yh = (p >= 0.5).astype(int)
        row[f"auc_{name}"] = float(roc_auc_score(yk, p))
        row[f"f1_{name}"] = float(f1_score(yk, yh, zero_division=0))
        row[f"precision_{name}"] = float(precision_score(yk, yh, zero_division=0))
        row[f"recall_{name}"] = float(recall_score(yk, yh, zero_division=0))
    print(f"[fold {fold}] AUC  ft={row['auc_finetuned']:.4f}  "
          f"ft-fullctx={row['auc_finetuned_fullctx']:.4f}  "
          f"zs={row['auc_zeroshot']:.4f}  |  F1  {row['f1_finetuned']:.4f}  "
          f"{row['f1_finetuned_fullctx']:.4f}  {row['f1_zeroshot']:.4f}", flush=True)
    return row


@app.local_entrypoint()
def main(folds: str = "0", species: str = "Caretta caretta", out: str = ""):
    if species not in SPECIES_CELLS:
        raise SystemExit(f"unknown species {species!r}; "
                         f"choose from {list(SPECIES_CELLS)}")
    out = out or (f"figures/{species.replace(' ', '_')}/"
                  f"tabicl_finetune_fullctx.json")
    want = [int(s) for s in folds.split(",") if s.strip() != ""]
    print(f"launching {len(want)} fold(s) of {species} on {GPU}: {want}")
    rows = sorted(run_fold.map([(f, species) for f in want]),
                  key=lambda r: r["fold"])

    secs = sum(r["seconds_zeroshot"] + r["seconds_finetune"]
               + r.get("seconds_fullctx", 0) for r in rows)
    cost = secs / 3600 * GPU_HOURLY_USD
    print(f"\nGPU seconds {secs:.0f}  ->  approx ${cost:.2f} at "
          f"${GPU_HOURLY_USD}/h on {GPU}")

    existing = {}
    if os.path.exists(out):
        existing = {r["fold"]: r for r in json.load(open(out))["folds"]}
    existing.update({r["fold"]: r for r in rows})
    merged = [existing[k] for k in sorted(existing)]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"species": species, "scheme": "spatial",
                   "model": "TabICLv2 fine-tuned", "gpu": GPU,
                   "epochs": EPOCHS, "learning_rate": LR,
                   "time_limit_s": TIME_LIMIT_S,
                   "threshold": 0.5, "folds": merged}, fh, indent=2)
    print(f"[saved] {out}")

    for r in merged:
        print(f"  fold {r['fold']}: AUC ft={r['auc_finetuned']:.4f} "
              f"ftfull={r.get('auc_finetuned_fullctx', float('nan')):.4f} "
              f"zs={r['auc_zeroshot']:.4f} | F1 ft={r['f1_finetuned']:.4f} "
              f"ftfull={r.get('f1_finetuned_fullctx', float('nan')):.4f} "
              f"zs={r['f1_zeroshot']:.4f}")
