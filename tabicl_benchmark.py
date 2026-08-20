#!/usr/bin/env python3
"""
tabicl_benchmark.py — TabICL (tabular foundation model) benchmark for dwelling
prediction, evaluated head-to-head with the TabLLM curve and the classical
baselines on *identical* cells, folds, exemplars and test subsamples.

TabICL (arXiv 2502.05564, soda-inria, BSD 3-Clause) is NOT a language model: it
is a transformer pretrained on synthetic tabular datasets that classifies in a
single forward pass, `y_pred = model(X_train, y_train, X_test)`. There is no
serialization, no prompt and no verbalizer — it consumes the numeric feature
matrix directly. So this is a *different experiment* from tabllm_pipeline, not
another backend for it:

    TabLLM  : can a LANGUAGE model reason about habitat from a text description?
    TabICL  : does a purpose-built TABULAR foundation model beat RF/GBT here?

Two evaluation modes, because they answer different questions and have
different honest peers:

  * `full`   — TabICL gets the WHOLE training fold as context. Its peer is the
               `rf_full` ceiling (~0.99 random / ~0.89 spatial), NOT the k-shot
               curve. This is TabICL used as designed.
  * `kshot`  — TabICL's context is restricted to the SAME k exemplars that
               tabllm_pipeline hands the LLM (same seeds via tp._seed_for), so
               it is directly comparable to the TabLLM few-shot curve and to
               rf_kshot / gbt_kshot. **Caveat: TabICL is pretrained for
               300-100K context rows, so k=8..32 is far outside its design
               range — read those points as "TabICL used adversarially", not as
               its intended operating point.** Reported because the matched-k
               comparison is the only apples-to-apples one against TabLLM.

Everything physical is reused from retention_core via tabllm_pipeline.load_cells,
so the cells, the median-of-nonzero `y_cls` threshold and the 3-degree spatial
blocks are byte-identical to the TabLLM/species_pipeline runs.

Heavy deps (torch, tabicl, sklearn, matplotlib) are imported lazily, so
`--selftest` runs on numpy + stdlib alone.

CLI:
    python tabicl_benchmark.py --selftest                     # offline, free
    python tabicl_benchmark.py --species "Caretta caretta" \
        --velocity Datasets/Oceanic_data.nc \
        --species-csv Datasets/caretta_data.csv               # CPU by default
"""

import argparse
import json
import os

import numpy as np

SHOTS = [8, 16, 32]
SCHEMES = ("random", "spatial")
SEED = 42
FIG_NUM = "24"   # per-species figure numbering: TabLLM used 21-23

# TabICLv2 weights (Qu et al., TabICLv2). The `tabicl` package already defaults
# to this file, but the default is a moving target across releases, so every run
# names it explicitly and `assert_checkpoint` re-checks the architecture that was
# actually loaded. v1/v1.1 are listed so a downgrade is a one-flag experiment.
CKPT_V2 = "tabicl-classifier-v2-20260212.ckpt"
CKPT_V1_1 = "tabicl-classifier-v1.1-20250506.ckpt"
CKPT_V1 = "tabicl-classifier-v1-20250208.ckpt"

# Config keys that exist only in the v2 architecture (target-aware column
# embedding + scalable-softmax attention). Checking the loaded config catches the
# case a local --model-path silently holds v1 weights: `model_path` wins over
# `checkpoint_version` inside TabICLClassifier, so the flag alone proves nothing.
V2_MARKERS = ("col_target_aware", "col_ssmax", "icl_ssmax")


def checkpoint_fingerprint(clf):
    """Which weights this fitted classifier actually loaded."""
    cfg = getattr(clf, "model_config_", {}) or {}
    return {
        "checkpoint_version_requested": clf.checkpoint_version,
        "model_path_loaded": str(getattr(clf, "model_path_", "")),
        "architecture": "v2" if all(k in cfg for k in V2_MARKERS) else "v1",
        "config": {k: cfg.get(k) for k in
                   ("embed_dim", "icl_num_blocks", "max_classes") + V2_MARKERS},
    }


def assert_checkpoint(clf, want="v2"):
    """Fail loudly if the loaded weights are not the generation we claim."""
    fp = checkpoint_fingerprint(clf)
    if fp["architecture"] != want:
        raise RuntimeError(
            f"expected TabICL{want} weights but the loaded checkpoint looks like "
            f"TabICL{fp['architecture']}: {fp}")
    return fp


# ----------------------------------------------------------------------------
#  Context selection (pure; unit-testable without torch)
# ----------------------------------------------------------------------------
def select_context(X, y_cls, train_idx, mode, k, seed, sample_exemplars):
    """Rows TabICL sees as in-context training data.

    mode='full'  -> the entire training fold (TabICL as designed)
    mode='kshot' -> exactly the k exemplars TabLLM got (same seed => same rows)

    Returns an index array. Always a subset of train_idx, so a held-out block
    under spatial CV can never leak into the context.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    if mode == "full":
        return train_idx
    if mode == "kshot":
        ex = sample_exemplars(X, y_cls, train_idx, k, seed)
        return np.asarray(sorted(ex), dtype=int)
    raise ValueError(f"unknown context mode {mode!r}")


def _auc_or_none(y, s, roc_auc_score, min_n=20):
    y = np.asarray(y)
    if len(y) < min_n or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


# ----------------------------------------------------------------------------
#  Runner
# ----------------------------------------------------------------------------
def run_benchmark(cells, shots=None, schemes=SCHEMES, modes=("full", "kshot"),
                  n_test=300, device=None, n_jobs=None, n_estimators=8,
                  verbose=True, with_baselines=True, model_path=None,
                  seed_variant="full", checkpoint_version=CKPT_V2,
                  want_arch="v2"):
    """TabICL under random AND spatial-block CV, scored on the same balanced
    test subsample as the TabLLM run. Writes tabicl_results.json + figure 24."""
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import roc_auc_score
    import tabllm_pipeline as tp
    from tabicl import TabICLClassifier

    X, y_cls, groups = cells["X"], cells["y_cls"], cells["groups"]
    species, out_dir = cells["species"], cells["out_dir"]
    shots = SHOTS if shots is None else shots

    def log(*a):
        if verbose:
            print(*a, flush=True)

    def folds(scheme):
        if scheme == "random":
            return list(StratifiedKFold(5, shuffle=True,
                                        random_state=SEED).split(X, y_cls))
        ng = len(np.unique(groups))
        return list(GroupKFold(min(5, ng)).split(X, y_cls, groups))

    results = {
        "species": species, "model": "TabICL", "n_cells": cells["n_cells"],
        "prevalence": cells["prevalence"], "n_test": n_test,
        "n_estimators": n_estimators, "device": device or "auto",
        "seed_variant": seed_variant,
        "checkpoint_version": checkpoint_version, "checkpoint": None,
        "points": [], "notes": [],
    }

    def fit_score(ctx_idx, kept):
        """One TabICL forward pass: context -> probabilities for `kept`.

        `model_path` points at a locally-downloaded checkpoint. Worth using: the
        default HF path goes through the Xet content-addressed backend, which
        fails hard on an unstable link ("CAS Client Error ... error decoding
        response body") instead of resuming. A local file also pins the exact
        weights, which matters for reproducing the reported numbers.
        """
        clf = TabICLClassifier(n_estimators=n_estimators, device=device,
                               n_jobs=n_jobs, random_state=SEED,
                               checkpoint_version=checkpoint_version,
                               **({"model_path": model_path} if model_path else {}))
        clf.fit(X[ctx_idx], y_cls[ctx_idx])
        if results["checkpoint"] is None:
            results["checkpoint"] = assert_checkpoint(clf, want_arch)
        proba = clf.predict_proba(X[kept])
        # column for the positive class, robust to class ordering
        pos = list(clf.classes_).index(1)
        return proba[:, pos]

    for scheme in schemes:
        for mode in modes:
            ks = [None] if mode == "full" else shots
            for k in ks:
                aucs, base_runs, ctx_sizes = [], {}, []
                for fold, (tr, te) in enumerate(folds(scheme)):
                    # `seed_variant` MUST match the tabllm_pipeline run's variant
                    # tag ("full" by default), because _seed_for hashes it: any
                    # other string silently draws DIFFERENT exemplars and a
                    # DIFFERENT test subsample, turning the head-to-head into two
                    # independent draws instead of a paired comparison.
                    seed = tp._seed_for(fold, scheme, species,
                                        k if k is not None else -1, seed_variant)
                    ctx = select_context(X, y_cls, tr, mode, k, seed,
                                         tp.sample_exemplars)
                    kept = tp.subsample_test(te, y_cls, n=n_test,
                                             mode="balanced", seed=seed)
                    # hard leakage assertions, every fold (same as TabLLM)
                    assert np.intersect1d(ctx, kept).size == 0, \
                        "TabICL context overlaps the test cells"
                    if scheme == "spatial":
                        assert set(groups[ctx]).isdisjoint(set(groups[kept])), \
                            "TabICL context shares a spatial block with test"
                    if len(np.unique(y_cls[ctx])) < 2:
                        continue
                    ctx_sizes.append(int(len(ctx)))
                    s = fit_score(ctx, kept)
                    a = _auc_or_none(y_cls[kept], s, roc_auc_score)
                    if a is not None:
                        aucs.append(a)
                    if with_baselines:
                        # identical peers on identical kept indices
                        bsc, _ = tp._baseline_scores(
                            X, y_cls,
                            {int(i): ("high" if y_cls[i] == 1 else "low")
                             for i in ctx} if mode == "kshot" else {},
                            tr, kept, seed)
                        for name, sc in bsc.items():
                            b = _auc_or_none(y_cls[kept], sc, roc_auc_score)
                            if b is not None:
                                base_runs.setdefault(name, []).append(b)

                point = {"scheme": scheme, "mode": mode, "k": k,
                         "auc_tabicl": float(np.mean(aucs)) if aucs else None,
                         # Per-fold values + spread, because a bare mean hides
                         # how noisy these are: a k=8 point moved 0.105 between
                         # two different exemplar draws. Without the fold values
                         # you cannot tell a real gap from a lucky split.
                         "auc_tabicl_folds": [float(a) for a in aucs],
                         "auc_tabicl_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
                         "auc_tabicl_sem": float(np.std(aucs, ddof=1) / np.sqrt(len(aucs)))
                                            if len(aucs) > 1 else None,
                         "n_folds": len(aucs),
                         "context_rows": int(np.mean(ctx_sizes)) if ctx_sizes else 0}
                for name, runs in base_runs.items():
                    point[f"auc_{name}"] = float(np.mean(runs)) if runs else None
                    point[f"auc_{name}_folds"] = [float(x) for x in runs]
                results["points"].append(point)
                log(f"[{scheme}/{mode}] k={k}  ctx={point['context_rows']:>6} "
                    f"rows  AUC_tabicl={point['auc_tabicl']}")

    if any(p["mode"] == "kshot" for p in results["points"]):
        results["notes"].append(
            "k-shot points use 8-32 context rows; TabICL is pretrained for "
            "300-100,000. Those points probe TabICL far outside its design "
            "range and exist only for matched-k comparability with TabLLM.")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tabicl_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"[saved] {path}")
    try:
        _make_figure(results, out_dir)
    except Exception as e:
        log(f"figure generation skipped: {e}")
    return results


def _make_figure(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = results["points"]
    schemes = sorted({p["scheme"] for p in pts})
    fig, axes = plt.subplots(1, len(schemes), figsize=(7 * len(schemes), 5),
                             squeeze=False)
    for ax, scheme in zip(axes[0], schemes):
        sub = [p for p in pts if p["scheme"] == scheme]
        ks = sorted({p["k"] for p in sub if p["k"] is not None})
        kshot = {p["k"]: p for p in sub if p["mode"] == "kshot"}
        if ks:
            ax.plot(ks, [kshot[k]["auc_tabicl"] for k in ks], marker="o",
                    lw=2, label="TabICL (k-shot)")
            for key, lbl in [("auc_rf_kshot", "RF (k-shot)"),
                             ("auc_gbt_kshot", "GBT (k-shot)")]:
                ys = [kshot[k].get(key) for k in ks]
                if any(y is not None for y in ys):
                    ax.plot(ks, ys, marker="s", ls="--", alpha=.7, label=lbl)
        full = next((p for p in sub if p["mode"] == "full"), None)
        if full and full["auc_tabicl"] is not None:
            ax.axhline(full["auc_tabicl"], color="tab:green", lw=2,
                       label=f"TabICL (full ctx, {full['context_rows']} rows)")
        if full and full.get("auc_rf_full") is not None:
            ax.axhline(full["auc_rf_full"], ls="--", color="k",
                       label="RF (full) ceiling")
        ax.axhline(0.5, ls=":", color="grey")
        ax.set_xlabel("context rows (k)")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{results['species']} — TabICL, {scheme} CV")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{FIG_NUM}_tabicl_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ----------------------------------------------------------------------------
#  Offline self-test (numpy + stdlib only)
# ----------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    N = 300
    X = rng.normal(size=(N, 6))
    y = (rng.random(N) < 0.4).astype(int)
    tr = np.arange(0, 210)
    te = np.arange(210, 300)

    def sample_exemplars(X, y_cls, train_idx, k, seed):
        r = np.random.default_rng(seed)
        ti = np.asarray(train_idx)
        pos, neg = ti[y_cls[ti] == 1], ti[y_cls[ti] == 0]
        kp, kn = k // 2, k - k // 2
        pick = np.concatenate([r.choice(pos, min(kp, len(pos)), replace=False),
                               r.choice(neg, min(kn, len(neg)), replace=False)])
        return {int(i): ("high" if y_cls[i] == 1 else "low") for i in pick}

    full = select_context(X, y, tr, "full", None, 1, sample_exemplars)
    assert np.array_equal(full, tr), "full mode must use the whole train fold"

    ks = select_context(X, y, tr, "kshot", 16, 123, sample_exemplars)
    assert len(ks) <= 16 and set(ks).issubset(set(tr.tolist()))
    assert np.intersect1d(ks, te).size == 0, "k-shot context leaked into test"
    # deterministic for a fixed seed, different for a different seed
    assert np.array_equal(ks, select_context(X, y, tr, "kshot", 16, 123,
                                             sample_exemplars))
    assert not np.array_equal(
        ks, select_context(X, y, tr, "kshot", 16, 999, sample_exemplars))
    try:
        select_context(X, y, tr, "nope", 8, 1, sample_exemplars)
        raise RuntimeError("should have rejected an unknown mode")
    except ValueError:
        pass

    from sklearn.metrics import roc_auc_score as _r
    assert _auc_or_none(np.zeros(50, int), np.zeros(50), _r) is None, \
        "single-class folds must yield None, not a crash"
    assert _auc_or_none(np.array([0, 1] * 15), rng.random(30), _r) is not None
    print("tabicl_benchmark self-test OK (context selection, leakage, auc guard)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="offline unit checks (no torch / no tabicl / no data)")
    ap.add_argument("--species")
    ap.add_argument("--velocity")
    ap.add_argument("--species-csv")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--n-test", type=int, default=300,
                    help="balanced test cells per fold (matches the TabLLM run)")
    ap.add_argument("--shots", default="8,16,32",
                    help="k values for the matched-k comparison against TabLLM")
    ap.add_argument("--modes", default="full,kshot",
                    help="'full' (TabICL as designed, peer of rf_full) and/or "
                         "'kshot' (matched-k peer of the TabLLM curve)")
    ap.add_argument("--schemes", default="random,spatial")
    ap.add_argument("--device", default=None,
                    help="'cpu', 'cuda', or omit to let TabICL choose")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="torch threads on CPU (default: TabICL's choice)")
    ap.add_argument("--n-estimators", type=int, default=8,
                    help="TabICL ensemble members; lower = faster on CPU")
    ap.add_argument("--no-baselines", action="store_true",
                    help="skip the RF/GBT/LR/kNN peers (faster)")
    ap.add_argument("--seed-variant", default="full",
                    help="variant tag fed to tabllm_pipeline._seed_for. MUST "
                         "match the TabLLM run being compared against ('full') "
                         "or the exemplars/test cells silently differ and the "
                         "comparison stops being paired.")
    ap.add_argument("--checkpoint-version", default=CKPT_V2,
                    choices=(CKPT_V2, CKPT_V1_1, CKPT_V1),
                    help="which pretrained TabICL generation to run. Default is "
                         "TabICLv2. Ignored by the library when --model-path "
                         "exists, which is why the loaded architecture is "
                         "re-asserted from the checkpoint config.")
    ap.add_argument("--want-arch", default="v2", choices=("v1", "v2"),
                    help="architecture the loaded weights must match, or abort")
    ap.add_argument("--model-path", default=None,
                    help="path to a locally-downloaded TabICL .ckpt. Skips the "
                         "HuggingFace Xet backend, which fails without resuming "
                         "on an unstable connection. Also pins exact weights: "
                         "curl -C - --retry 20 -o tabicl.ckpt https://huggingface.co"
                         "/jingang/TabICL/resolve/main/"
                         "tabicl-classifier-v2-20260212.ckpt")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if not (args.species and args.velocity and args.species_csv):
        ap.error("provide --selftest, or --species/--velocity/--species-csv")

    import tabllm_pipeline as tp
    cells = tp.load_cells(args.species, args.velocity, args.species_csv,
                          fig_dir=args.fig_dir)
    print(f"species={args.species} n_cells={cells['n_cells']} "
          f"prevalence={cells['prevalence']:.3f} n_blocks={cells['n_blocks']}")
    run_benchmark(
        cells,
        shots=[int(s) for s in args.shots.split(",") if s.strip()],
        schemes=tuple(s.strip() for s in args.schemes.split(",") if s.strip()),
        modes=tuple(m.strip() for m in args.modes.split(",") if m.strip()),
        n_test=args.n_test, device=args.device, n_jobs=args.n_jobs,
        n_estimators=args.n_estimators, with_baselines=not args.no_baselines,
        model_path=args.model_path, seed_variant=args.seed_variant,
        checkpoint_version=args.checkpoint_version, want_arch=args.want_arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
