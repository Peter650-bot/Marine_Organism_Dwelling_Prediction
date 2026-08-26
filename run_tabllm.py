#!/usr/bin/env python3
"""
run_tabllm.py -- TabLLM at k=32, merged into `figures/<species>/results.json`
so the model-comparison table can carry a TabLLM row. Run `run_pipeline.py`
first; this script adds to what it wrote.

Comparability is the whole point, so this reuses, not re-derives:
  * the same KDE core-area labels and contiguous LORO folds (`label_cv`);
  * the same test cells -- the RNG draw order of `run_pipeline.py` is replayed
    and then ASSERTED against the recorded `n_test` / `n_pos_test`;
  * the same k=32 exemplars -- drawn with `run_pipeline._kshot_draw` under the
    same per-(k, draw) seeds, so TabLLM sees exactly the rows TabICLv2, RF,
    GBT, kNN and LogReg saw.

Scoring is `logprob` (P("high") vs P("low") from the answer-token
distribution), which is what TabLLM (Hegselmann et al.) actually specifies and
which avoids the verbalized read-out's tie ceiling of 0.681. Guided decoding is
off in that mode by design -- it would renormalise the distribution being
measured.

Serialization uses the `full` variant: the classical peers and TabICLv2 all see
all six features including latitude and longitude, so a geography-blind TabLLM
would not be a like-for-like row.

Backend: an OpenAI-compatible vLLM server (Modal). BILLS GPU TIME. Responses
are cached in `figures/tabllm_cache.sqlite`, keyed by model id and prompt,
so an interrupted run resumes without repaying for completed cells.

CLI:
    modal deploy deploy/vllm_server.py
    python run_tabllm.py --base-url https://<endpoint>/v1
"""

import argparse
import json
import os
import sys
import time

import numpy as np

FIG_ROOT = "figures"
CACHE = os.path.join(FIG_ROOT, "tabllm_cache.sqlite")
K = 32
SEED = 42
N_TEST = 300
VARIANT = "full"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    ap.add_argument("--n-draws", type=int, default=5,
                    help="exemplar draws per fold; must match run_pipeline.N_DRAWS "
                         "for the k-shot rows to be comparable")
    ap.add_argument("--species", action="append", default=None)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--error-abort", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true",
                    help="build every prompt and report the call count, then stop "
                         "without contacting the server (FREE)")
    a = ap.parse_args(argv)
    if not a.base_url and not a.dry_run:
        raise SystemExit("--base-url is required (or use --dry-run)")

    sys.path.insert(0, os.getcwd())
    import retention_core as rc
    import label_cv as lcv
    import tabllm_prompt as tp
    import run_pipeline as rp
    from sklearn.metrics import roc_auc_score, f1_score

    client = None
    if not a.dry_run:
        from anthropic_tabllm import TabLLMClient
        os.makedirs(FIG_ROOT, exist_ok=True)
        client = TabLLMClient(model=a.model, cache_path=CACHE, n_vote=1,
                              max_tokens=8, scoring="logprob",
                              base_url=a.base_url, api_key=a.api_key,
                              max_retries=a.max_retries, error_abort=a.error_abort)

    total_calls = 0
    for species in (a.species or list(rp.RUNS)):
        vel_nc, csv, _basin = rp.RUNS[species]
        path = os.path.join(FIG_ROOT, species.replace(" ", "_"), "results.json")
        if not os.path.exists(path):
            print(f"skip {species}: no {path}")
            continue
        with open(path) as fh:
            res = json.load(fh)

        cells = tp.load_cells(species, vel_nc, csv)
        X, lats, lons, vel = cells["X"], cells["lats"], cells["lons"], cells["vel"]
        land_present = cells.get("land_present", True)
        df = rc.load_sightings(csv, species, vel)
        dens = lcv.kde_density(df["longitude"].values, df["latitude"].values,
                              df["weighted_records"].values, lons, lats)
        y = lcv.core_area_label(dens)
        splits, _ = lcv.loro_kmeans_folds(lats, lons)
        prevalence = float(y.mean())

        system = tp.build_system(species, VARIANT, prevalence, prior=False,
                                 scoring="logprob")
        recorded = {f["fold"]: f for f in res["folds"]}
        rng = np.random.default_rng(SEED)      # replay run_pipeline's draw order
        per_fold = []

        for i, (tr, te) in enumerate(splits):
            kept = lcv.balanced_subsample(te, y, N_TEST, rng)
            if kept is None or len(np.unique(y[tr])) < 2:
                continue
            rec = recorded.get(i)
            if rec is None:
                raise SystemExit(f"{species} fold {i}: scoreable in replay but "
                                 f"absent from results.json")
            if len(kept) != rec["n_test"] or int(y[kept].sum()) != rec["n_pos_test"]:
                raise SystemExit(
                    f"{species} fold {i}: replay mismatch -- n_test "
                    f"{len(kept)} vs {rec['n_test']}, n_pos "
                    f"{int(y[kept].sum())} vs {rec['n_pos_test']}")

            for draw in range(a.n_draws):
                drng = np.random.default_rng(SEED + 1000 * K + draw)
                sh = rp._kshot_draw(tr, y, K, drng)   # identical to the peers'
                if sh is None:
                    continue
                # build_messages wants the ordered {cell_index: "high"|"low"}
                # map that sample_exemplars returns -- construct it from the
                # SAME cells the other k-shot models were given, so the demos
                # are identical rows and not merely an identical count.
                ex = {int(ix): ("high" if y[ix] == 1 else "low") for ix in sh}
                jobs = [{"id": f"{i}:{draw}:{j}", "system": system,
                         "messages": tp.build_messages(X, ex, int(ci), VARIANT,
                                                       land_present, "logprob")}
                        for j, ci in enumerate(kept)]
                total_calls += len(jobs)
                if a.dry_run:
                    continue
                t0 = time.time()
                out = client.classify_batch(jobs)
                p = np.array([out[j["id"]]["p_high"] for j in jobs], float)
                ok = np.array([out[j["id"]].get("ok", True) for j in jobs], bool)
                yt = y[kept]
                if ok.sum() < 10 or len(np.unique(yt[ok])) < 2:
                    print(f"  fold {i} draw {draw}: too few usable answers "
                          f"({int(ok.sum())}/{len(jobs)}) -- skipped", flush=True)
                    continue
                auc = float(roc_auc_score(yt[ok], p[ok]))
                f1 = float(f1_score(yt[ok], (p[ok] >= 0.5).astype(int),
                                    zero_division=0))
                per_fold.append({"fold": i, "draw": draw, "auc": auc, "f1": f1,
                                 "n_used": int(ok.sum()),
                                 "n_no_answer": int((~ok).sum())})
                print(f"  [{time.time()-t0:.0f}s] fold {i} draw {draw}: "
                      f"AUC {auc:.4f} F1 {f1:.4f} "
                      f"({int((~ok).sum())} no-answer)", flush=True)

        if a.dry_run:
            continue
        res.setdefault("kshot", {}).setdefault(str(K), {})["tabllm"] = per_fold
        res["tabllm"] = {"model": a.model, "scoring": "logprob", "k": K,
                         "variant": VARIANT, "n_draws": a.n_draws,
                         "cache": CACHE,
                         "note": ("same test cells and same k=32 exemplars as the "
                                  "other k-shot models")}
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)
        if per_fold:
            arr = np.array([r["auc"] for r in per_fold], float)
            print(f"{species}: TabLLM k={K} AUC {arr.mean():.4f} "
                  f"(sd {arr.std(ddof=1):.4f}, n={len(arr)})", flush=True)

    if a.dry_run:
        print(f"\nDRY RUN: {total_calls} model calls would be made "
              f"(k={K}, {a.n_draws} draws). Nothing was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
