# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Master's Applied ML final project investigating whether modeled Lagrangian particle-retention zones correlate with marine-megafauna sightings. The original study is *Caretta caretta* (loggerhead turtle) in the Mediterranean; an in-progress extension generalises the same method to any species / any ocean basin (a second species, *Cetorhinus maximus* / basking shark in the NE Atlantic, is wired up). The deliverables are the paper (`paper.tex`, ACL LaTeX template in `Applied_Machine_Learning_Template.zip`) and the figures under `figures/`. Background and motivation live in `Peter_Saba_project_proposal.pdf` and `Modeled Particle Retention for Habitat Prediction publication.pdf` — read those before changing methodology.

**Two codepaths exist for the same method, and they must stay numerically consistent:**

- `enhanced_analysis.py` — the original, *Caretta*-hardcoded monolith. Authoritative source of the paper's canonical figures `01_*`–`19_*`.
- `retention_core.py` + `species_pipeline.py` + `species_figures.py` + `ablation_study.py` + `alignment_check.py` — the generalised, multi-species refactor. `retention_core.py` is the single source of truth: its `simulate_retention`/`load_sightings`/`build_features` reproduce `enhanced_analysis.py` Sections 1–2/4a *exactly* under the default params and seeds. That equivalence is the de-facto regression test — if you change `retention_core.py`, re-run the Caretta/Mediterranean case and confirm Spearman ρ≈0.362 and the figure 01–19 numbers are unchanged.

**Key methodological caveat (see `ablation_study.py`):** net of geography (lat/lon) and the static flow field, the retention feature adds ~nothing to predictive skill, and the paper's headline scores are inflated by spatial autocorrelation. Always report metrics under **both** random CV (optimistic, paper-comparable) and **spatial-block / leave-region-out CV** (honest). The generalised pipeline does this by default; don't quote random-CV numbers alone.

## Running the analysis

**Original monolith** (produces figures `01_*`–`19_*`):

```bash
python enhanced_analysis.py
```

**Critical:** this script opens `"Oceanic_data.nc"` and `"caretta_data.csv"` with bare filenames, but the data files actually live in `Datasets/`. Run from `Datasets/` (or symlink the two files into the working directory) or it will fail at section 1. `figures/` is created relative to the working directory.

**Generalised tooling** (CLI scripts; take explicit `--velocity`/`--species-csv` paths, so run from the repo root):

```bash
# feature ablation — writes figures/20_ablation.png, leaves 01-19 untouched
python ablation_study.py --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv

# coverage diagnostic only (no simulation, cheap): does a velocity cube cover a
# species' records in space/time, and if not, what Copernicus bounds to download
python alignment_check.py --velocity Datasets/NE_Atlantic.nc \
  --species-csv "Datasets/obis_seamap_custom_.../...dist_sp_1deg_csv.csv" --species "Cetorhinus maximus"

# full per-species run(s) + pooled cross-species model; --run is repeatable
python species_pipeline.py \
  --run "Caretta caretta:Datasets/Oceanic_data.nc:Datasets/caretta_data.csv" \
  --run "Cetorhinus maximus:Datasets/NE_Atlantic.nc:Datasets/<multispp>.csv"

# in-context TabLLM extension (serialize each cell -> Claude classifies HIGH/LOW)
python tabllm_pipeline.py --selftest          # offline unit checks, FREE
python tabllm_pipeline.py --probe --species "Caretta caretta" \
  --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv  # FREE: loads cells, prints prevalence/land/counts + a sample serialization
python tabllm_pipeline.py --species "Caretta caretta" --smoke ...   # BILLS the Anthropic API (unless --base-url)
python species_pipeline.py --run "..." --tabllm                     # full curve; BILLS the API (unless --tabllm-base-url)

# FREE alternative: point at a self-hosted vLLM server (OpenAI-compatible /v1)
export VLLM_BASE_URL="https://<tunnel>/v1"            # or pass --base-url
python tabllm_pipeline.py --species "Caretta caretta" \
  --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv \
  --model unsloth/Llama-3.2-3B-Instruct --smoke        # $0 via vLLM
python species_pipeline.py --run "..." --tabllm \
  --tabllm-model unsloth/Llama-3.2-3B-Instruct         # $0 (reads $VLLM_BASE_URL)
```

**Two inference backends, one client.** `anthropic_tabllm.py::TabLLMClient` talks to the pay-as-you-go **Anthropic API** by default (`ANTHROPIC_API_KEY`, separate from any subscription; `--smoke`/`--tabllm`/`run_curve` cost money — Haiku 4.5 bulk + Batch; full-grid ≈ $40–120; don't launch billed runs without go-ahead), **or** to a **free self-hosted vLLM server** when `--base-url`/`--tabllm-base-url` (or `$VLLM_BASE_URL`) is set. The vLLM path uses the `openai` SDK against vLLM's OpenAI-compatible `/v1/chat/completions`, `response_format` json_schema for the same `{label, p_high}` structured output, and — since vLLM has no Batch API — a `ThreadPoolExecutor` fan-out (vLLM's continuous batching turns concurrent requests into GPU batches). vLLM needs an NVIDIA GPU, so it is expected to run off-box (e.g. a free Colab T4 behind a cloudflared tunnel) with the pipeline hitting the tunnel URL; the served model id (`--model`) flows into the cache key, so switching backends cleanly invalidates old entries. `--selftest`/`--probe` are free on both paths. Responses are cached in a committed `figures/<species>/tabllm_cache.sqlite` — that cache, not the seed, is the reproducibility anchor (LLM output isn't byte-deterministic).

There are no tests, no lint config, no build step, and no `requirements.txt`. Dependencies are discovered by reading imports. Required: `numpy pandas xarray h5py matplotlib seaborn scipy scikit-learn pillow`. Optional (scripts degrade gracefully and print an install hint if missing): `libpysal esda` (Moran's I), `shap` (feature attribution), `plotly` (interactive HTML map), `anthropic` (paid TabLLM backend), `openai` (free vLLM TabLLM backend). The `anthropic`/`openai` SDKs are imported lazily inside `TabLLMClient`, so only whichever backend you actually use needs installing.

The generalised pipeline **caches the slow ensemble retention field** to a hidden `.npy` (keyed by velocity filename + grid shape + step count) under the figure dir, so re-runs and `species_figures.py` regeneration are fast. `enhanced_analysis.py` does not cache — expect a multi-minute run (10 ensemble × 500 particles × 365 days, plus a diffusivity sweep and seasonal split in section 6).

`alignment_check.py` is the prerequisite step before adding any new species: it verifies spatial/temporal overlap and prints ready-to-run `copernicusmarine subset` commands (DUACS L4 geostrophic `ugos`/`vgos`). The altimetry record starts 1993, so records before then are unrecoverable; geostrophy is unreliable within ~5° of the equator (down-weighted, not dropped).

## Code architecture

The script is one long top-level program divided into seven numbered sections, executed in order. State flows between sections via module-level globals — there are no functions wrapping the pipeline, so reordering or partially running sections requires care.

1. **Data preprocessing** (`Section 1`) — Loads `Oceanic_data.nc` via `h5py` directly (not `xarray.open_dataset`), wraps `ugos`/`vgos` velocity fields in an `xr.Dataset`, decodes the CF time axis manually. Loads `caretta_data.csv`, drops fully-empty columns, parses dates, applies a Gaussian temporal-proximity weight (σ = 365 days) centered on the ocean-model midpoint to give each sighting a `weighted_records`.
2. **Particle simulation** (`Section 2`) — Stochastic Lagrangian advection: deterministic drift from `(ugos, vgos)` plus Gaussian random walk with diffusivity `K_diff = 10 m²/s`. Uses a hand-rolled bilinear interpolator on raw numpy arrays (`_interp_uv`) instead of `xr.DataArray.interp` for speed — this matters, the xarray path in section 6 is ~100× slower. Builds an ensemble retention map (`retention_mean`, `retention_std`) by binning visited grid cells across 10 runs × 500 particles. After this section, `ds.close(); del ds` frees the xarray dataset; downstream code uses the cached `_u_data`/`_v_data` numpy arrays.
3. **Statistical analysis** (`Section 3`) — Spearman correlation between `retention_norm` and `sightings_norm` on cells where either is non-zero, plus permutation null (N=1000), bootstrap CI (N=1000), Moran's I on a 4×-coarsened grid, KDE for both point distributions, and a hand-rolled GWR-lite (Gaussian-weighted local OLS, bandwidth 2°, subsampled to 2000 points for tractability).
4. **Machine learning** (`Section 4`) — Builds a 6-feature matrix per grid cell: `[retention, mean_speed, vorticity, latitude, longitude, dist_to_coast]`. The coast-distance feature uses `scipy.ndimage.distance_transform_edt` on the NaN mask of the velocity field — i.e. land is wherever the ocean model has no data. Targets: continuous `flat_sightings` and a binary high/low split at the median of nonzero sightings. Models: RandomForest (class-balanced, 5-fold CV), GradientBoostingRegressor + SHAP, DBSCAN on raw sighting coords and particle endpoints, GMM with BIC-selected component count (2–7), and MLPClassifier.
5. **Visualizations** (`Section 5`) — Saves figures `01_*` through `17_*` in order; numbering is load-bearing because the paper references them by index. The animated GIF is built frame-by-frame via PIL (matplotlib's `FuncAnimation` was avoided for memory). Plotly is optional.
6. **Validation & robustness** (`Section 6`) — Diffusivity sensitivity sweep (`K ∈ {0, 5, 10, 50, 100}`) and a winter-vs-summer seasonal split. Note this section *does* use `xr.DataArray.interp` and is the slowest part of the run. Produces figures `18_*` and `19_*`.
7. **Summary report** (`Section 7`) — Prints a formatted text summary to stdout. Nothing is written to disk here.

### Generalised multi-species pipeline

- **`retention_core.py`** — the shared, validated building blocks (`load_velocity`, `make_interp`, `simulate_retention`, `load_sightings`, `build_features`, `spatial_groups`). This is where the simulation constants (`K_DIFF=10`, `N_PARTICLES=500`, `N_ENSEMBLE=10`, `MAX_STEPS=365`, `SEED=42`, `ALTIMETRY_START_YEAR=1993`) and the 6-feature `FEATURES` list live. It generalises the monolith: reads grid/extent/resolution from the file (no hardcoded Mediterranean), reports the cube's longitude convention (`pm180` vs `0360`) and aligns occurrence longitudes to it, filters by species name, drops `;`-merged gridded-summary cells, and applies the altimetry-era cutoff. **Changing anything here changes the canonical results — treat it as load-bearing.**
- **`species_pipeline.py`** — `run_species()` (per-species: retention → association stats → RF/GBR under random *and* spatial CV → permutation importance) and `run_pooled()` (stacks all species' per-cell matrices with a species one-hot, evaluates transfer via leave-one-region-out and leave-one-**species**-out CV). Writes `figures/<Species_name>/results.json` + `summary.png` and `figures/pooled/pooled_results.json`.
- **`species_figures.py`** — the full figure battery for the generalised runs (`figures/<species>/01_*`–`08_*`, `figures/pooled/01_*`–`05_*`). Callable inline by `species_pipeline.py` (no recompute) or standalone from the cached retention field. **This numbering is independent of the root `figures/01_*`–`20_*` set** — per-species figures live in subfolders.
- **`ablation_study.py`** — cumulative-tier (geography → +static ocean → +retention), LOFO, geography-controlled, and spatial-null (retention shuffled within latitude bands) experiments; quantifies the retention feature's marginal skill honestly. Writes only `figures/20_ablation.png`.
- **`alignment_check.py`** — pure diagnostic, no simulation. Coverage report + Copernicus download-bound generator for onboarding new species/basins.

### In-context TabLLM extension

Applies the TabLLM method (arXiv 2210.10723: serialize a tabular row to natural-language text, then classify with an LLM) to dwelling prediction, using a modern Claude model **in-context** (zero-/few-shot prompting, no fine-tuning). Each grid cell is one TabLLM "row." All three modules import `anthropic`/`sklearn`/heavy deps **lazily**, so the serialization, exemplar-sampling, and leakage logic import and unit-test with numpy + stdlib only (`--selftest`).

- **`anthropic_tabllm.py`** — `TabLLMClient`: structured-output `{label, p_high}` (numeric bounds aren't enforced → clamp client-side), an sqlite response cache keyed by `sha256(model|system|messages|gen-params|vote_idx|schema)`, and an N-vote ensemble. **Two backends, selected by `base_url`:** (a) default **Anthropic** — `output_config` JSON schema, `messages.create` + Batch-API (results keyed by `custom_id`), temperature Haiku-only (Opus 4.8 rejects it); model IDs `claude-haiku-4-5` (bulk) / `claude-opus-4-8` (spot-check). (b) **vLLM/OpenAI** (`base_url` or `$VLLM_BASE_URL` set) — lazy `openai` SDK against `/v1/chat/completions`, `response_format` json_schema for the same `OUT_SCHEMA`, temperature always sent, and `_classify_batch_openai` (a `ThreadPoolExecutor` fan-out; cache reads/writes stay on the main thread since sqlite isn't thread-safe). Both paths share `_infer_one` and the same `_parse`/cache/aggregation, so downstream code (`tabllm_pipeline`, `species_pipeline`, `aquax_benchmark`) is backend-agnostic. **Failure handling (batched path):** failed requests are **never cached** (a re-run re-attempts them once the backend is healthy), and a **circuit-breaker** (`error_abort`, default 25; chunked submission so it aborts early) raises instead of quietly caching thousands of neutral `0.0`s when the server is down. Tune via `--max-retries` (per-request retries; lower = fail fast against a dead vLLM) and `--error-abort` (0 disables); `species_pipeline` mirrors these as `--tabllm-max-retries`/`--tabllm-error-abort`.
- **`tabllm_pipeline.py`** — reuses `rc.load_velocity/load_sightings/build_features/spatial_groups` and the **same** median-of-nonzero `y_cls` threshold and 3° blocks as `species_pipeline`. `serialize_row` (4 variants: `full`, `geo_blind`, `georegion_blind`, `ret_ablated`), `build_system` (prior-free by default; base rate stated), `sample_exemplars` (train-only, class-balanced, **label-only** demos), `subsample_test`, `assert_no_leak`, and `run_curve` (few-shot learning curve under random AND spatial-block CV vs RF/GBT/LR/kNN baselines + a full-data RF ceiling). Writes figures `21`/`22` + `tabllm_results.json`.
- **`aquax_benchmark.py`** — external SDM benchmark. Loads AquaMaps/AquaX per-species habitat-suitability CSV, regrids onto our cells, and reports Spearman ρ + a **whole-grid** AUC (AquaMaps shares OBIS occurrences → partly circular, and is untrained → never a per-fold CV peer). Writes figure `23` + `23_aquax_metrics.json`.
- **Hook:** `species_pipeline.py --tabllm` runs the curve per species after the canonical run (non-breaking; `run_species` is untouched). Headline config = spatial-block CV + prior-free prompt + `georegion_blind` (the honest analogue of the geography-dominance finding). Per-species/pooled figure folders use their own numbering; TabLLM figures `21`–`23` live under `figures/<species>/`.

### TabLLM model roadmap — raising the AUC

The pipeline is **OpenAI-compatible** (`--base-url` + `--model` + `--api-key`, `response_format: json_schema`) and the sqlite cache (`figures/<species>/tabllm_cache.sqlite`) is **keyed by model id → resumable and idempotent**, so a bigger model is a matter of flags, not code, and free-tier daily caps only stretch wall-clock (cached cells are skipped on re-run). Progression of measured AUC (random-CV): 7B→0.51, 72B→0.58–0.61. The binding constraint on "bigger for free" is *calls/day × structured-output support*, not VRAM — so hosted free tiers (someone else serves the big model) beat self-hosting ever-larger on Modal.

- **Option 1 (in progress) — hosted frontier/large free tier.** Point the pipeline at **Cerebras Qwen3-235B-A22B** (`--base-url https://api.cerebras.ai/v1 --model qwen-3-235b-a22b`) or **Gemini 2.5 Pro** free tier, running the *headline config only* (`--schemes spatial --variants georegion_blind --shots 16 --n-test 300`, ~3k calls) to fit daily quota. Cerebras is the cleaner drop-in (strict `json_schema`, ~1M tok/day, very fast); Gemini has the higher ceiling but a low ~100 RPD free cap → spread over days. Requires a free provider API key (user-supplied). Expected AUC ~0.63–0.72.
- **Option 2 (later) — reasoning MoE, full sweep.** Self-host **`openai/gpt-oss-120b`** on a **single H100** (MXFP4 fits one GPU) on Modal: swap `MODEL`/`GPU`/`TENSOR_PARALLEL` in `deploy/vllm_server.py` (`GPU="H100"`, `TENSOR_PARALLEL=1`) with reasoning enabled. Cheaper than the current 4×A10G and stronger; runs all 18k calls in ~3h.
- **Option 3 (later) — squeeze, don't just scale.** `--n-vote 3` (currently 1) averages `p_high` over votes → smoother ranking → higher AUC, and a reasoning model calibrates `p_high` better than the non-reasoning 72B (which emits a whitespace tail). Likely a larger AUC gain per dollar than any single size bump; also strengthens the paper's methodology.

**Honest caveat (keep in the writeup):** bigger lifts *ranking*/AUC but prompt-only over lat/lon/retention/coast approaches — does not beat — the supervised RF ceiling (~0.99) and is already near the classical k-shot peers (0.63–0.65). The finding (geography dominates; in-context LLMs land near cheap baselines) is robust regardless of model size.

### TabLLM larger-model run — execution plan & backend findings (probed 2026-07-23)

**Decision:** the larger-model (`gpt-oss-120b`, reasoning) run is deferred to a **long-running lab machine that can stay open for weeks**, because *every* free frontier tier is day-capped and no free backend finishes the sweep in one sitting. Run there via **Cerebras** (cheap, free-trial rate-capped → multi-day, cache auto-resumes) or **Modal H100** (paid, no cap, ~3h). All code is staged; it's a matter of flags + wall-clock, not new work.

**Backend reality (measured, supersedes Option 1's assumptions above):**
- **Cerebras** — this account does **not** serve Qwen3-235B; the menu is `gpt-oss-120b`, `zai-glm-4.7`, `gemma-4-31b`. Inference needs billing (402 until a card is added). It is **paid**: **$0.35/M in, $0.75/M out**, with a free-trial **rate cap of ~1M tokens/day, 2,400 req/day, 5 req/min** and **$5 trial credit**. Strict `json_schema` **works**. `gpt-oss-120b` is a reasoning model (thinking on a separate `reasoning_content` channel; clean JSON in `content`).
- **Groq** — serves `openai/gpt-oss-120b` but the free tier is *tighter* (~1,000 req/day, ~8,000 TPM) **and rejects our strict `json_schema`** (it requires every `properties` key to be in `required`) → must fall back to `response_format: json_object`. Not worth it over Cerebras.
- **OpenRouter** — needs a $5 deposit to be useful; `:free` variants are day-capped (50/day, 1,000/day only if ≥$10 deposited).
- **Modal H100** — the only no-rate-limit path. Full `n_test=300` sweep ≈ ~3h ≈ **$12–16**, *tight* against the ~$13 Modal free credit → size `--n-test ≤150` (~$9–10) or wait for the monthly credit reset.

**Sizing (calls = `schemes(2) × variants(2) × shots(3) × folds(5) × n_test`):** `n_test=300` → 18k calls / ~24M tok / ~$9.5 on Cerebras (or ~3h on Modal); `n_test=80` → 4.8k / ~6.4M / ~$2.5 / ~6–7 days on the Cerebras free cap; `n_test=40` → 2.4k / ~3.2M / ~$1.3 / ~3 days. Cost stays well under the $5/$13 credits; **the binding constraint is the daily rate cap, not money.**

**Code already staged:** `tabllm_pipeline.py --max-tokens N` (default 128, unchanged for Haiku/Qwen). **Reasoning models truncate at 128 → broken JSON → neutral 0.0**, so pass **`--max-tokens 384`** for `gpt-oss-120b`/GLM.

**Output-safety (already done; never lose the 72B baseline):** the canonical 72B artifacts are snapshotted to `figures/Caretta_caretta/{tabllm_results_qwen72b.json, 21_tabllm_learning_curve_qwen72b.png, 22_tabllm_geography_ablation_qwen72b.png}`. A new run **overwrites** `tabllm_results.json` + `21_*`/`22_*` (cache is model-keyed, so no answer leakage). **After the run:** regenerate `21`/`22` with **both** curves overlaid (72B snapshot + new model), then add the new model's rows/bar to `TabLLM_model_comparison.docx`. Keep every 72B number.

**Ready-to-run on the lab machine** (keys are user-supplied — **never commit them**):
```bash
# --- Cerebras (cheap, multi-day, resumable: just re-run daily) ---
export VLLM_BASE_URL="https://api.cerebras.ai/v1"; export VLLM_API_KEY="$CEREBRAS_KEY"
python tabllm_pipeline.py --species "Caretta caretta" \
  --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv \
  --base-url "$VLLM_BASE_URL" --api-key "$VLLM_API_KEY" \
  --model gpt-oss-120b --max-tokens 384 --n-test 80     # cache skips done cells on re-run

# --- Modal H100 (paid, no cap, one ~3h sitting) ---
# edit deploy/vllm_server.py: MODEL="openai/gpt-oss-120b", GPU="H100", TENSOR_PARALLEL=1,
#   drop `--quantization awq` (gpt-oss is MXFP4, auto-detected); then `modal deploy ...`
python tabllm_pipeline.py --species "Caretta caretta" \
  --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv \
  --base-url "https://<user>--vllm-tabllm-serve.modal.run/v1" \
  --model "openai/gpt-oss-120b" --max-tokens 384 --n-test 150
```
Credits as of 2026-07-23: **~$13 Modal free**, **$5 Cerebras trial**. Free-provider keys are user-supplied and must stay out of the repo.

## Conventions worth respecting

- **All randomness is seeded.** RNGs use `np.random.default_rng(42)` (main) and `42 + run_idx` (ensemble); seasonal/sensitivity use 77 and 99; the generalised pipeline reuses `rc.SEED=42`. Don't introduce unseeded randomness — reproducibility of the reported correlations depends on it.
- **Root `figures/01_*`–`20_*` are cited by the paper by number.** Renaming or renumbering breaks the writeup. `01–19` come from `enhanced_analysis.py`, `20` from `ablation_study.py`; add further root figures as `21_*` etc. Per-species/pooled figures live in `figures/<Species>/` and `figures/pooled/` subfolders with their own independent numbering — don't conflate the two schemes.
- **The `figures/` directory is the authoritative output.** It already contains a full set of generated artifacts from prior runs; re-running overwrites them.
- **Don't commit the `.nc` cubes** — `Datasets/Oceanic_data.nc` (~626 MB) and `Datasets/NE_Atlantic.nc` (~690 MB). The CSVs are fine. Velocity cubes are DUACS L4 geostrophic (`ugos`/`vgos`); sightings are OBIS-SEAMAP gridded-summary exports (the multi-species file is under `Datasets/obis_seamap_custom_.../`).
- **xarray vs raw-numpy split is intentional.** The performance gap between the bilinear interpolator (`_interp_uv` in the monolith, `make_interp` in core) and `xr.interp` (section 6 of the monolith) is ~100×; if you speed up section 6, port the bilinear interpolator rather than reintroducing xarray on the hot path.
- **Keep the two codepaths in sync.** Prefer adding capability to `retention_core.py` and the CLI scripts over editing the monolith; if you must touch shared logic, verify the Caretta case still reproduces ρ≈0.362 and figures 01–19.
