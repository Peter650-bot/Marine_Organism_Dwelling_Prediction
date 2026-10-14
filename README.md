# Lagrangian Retention and Marine Megafauna Habitat Prediction

Do modelled Lagrangian particle-retention zones predict where marine megafauna are
observed? This repository contains the code and results for a per-cell habitat
classification study across three species and three ocean basins, comparing a
tabular foundation model (**TabICLv2**), an in-context LLM (**TabLLM**), classical
baselines, and a published species-distribution model (**AquaMaps**).
[`extension/`](extension/README.md) repeats the TabICLv2 comparison against
MaxEnt on the MarineSPEED benchmark (40 species, environment-only predictors).

| species | basin | ocean cells | core-area prevalence |
|---|---|---|---|
| *Caretta caretta* (loggerhead turtle) | Mediterranean | 13,683 | 0.292 |
| *Cetorhinus maximus* (basking shark) | NE Atlantic | 36,902 | 0.219 |
| *Balaenoptera musculus* (blue whale) | Northeast Pacific (California Current) | 42,476 | 0.052 |

All headline numbers are in **[`RESULTS.md`](RESULTS.md)**; the same
numbers as copy-paste LaTeX are in **[`tables.tex`](tables.tex)**.

---

## Method in one paragraph

A stochastic Lagrangian ensemble (deterministic drift from DUACS L4 geostrophic
`ugos`/`vgos`, plus a Gaussian random walk at `K = 10 m²/s`; 10 runs × 500
particles × 365 days) produces a per-cell **retention** field. Each ocean cell
becomes one row with six features — `retention`, `mean_speed`, `vorticity`,
`latitude`, `longitude`, `dist_to_coast` — and a binary label. Models predict
whether a cell falls inside the species' core area.

Two methodological choices are load-bearing and differ from the earlier version
of this work:

1. **Label — KDE core area.** The positive class is the **50% volume contour** of
   a sighting-weighted Gaussian KDE: cells are sorted by density and accumulated
   until they hold half the total probability mass. This is the standard
   utilization-distribution core-area isopleth. It replaces a median split over
   non-zero cells, which selects half the *cells* rather than half the *mass*.
2. **Cross-validation — contiguous leave-one-region-out.** Folds are five compact
   regions from k-means on coordinates (longitude scaled by `cos(mean latitude)`).
   This replaces `GroupKFold` over 3° blocks, which balances fold *sizes* and
   therefore interleaves the blocks — held-out cells ended up a median of ~0.5°
   from training data. Under contiguous LORO the median separation is 1.9–3.8°.

Both corrections make the task harder and the reported scores lower. Report them
as the honest numbers.

### Caveats worth keeping

- **Geography dominates.** Net of latitude/longitude and the static flow field,
  the retention feature adds little marginal predictive skill. Never quote
  random-CV numbers alone.
- **Regions can be unscoreable.** A contiguous region may contain no core-area
  cells at all — that is what genuine extrapolation looks like. Such folds are
  recorded in `folds_unscoreable`, never silently dropped (4/5, 5/5 and 3/5
  regions are scoreable for the three species).
- **AquaMaps is an external reference, not a peer.** It is a fixed, untrained
  0.5° surface scored on the identical test cells. It was never fitted to this
  data, but its occurrence base partly overlaps OBIS-SEAMAP, so the comparison
  is partly circular.

---

## Setup

```bash
pip install -r requirements.txt
```

### Data you must download

Occurrence CSVs (OBIS-SEAMAP gridded summaries) and the AquaMaps native-range
tables are committed under `Datasets/`. The three **velocity cubes are not** —
they total ~1.8 GB, over GitHub's limit.

They are **DUACS L4 geostrophic currents** (`ugos`/`vgos`) from the E.U.
Copernicus Marine Service, product `SEALEVEL_GLO_PHY_L4_MY_008_047`, dataset
`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`. Download with the
[`copernicusmarine`](https://pypi.org/project/copernicusmarine/) CLI using
exactly these bounds:

```bash
# Datasets/Oceanic_data.nc — Mediterranean  (732 days, grid 125 x 299)
copernicusmarine subset -i cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D \
  -v ugos -v vgos -x -0.062 -X 37.188 -y 26.312 -Y 41.812 \
  -t 2012-01-01 -T 2014-01-01 -o Datasets -f Oceanic_data.nc

# Datasets/NE_Atlantic.nc — NE Atlantic  (1036 days, grid 200 x 208)
copernicusmarine subset -i cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D \
  -v ugos -v vgos -x -27.938 -X -2.062 -y 40.062 -Y 64.938 \
  -t 2022-01-19 -T 2024-11-19 -o Datasets -f NE_Atlantic.nc

# Datasets/California_Current.nc — California Current  (1095 days, grid 240 x 280)
copernicusmarine subset -i cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D \
  -v ugos -v vgos -x -139.938 -X -105.062 -y 20.062 -Y 49.938 \
  -t 2001-01-01 -T 2003-12-31 -o Datasets -f California_Current.nc
```

> Use the **reprocessed** (`_my_`) product for anything historical; the
> near-real-time variants start in 2022. The altimetry record starts **1993**, so
> earlier occurrences are unrecoverable, and geostrophy is unreliable within ~5°
> of the equator (down-weighted, not dropped).

TabICLv2 weights are fetched from the Hugging Face Hub on first use, or supply a
local checkpoint with `--ckpt`.

---

## Reproducing the results

Run from the repository root, **in this order** — each step writes into the
`results.json` the previous one produced. Everything except the TabLLM step is
free and CPU-only.

```bash
# 1. main run: retention -> KDE core-area labels -> contiguous LORO CV
#    TabICLv2 and RF at full context; TabICLv2/RF/GBT/kNN/LogReg at k = 8,16,32
#    -> figures/<species>/results.json
python run_pipeline.py

# 2. score the untrained AquaMaps surface on the identical test cells
#    -> merges auc_aquamaps / f1_aquamaps into results.json
python aquamaps.py

# 3. tune the Random Forest baseline by nested cross-validation   (~35 min)
#    -> adds auc_rf_tuned / f1_rf_tuned + the selected configuration per fold
python tune_rf.py

# 4. tuned RF at k=32, and permutation importance from the tuned forest (~35 min)
#    REQUIRES step 3 -- it reuses the configuration tune_rf.py selected
#    -> adds kshot["32"]["rf_tuned"] and perm_importance_tuned
python tune_rf_kshot.py

# 5. figures: per-region confidence intervals for AUC and F1
#    -> figures/02_ci_auc.png, figures/03_ci_f1.png
python make_figures.py                      # against the fixed-config RF
python make_figures.py --rf-key rf_tuned \
    --out-dir figures/tuned                 # against the tuned RF (paper version)

# 6. all tables -> tables.tex (LaTeX) and RESULTS.md (markdown)
python make_tables.py

# 7. read out the comparison and every number the paper quotes
python compare_rf.py        # does the TabICLv2 advantage survive tuning?
python paper_numbers.py     # means, sds, worst folds, CIs, k=32, importances
```

The steps below are independent of each other and only read `results.json`
(steps 1-4 must have run):

```bash
# pooled tests over all 12 scoreable regions (sign-flip + Wilcoxon, Bonferroni m=2)
python pooled_tests.py

# dependence-aware check: Bayesian hierarchical correlated t-test (needs pystan)
#    -> figures/bayes_hierarchical.json    (SEED = 1; the paper averages seeds 1-3)
python bayes_hierarchical.py

# loggerhead with 15 regions instead of 5 (~95 min on 4 cores)
#    -> figures/Caretta_caretta/results_k15.json
python loro15_caretta.py

# geography-only baseline: lon/lat KDE classifier on the same test cells
#    -> figures/kde_baseline.json, figures/kde_baseline.png/.pdf
python kde_baseline.py

# loggerhead context-size sweep: time and AUC of TabICLv2 vs tuned/fixed RF
#    -> context_sweep.jsonl (resumable), then the figures
python bench_context_sweep.py
python make_context_sweep_figure.py   # figures/context_sweep_{time,auc}.*
python make_timing_figure.py          # figures/timing_context_vs_rf_tuned*.*

# KDE core-area label maps -> figures/label_map_<Species>.*
python make_label_maps.py
```

`time.csv` holds the earlier timing runs (2k-row and full context) that
`make_timing_figure.py` reads; the script that wrote it is not in the
repository.

The first run simulates the Lagrangian ensemble, which takes several minutes per
species. The field is cached to a hidden `.npy` under `figures/<species>/`, keyed
by cube filename, grid shape and step count, so re-runs are fast.

Everything is seeded (`SEED = 42`; the tuner's inner subsampling uses a separate
stream, `SEED + 9999`, so it cannot perturb the outer fold draws). Re-running any
step reproduces its numbers bit-for-bit. `n_jobs` is a scheduling knob only —
with `random_state` fixed, a forest fitted on one thread or many is identical.

### Tuning the Random Forest baseline

`tune_rf.py` exists so the headline comparison is against a baseline that got a
fair shot. `run_pipeline.py` fits one fixed configuration (200 trees, depth 10,
class-balanced); this selects over a 32-point grid — `max_depth`
$\in$ {5, 10, 20, None}, `min_samples_leaf` $\in$ {1, 5}, `max_features`
$\in$ {sqrt, 0.5}, `class_weight` $\in$ {balanced, None}.

Two properties make it honest rather than decorative:

- **The selection is nested.** The outer loop is the same contiguous
  leave-one-region-out split; hyperparameters are chosen by leave-one-region-out
  *within each outer training set*, so selection never sees the held-out region.
  Tuning on the test region would leak and would be a worse criticism than not
  tuning at all.
- **AUC and F1 are selected separately.** Choosing on AUC and then reporting F1
  would hand RF a configuration it was never tuned for. Each candidate is scored
  on both metrics in the same inner pass and the two winners are refit
  independently, so RF gets its best available shot on each metric.

`tune_rf_kshot.py` does the same for the $k=32$ row, but selects by stratified
4-fold CV *within the 32 exemplars* — region-based inner CV is impossible there,
since 32 class-balanced exemplars do not populate enough regions. It also
recomputes permutation importance from the tuned forest, because the published
attribution was computed with the fixed configuration.

Both replay `run_pipeline.py`'s RNG draw order and **assert** the replayed fold
sizes against the recorded ones before writing, so the tuned rows are scored on
byte-identical test cells or the run aborts.

### TabLLM row (needs a GPU)

TabLLM is scored by **label-verbalizer logprob** — `P("high") / (P("high") +
P("low"))` read from the answer-token distribution, which is what the TabLLM
method actually specifies. This requires an OpenAI-compatible endpoint; the
Anthropic Messages API does not return logprobs.

```bash
# 5. deploy an OpenAI-compatible vLLM server (Qwen2.5-72B-Instruct-AWQ), then:
modal deploy deploy/vllm_server.py
python run_tabllm.py --base-url https://<endpoint>/v1
```

`run_tabllm.py --dry-run` builds every prompt and reports the call count
without contacting the server (free). Responses are cached in
`figures/tabllm_cache.sqlite`, keyed by model id and prompt — that cache,
not a seed, is the reproducibility anchor, because LLM output is not
byte-deterministic. It is committed, so the TabLLM numbers are reproducible
without re-running the server.

TabLLM is given the **same test cells and the same 32 exemplars** as every other
k-shot model. Both `run_tabllm.py` and `aquamaps.py` replay the main run's RNG
draw order and then *assert* the replayed fold sizes against the recorded ones,
so a silent divergence cannot slip through.

---

## Repository layout

| file | role |
|---|---|
| `retention_core.py` | velocity loading, CF unpacking, Lagrangian ensemble, occurrence loading, feature matrix |
| `label_cv.py` | the label and the folds: KDE density, core-area label, k-means LORO folds, balanced subsampling, separation metric |
| `run_pipeline.py` | **main entry point** — all models, all species, all folds |
| `aquamaps.py` | AquaMaps native-range loading, zero-filling, regridding, and scoring on the pipeline's test cells |
| `tabllm_prompt.py` | cell loading, row serialization, prompt construction, exemplar sampling, leakage assertions |
| `anthropic_tabllm.py` | backend-agnostic LLM client: sqlite cache, vote ensemble, logprob and verbalized scoring, circuit breaker |
| `run_tabllm.py` | TabLLM at k=32 |
| `tune_rf.py` | nested-CV hyperparameter selection for RF-full; AUC and F1 configurations chosen separately |
| `tune_rf_kshot.py` | the same for the k=32 RF row, plus permutation importance from the tuned forest |
| `compare_rf.py` | paired per-region comparison of TabICLv2 against RF before and after tuning |
| `paper_numbers.py` | every quantity the paper reports, with a self-check against the published untuned values |
| `make_figures.py` | confidence-interval figures (`--rf-key rf_tuned` for the tuned baseline) |
| `make_tables.py` | emits `tables.tex` and `RESULTS.md` |
| `pooled_tests.py` | sign-flip and Wilcoxon tests pooled over the 12 scoreable regions |
| `bayes_hierarchical.py` | Bayesian hierarchical correlated t-test (regions nested in species, fold correlation 1/5) |
| `loro15_caretta.py` | loggerhead TabICLv2 vs tuned RF with 15 held-out regions |
| `kde_baseline.py` | geography-only (lon/lat KDE) baselines on the pipeline's test cells |
| `bench_context_sweep.py` | loggerhead timing/AUC benchmark as the context grows |
| `make_context_sweep_figure.py`, `make_timing_figure.py` | context-sweep and timing figures |
| `make_label_maps.py` | maps of the KDE core-area label |
| `extension/` | TabICLv2 vs MaxEnt on MarineSPEED — see [`extension/README.md`](extension/README.md) |
| `reports/` | report on TabICL context (KV) caching and its speed |
| `deploy/vllm_server.py` | Modal deployment of the vLLM server used for TabLLM |

Reproducibility: all randomness is seeded (`SEED = 42`; ensemble runs use
`42 + run_idx`). Do not introduce unseeded randomness.

Offline self-checks, no API calls and no cost:

```bash
python tabllm_prompt.py --selftest
```

---

## Results

`figures/` holds the generated artefacts:

```
figures/02_ci_auc.png              per-region AUC with confidence intervals
figures/03_ci_f1.png               per-region F1 with confidence intervals
figures/tuned/                     the same two figures against the tuned RF
figures/<species>/results.json     every per-fold, per-draw number
figures/tabllm_cache.sqlite        cached LLM responses (reproducibility anchor)
figures/bayes_hierarchical.json    Bayesian hierarchical test (P better / equivalent / worse)
figures/Caretta_caretta/results_k15.json   15-region loggerhead run
figures/kde_baseline.*             geography-only baseline
figures/context_sweep_*, figures/timing_context_vs_rf_tuned*   timing figures
figures/label_map_*                KDE core-area label maps
```

Each `results.json` carries both baselines side by side — `auc_rf_full` /
`f1_rf_full` for the fixed configuration and `auc_rf_tuned` / `f1_rf_tuned` for
the nested-CV-selected one, with `rf_tuned_params_auc` and `rf_tuned_params_f1`
recording what was chosen per fold. Nothing is overwritten, so the before/after
is auditable.

Headline: against a **nested-CV-tuned** Random Forest, TabICLv2 scores higher on
AUC in 9 of 12 held-out regions and on F1 in 8 of 12, but no per-species
difference is separable from zero — every 95% confidence interval on the paired
across-region difference spans zero except loggerhead F1, `[+0.012, +0.118]`.

Pooled and dependence-aware checks:

- **Pooled over the 12 regions** (`pooled_tests.py`): AUC +0.045, sign-flip
  p = 0.017 (0.033 after Bonferroni for two metrics); F1 +0.048, p = 0.118.
  These tests treat the regions as independent, which they are not.
- **Bayesian hierarchical test** (`bayes_hierarchical.py`; ROPE ±0.01, fold
  correlation 1/5): P(TabICLv2 better on a new species) ≈ 0.7 for AUC and
  ≈ 0.8 for F1, P(tuned RF better) ≈ 0.21–0.26. Suggestive, not conclusive.
- **15 regions, loggerhead** (`loro15_caretta.py`; 10 of 15 scoreable):
  AUC +0.023 (TabICLv2 higher in 10/10, Wilcoxon p = 0.002), F1 +0.101 (8/10,
  p = 0.049). Smaller regions sit closer to training data (median separation
  0.95° vs 2.40°), so this is nearer interpolation than the 5-region split.
- **Geography alone** (`kde_baseline.py`): a lon/lat-only KDE classifier scores
  AUC 0.973 / 0.827 / 0.929 (loggerhead / shark / whale) against TabICLv2's
  0.988 / 0.854 / 0.920 and the tuned RF's 0.899 / 0.829 / 0.902, so these
  labels cannot separate habitat learning from spatial interpolation. The
  MarineSPEED extension, which gives neither model coordinates, addresses that:
  TabICLv2 beats untuned MaxEnt by +0.016 AUC across 40 species (Wilcoxon
  p = 0.017, 95% bootstrap CI [+0.00006, +0.034]), mostly on species with few
  records.

Tuning matters, and not in the direction you would guess. It made RF's **AUC
slightly worse** (the fixed depth-10 cap was not handicapping it, and configurations
selected on the training regions did not transfer across a spatial gap), so the
AUC comparison was never unfair. But it substantially closed the **F1** gap: the
blue whale's advantage fell from +0.231 to +0.010, driven by one region where
RF's F1 went 0.123 to 0.884 on tuning alone. That gap had been a calibration
artifact of `class_weight="balanced"` at 5.2% prevalence, not a ranking
difference — `compare_rf.py` prints the before/after.

The defensible claim is therefore about cost, not accuracy: TabICLv2 matches a
tuned supervised baseline with no per-task tuning of its own. It is faster than
*tuning and fitting* the RF by nested CV (13.4× at a 500-row context), but a
single untuned RF fit is still 7–96× faster than TabICLv2. Neither
prompt-only nor foundation-model in-context learning exceeds the supervised
ceiling, and geography remains the dominant signal.

## License

MIT — see [`LICENSE`](LICENSE).
