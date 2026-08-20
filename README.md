# Lagrangian Particle Retention & Marine Megafauna

Do modeled **Lagrangian particle-retention zones** — regions where simulated drifters tend to
linger under the surface geostrophic flow — coincide with where **marine megafauna** are actually
sighted? This repository investigates that question, then asks a harder one: *can retention be
used to **predict** where animals dwell?* — and a harder one still: *does a tabular foundation
model that was never trained on this problem match a supervised model that was?*

Three species are wired up, each in its own ocean basin:

| Species | Common name | Basin | Velocity cube | Grid cells | Prevalence |
|---|---|---|---|---|---|
| *Caretta caretta* | Loggerhead sea turtle | Mediterranean | 2012-01 → 2014-01 | 13,683 | 0.435 |
| *Cetorhinus maximus* | Basking shark | NE Atlantic | 2022-01 → 2024-11 | 36,902 | 0.263 |
| *Balaenoptera musculus* | Blue whale | California Current | 2001-01 → 2003-12 | 67,200 | 0.084 |

The blue whale is cosmopolitan; the California Current cube covers **one basin of its range by
design**, so results characterise that basin, not the species globally.

---

## ⚠️ Read this first: the honest-metrics caveat

The headline correlation is real (Spearman ρ ≈ 0.36 for the loggerhead/Mediterranean case), **but**
a feature-ablation study (`ablation_study.py`) shows that **net of geography (lat/lon) and the
static flow field, the retention feature adds ~nothing to predictive skill** — the paper-style
scores are inflated by **spatial autocorrelation**.

Because of this, every predictive metric here is reported under **two** cross-validation schemes:

- **Random CV** — optimistic, comparable to the original publication.
- **Spatial-block / leave-region-out CV** — the honest number.

Do not quote the random-CV numbers alone. The generalised pipeline reports both by default, and
**every headline result below is spatial-block CV only**.

---

## Headline result

TabICLv2 (a tabular foundation model, **not** an LLM) is given the training fold as *in-context
data in a single forward pass* — nothing is fitted — and compared against a RandomForest trained
on the same fold. Spatial-block 5-fold CV, decision threshold P(HIGH) = 0.5, test subsamples
class-balanced at 300 cells (so the trivial all-HIGH baseline is F1 = 0.667, not 0):

| Species | AUC TabICL vs RF | Δ | *p* | F1 TabICL vs RF | Δ | *p* |
|---|---|---|---|---|---|---|
| *Caretta caretta* | 0.9208 vs 0.8834 | +0.037 | 0.279 | 0.8187 vs 0.8153 | +0.003 | 0.946 |
| *Cetorhinus maximus* | 0.9585 vs 0.8735 | +0.085 | **0.020** | 0.8595 vs 0.7915 | +0.068 | 0.125 |
| *Balaenoptera musculus* | 0.9914 vs 0.9604 | +0.031 | **0.016** | 0.8982 vs 0.8203 | +0.078 | **0.039** |

**Fine-tuning barely helps.** Loading the pretrained checkpoint, fine-tuning it on each training
fold, then scoring the identical cells moves AUC by **+0.006 pooled over all 15 folds** (12/15
folds positive, Wilcoxon *p* = 0.009; no single species significant) and F1 by −0.002 (n.s.). So
the zero-training result is not leaving measurable headroom on the table — a useful negative
result, since fine-tuning would forfeit the zero-shot claim entirely.

---

## Repository layout

**Original monolith** (authoritative source of the canonical figures `01–19`):

| File | Role |
|---|---|
| `enhanced_analysis.py` | One long, *Caretta*-hardcoded program: preprocessing → particle simulation → statistics → ML → figures → robustness. Run it from `Datasets/`. |

**Generalised, multi-species refactor** (any species / any basin):

| File | Role |
|---|---|
| `retention_core.py` | Single source of truth: `simulate_retention` / `load_sightings` / `build_features` / `spatial_groups`. Reproduces the monolith exactly under default params/seeds. |
| `species_pipeline.py` | Per-species run (association stats + RF/GBR under random **and** spatial CV) + a pooled cross-species transfer model. |
| `species_figures.py` | The per-species / pooled figure battery (own numbering, under `figures/<species>/`). |
| `ablation_study.py` | The honest marginal-skill experiments (→ `figures/20_ablation.png`). |
| `alignment_check.py` | Pure diagnostic: does a velocity cube cover a species' records, and if not, what to download. |

**Tabular foundation model (TabICLv2):**

| File | Role |
|---|---|
| `tabicl_benchmark.py` | TabICL vs classical baselines, k-shot and full-context, both CV schemes. Pins and **verifies** the v2 checkpoint from the loaded config, not the flag. |
| `tabicl_f1_benchmark.py` | The headline spatial-CV run: TabICL vs RF-full, AUC **and** F1, per-fold checkpointing so a killed run resumes. |
| `tabicl_v2_report.py` | Cross-species report → `figures/tabicl_v2_spatial_report.{md,json}`. |
| `export_cells.py` | Freezes a species' cell matrix + its exact CV split to `.npz`, **verified against the committed results** before writing. Feeds the Modal run. |
| `deploy/tabicl_finetune_modal.py` | Three-arm fine-tuning study on a Modal GPU: zero-shot / fine-tuned / fine-tuned-with-full-context. |

**In-context TabLLM extension** (serialize each cell to text, let an LLM classify it):

| File | Role |
|---|---|
| `anthropic_tabllm.py` | `TabLLMClient` — one client, two backends: pay-as-you-go **Anthropic API** *or* a free self-hosted **vLLM** (OpenAI-compatible) server. sqlite response cache; N-vote ensemble; circuit-breaker. |
| `tabllm_pipeline.py` | Serialization + few-shot learning-curve harness vs classic baselines (→ figures `21`/`22` per species). Also owns `load_cells`, which every TabICL script reuses. |
| `deploy/vllm_server.py`, `colab/vllm_server.ipynb` | Self-hosted vLLM on Modal or a free Colab T4. |

**Figure generators:**

| File | Writes |
|---|---|
| `ci_errorbar_figure.py` | `figures/21_ci_auc.png`, `figures/22_ci_f1.png` — per-model CIs, all three species |
| `tabicl_vs_rf_figure.py` | `TabICL_vs_RF_full_AUC.png` |
| `tabicl_vs_rf_roc.py` | `figures/Caretta_caretta/26_roc_tabicl_vs_rf_spatial.png` |
| `spatial_kshot_figure.py` | `figures/Caretta_caretta/25_spatial_kshot_tabicl.png` |

`CLAUDE.md` contains a deeper architecture write-up.

## Getting the data

The three ocean-velocity cubes are **not in this repo** (~630 MB + ~690 MB + ~560 MB, over GitHub's
limit). They are **DUACS L4 geostrophic currents** (`ugos`/`vgos`) from the E.U. Copernicus Marine
Service, product `SEALEVEL_GLO_PHY_L4_MY_008_047`, dataset
`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` (reprocessed, 1993 → present).

Download them into `Datasets/` with the [`copernicusmarine`](https://pypi.org/project/copernicusmarine/)
CLI. `alignment_check.py` prints ready-to-run `copernicusmarine subset` commands with the exact
bounds a given species' records need:

```bash
python alignment_check.py --velocity Datasets/NE_Atlantic.nc \
  --species-csv "Datasets/obis_seamap_custom_.../...dist_sp_1deg_csv.csv" \
  --species "Cetorhinus maximus"
```

> Use the **reprocessed** (`_my_`) product for anything historical. The near-real-time variants
> start in 2022 and cannot cover earlier occurrences. The altimetry record starts **1993**;
> occurrences before then are unrecoverable, and geostrophy is unreliable within ~5° of the
> equator (down-weighted, not dropped).

The occurrence CSVs (OBIS-SEAMAP gridded-summary exports) **are** included under `Datasets/`.

Occurrences are joined to the velocity grid **spatially only**, by linear interpolation of
weighted record counts onto the cube's mesh (`build_features`). Time is a filter and a weight, not
a join key: a record survives if its midpoint date falls within the cube's period ±2 years, then
carries a Gaussian temporal weight (σ = 365 d) about the cube midpoint.

## Install & run

```bash
pip install -r requirements.txt
```

```bash
# 1) Original monolith — canonical figures 01–19 (opens bare filenames, so run FROM Datasets/)
cd Datasets && python ../enhanced_analysis.py

# 2) Feature ablation — the honest marginal-skill check (writes figures/20_ablation.png)
python ablation_study.py --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv

# 3) Full per-species + pooled cross-species run
python species_pipeline.py \
  --run "Caretta caretta:Datasets/Oceanic_data.nc:Datasets/caretta_data.csv" \
  --run "Cetorhinus maximus:Datasets/NE_Atlantic.nc:Datasets/<multispp>.csv"
```

### TabLLM (LLM-based prediction)

Free offline checks cost nothing:

```bash
python tabllm_pipeline.py --selftest   # offline unit checks
python tabllm_pipeline.py --probe --species "Caretta caretta" \
  --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv
```

To actually classify, point the pipeline at a **free self-hosted vLLM server** (recommended) or the
paid Anthropic API:

```bash
export VLLM_BASE_URL="https://<your-tunnel>.trycloudflare.com/v1"
python tabllm_pipeline.py --species "Caretta caretta" \
  --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv \
  --model unsloth/Llama-3.2-3B-Instruct --scoring logprob --smoke
```

Use `--scoring logprob` (score = `P("high")/(P("high")+P("low"))` from the answer-token
distribution) rather than the default `verbalized`: LLMs emit round numbers, and across 39k cached
calls only 66 distinct scores appeared, tying 26% of positive/negative pairs and capping AUC at
0.681 no matter how the ties broke. Logprob scoring needs an OpenAI-compatible `--base-url`; the
Anthropic Messages API does not return logprobs.

## The Google Colab vLLM server

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Peter650-bot/Marine_Organism_Dwelling_Prediction/blob/main/colab/vllm_server.ipynb)

`colab/vllm_server.ipynb` boots an OpenAI-compatible vLLM server on a **free Colab T4**, exposes it
via a **cloudflared** quick-tunnel, and prints a `https://….trycloudflare.com/v1` URL. Export that
as `$VLLM_BASE_URL` on the machine running the pipeline. The tunnel URL changes every Colab
session — re-export it each run.

---

## Reproducibility

Everything below regenerates the numbers in **Headline result** from the committed artefacts. The
velocity cubes are the only external dependency.

### What is committed, and why

| Artefact | Why it is in the repo |
|---|---|
| `figures/<species>/tabicl_f1_spatial.json` | Per-fold AUC/F1 **plus the raw `proba_*` vectors and `y_true`** — so threshold analyses and CIs are recomputable without a GPU or a re-run. |
| `figures/<species>/tabicl_finetune_fullctx.json` | The three-arm fine-tuning study (zero-shot / fine-tuned / fine-tuned + full context). |
| `figures/<species>/cells_export.npz` | The frozen cell matrix and CV split (< 1 MB each). |
| `figures/tabicl_v2_spatial_report.{md,json}` | The cross-species table. |
| `figures/<species>/tabllm_cache.sqlite` | Every LLM response. **This, not a seed, is the TabLLM reproducibility anchor** — LLM output is not byte-deterministic. |
| `figures/<species>/*.run.log` | Wall-clock provenance for the long CPU runs. |

The 110 MB TabICL checkpoint is **not** committed (over GitHub's file limit). It is auto-downloaded
from the HF Hub on first use, or pass `--model-path`.

### Determinism

All randomness is seeded: `np.random.default_rng(42)` (main), `42 + run_idx` (ensemble), `77`
(seasonal), `99` (sensitivity). Per-fold seeds are derived by
`sha256(f"{fold}|{scheme}|{species}|{k}|{variant}")`, so a fold's test subsample is a pure function
of its identity and does not depend on execution order.

The TabICL full-context spatial run is **bit-for-bit reproducible**: `tabicl_f1_benchmark.py`
re-checks each fold's AUC against `tabicl_results.json` and aborts on drift.

### Reproducing the headline table

```bash
# per-species spatial-CV run (resumable — each fold is checkpointed as it finishes)
python tabicl_f1_benchmark.py --species "Caretta caretta"
python tabicl_f1_benchmark.py --species "Cetorhinus maximus"
python tabicl_f1_benchmark.py --species "Balaenoptera musculus"

# the cross-species table (reads the artefacts above; refits nothing)
python tabicl_v2_report.py

# the CI figures
python ci_errorbar_figure.py
```

`--species` resolves the velocity cube and occurrence CSV from `SPECIES_PRESETS`, so the two
extension species need no path flags. Expect **hours per species on CPU** (measured:
~12 min/fold for *Cetorhinus*, 34-94 min/fold for *Balaenoptera*); a GPU makes it seconds.

### Reproducing the fine-tuning study

```bash
python export_cells.py --all          # verifies the split, then freezes it (needs the cubes)
modal run deploy/tabicl_finetune_modal.py --folds 0,1,2,3,4 --species "Caretta caretta"
```

`export_cells.py` **aborts** unless every fold's seed, training-fold size, and all 300 test labels
match the committed `tabicl_f1_spatial.json`. A successful export is therefore proof that the GPU
run scores the published split rather than a remote re-derivation of it. Each container also
re-scores the fold zero-shot, so the fine-tuned/zero-shot delta is free of hardware drift (measured
GPU-vs-CPU zero-shot difference: ≤ 0.0018 AUC).

Cost on Modal A10G: ~$0.01 for one pilot fold, ~$0.15 per species for all five.

### Regression test

There is no test suite. The de-facto regression test is numerical equivalence: if you modify
`retention_core.py`, re-run the *Caretta*/Mediterranean case and confirm **Spearman ρ ≈ 0.362** and
that the figure `01–19` numbers are unchanged. `tabllm_pipeline.py --selftest` runs the offline
serialization/leakage unit checks for free.

### Known non-determinism

- **LLM output** is not byte-deterministic — hence the committed response cache.
- **Root figures `01–20` are cited by number.** Renaming or renumbering breaks the write-up.
  Per-species figures live in `figures/<species>/` under independent numbering.

---

## License

Source code is released under the **MIT License** (see `LICENSE`). Datasets (OBIS-SEAMAP
occurrences, DUACS/Copernicus velocities) retain their **own upstream licenses** and are not
relicensed here — please cite the original providers.
