# Lagrangian Particle Retention & Marine Megafauna

Do modeled **Lagrangian particle-retention zones** — regions where simulated drifters tend to
linger under the surface geostrophic flow — coincide with where **marine megafauna** are actually
sighted? This repository investigates that question and then asks a harder one: *can retention be
used to **predict** where animals dwell?*

Two case studies are wired up:

| Species | Common name | Basin |
|---|---|---|
| *Caretta caretta* | Loggerhead sea turtle | Mediterranean |
| *Cetorhinus maximus* | Basking shark | NE Atlantic |

---

## ⚠️ Read this first: the honest-metrics caveat

The headline correlation is real (Spearman ρ ≈ 0.36 for the loggerhead/Mediterranean case), **but**
a feature-ablation study (`ablation_study.py`) shows that **net of geography (lat/lon) and the
static flow field, the retention feature adds ~nothing to predictive skill** — the paper-style
scores are inflated by **spatial autocorrelation**.

Because of this, every predictive metric here is reported under **two** cross-validation schemes:

- **Random CV** — optimistic, comparable to the original publication.
- **Spatial-block / leave-region-out CV** — the honest number.

Do not quote the random-CV numbers alone. The generalised pipeline reports both by default.

---

## How the project evolved (why the layout looks the way it does)

This grew over ~6 months through three phases, and the code reflects that history:

1. **Correlation** — establish whether retention and sightings co-occur (statistics, KDE, Moran's I, GWR-lite).
2. **Prediction** — use retention (plus flow/geography features) to *predict* dwelling with classic ML (RandomForest, gradient boosting, DBSCAN/GMM clustering, MLP).
3. **In-context TabLLM** — apply the TabLLM idea ([arXiv 2210.10723](https://arxiv.org/abs/2210.10723)): serialize each ocean grid cell to natural-language text and have an LLM classify it HIGH/LOW dwelling — served **free** from a self-hosted [vLLM](https://github.com/vllm-project/vllm) model.

## Repository layout

**Original monolith** (authoritative source of the canonical figures `01–19`):

| File | Role |
|---|---|
| `enhanced_analysis.py` | One long, *Caretta*-hardcoded program: preprocessing → particle simulation → statistics → ML → figures → robustness. Run it from `Datasets/`. |

**Generalised, multi-species refactor** (any species / any basin):

| File | Role |
|---|---|
| `retention_core.py` | Single source of truth: `simulate_retention` / `load_sightings` / `build_features`. Reproduces the monolith exactly under default params/seeds. |
| `species_pipeline.py` | Per-species run (association stats + RF/GBR under random **and** spatial CV) + a pooled cross-species transfer model. |
| `species_figures.py` | The per-species / pooled figure battery (own numbering, under `figures/<species>/`). |
| `ablation_study.py` | The honest marginal-skill experiments (→ `figures/20_ablation.png`). |
| `alignment_check.py` | Pure diagnostic: does a velocity cube cover a species' records, and if not, what to download. |

**In-context TabLLM extension:**

| File | Role |
|---|---|
| `anthropic_tabllm.py` | `TabLLMClient` — one client, two backends: pay-as-you-go **Anthropic API** *or* a free self-hosted **vLLM** (OpenAI-compatible) server. sqlite response cache; N-vote ensemble; circuit-breaker. |
| `tabllm_pipeline.py` | Serialization + few-shot learning-curve harness vs classic baselines (→ figures `21`/`22`). |
| `aquax_benchmark.py` | External SDM benchmark vs AquaMaps/AquaX suitability (→ figure `23`). *(work in progress)* |

`CLAUDE.md` contains a deeper architecture write-up.

## Getting the data

The two ocean-velocity cubes are **not in this repo** (627 MB + 658 MB, over GitHub's limit). They
are **DUACS L4 geostrophic currents** (`ugos`/`vgos`) from the E.U. Copernicus Marine Service.

Download them into `Datasets/` with the [`copernicusmarine`](https://pypi.org/project/copernicusmarine/) CLI.
`alignment_check.py` prints ready-to-run `copernicusmarine subset` commands with the exact bounds a
given species' records need:

```bash
python alignment_check.py --velocity Datasets/NE_Atlantic.nc \
  --species-csv "Datasets/obis_seamap_custom_.../...dist_sp_1deg_csv.csv" \
  --species "Cetorhinus maximus"
```

> The altimetry record starts **1993**; occurrences before then are unrecoverable, and geostrophy is
> unreliable within ~5° of the equator (down-weighted, not dropped).

The occurrence CSVs (OBIS-SEAMAP gridded-summary exports) **are** included under `Datasets/`.

## Install & run

```bash
pip install -r requirements.txt
```

```bash
# 1) Original monolith — canonical figures 01–19 (open bare filenames, so run FROM Datasets/)
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
paid Anthropic API. The vLLM server runs on a free Google Colab T4 GPU behind a cloudflared tunnel —
see the notebook below — and the WSL/local pipeline hits the tunnel URL:

```bash
export VLLM_BASE_URL="https://<your-tunnel>.trycloudflare.com/v1"
python tabllm_pipeline.py --species "Caretta caretta" \
  --velocity Datasets/Oceanic_data.nc --species-csv Datasets/caretta_data.csv \
  --model unsloth/Llama-3.2-3B-Instruct --smoke        # $0 via vLLM
```

The served model id (`--model`) flows into the response-cache key, so switching backends/models
cleanly invalidates old entries. Responses are cached in `figures/<species>/tabllm_cache.sqlite`
(committed — it, not the seed, is the reproducibility anchor, since LLM output isn't byte-deterministic).

## The Google Colab vLLM server

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Peter650-bot/vLLM/blob/main/colab/vllm_server.ipynb)

`colab/vllm_server.ipynb` boots an OpenAI-compatible vLLM server on a **free Colab T4**, exposes it
via a **cloudflared** quick-tunnel, and prints a `https://….trycloudflare.com/v1` URL. Export that as
`$VLLM_BASE_URL` on the machine running the pipeline. The notebook also covers the CUDA-version fix,
a keep-alive heartbeat, and which models a T4 can serve (3B for full runs; up to a 4-bit 14B for
smoke tests). The tunnel URL changes every Colab session — re-export it each run.

## Reproducibility

All randomness is seeded (`np.random.default_rng(42)` and friends). If you modify `retention_core.py`,
re-run the *Caretta*/Mediterranean case and confirm Spearman ρ ≈ 0.362 and the figure `01–19` numbers
are unchanged — that equivalence is the project's de-facto regression test.

## License

Source code is released under the **MIT License** (see `LICENSE`). Datasets (OBIS-SEAMAP occurrences,
DUACS/Copernicus velocities) retain their **own upstream licenses** and are not relicensed here —
please cite the original providers.
