# Extension: TabICLv2 vs MaxEnt on MarineSPEED (environment only, spatial folds)

## Why
In the main study, a lon/lat-only KDE ties or beats TabICLv2 and RF
(`../kde_baseline.py`). That means the turtle/shark/whale labels cannot tell
habitat learning apart from spatial interpolation. This extension asks a
cleaner question on an established marine SDM benchmark:

- **No coordinates** are given to either model.
- **Spatially blocked** test sets are used.
- The baseline is **MaxEnt**, the standard presence-only SDM.

## Data
**MarineSPEED V1** (Bosch et al. 2018, *Diversity and Distributions* 24:144–157),
from `https://rshiny.lifewatch.be/marinespeed/data/V1/`.

- The full set has 514 species, with Bio-ORACLE and MARSPEC predictors at every
  occurrence and at 20,000 random background points.
- **Folds.** MarineSPEED's own spatial "pseudodisc" 5-fold CV. Test presences
  lie in a held-out disc, and background points within 200 km of it are
  excluded. Random folds were dropped at the user's request.
- **Predictors.** 15 of the 68 candidate layers, chosen by a greedy filter
  (|Spearman ρ| < 0.7 on the background, in priority order starting with SST,
  salinity, bathymetry, distance to shore and chlorophyll). The list is in
  `results/predictors.json`. Longitude and latitude are dropped.
- **Species.** 40 species, stratified by dominant ecoregion and record-count
  tercile with seed 42. That gives 3 polar, 21 temperate and 16 tropical
  species, with 73 to 32,788 records each (`results/species.json`).
- **Caps per fold** (fixed seed, identical for both models):
  - at most 5,000 training presences and 5,000 training background points;
  - at most 2,000 test presences and 2,000 test background points.

## Models
- **TabICLv2.** `n_estimators=2`, seed 42, the local v2 checkpoint, CPU, 4 torch threads.
- **MaxEnt.** `elapid` 1.0.3 `MaxentModel` with the maxent.jar defaults:
  - beta 1.0, cloglog output, clamping on;
  - feature classes by the jar's rule on training presences: ≥80 → LQPHT, 15–79 → LQH, 10–14 → LQ, otherwise L.
  - No per-species tuning of the regularisation (beta) was done.
- **AUC.** Held-out presences vs held-out background.
- **Statistics.** Each species' AUC is the mean of its 5 folds. The species is
  the replication unit for the Wilcoxon signed-rank test and the 10,000-sample
  bootstrap.

## Results (40 species × 5 spatial folds)

| | TabICLv2 | MaxEnt |
|---|---|---|
| Mean ROC-AUC | **0.870** | 0.854 |
| Median time per fold (fit + predict) | 39 s | 3.6 s |
| Total time | 2.7 h | 0.5 h |

**TabICLv2 − MaxEnt across species:**
- mean +0.016, with 95% bootstrap CI [+0.000, +0.034];
- median +0.012;
- Wilcoxon p = 0.017;
- TabICLv2 is higher for 30 of 40 species.

**The edge is small, and it shrinks as the data grow.**

| Subset | n | Mean Δ | TabICLv2 higher |
|---|---|---|---|
| Few records (lowest tercile) | 14 | +0.029 | 12 |
| Mid | 14 | +0.019 | 10 |
| Many records (top tercile) | 12 | −0.002 | 8 |
| Polar | 3 | −0.013 | 1 |
| Temperate | 21 | +0.021 | 15 |
| Tropical | 16 | +0.016 | 14 |

- **Mean driven by two species.** Most of the mean comes from two species where
  MaxEnt fails: *Platyxanthus patagonicus* (+0.21) and *Zostera noltei* (+0.16).
- **Large TabICLv2 losses** are on cold or pelagic species with many records:
  - *Eukrohnia hamata* and *Limacina retroversa*, both −0.10;
  - minke whale, −0.06.

**Context sweep** (6 species, spatial fold 1; `figures/sweep_time.pdf`):

| Context | TabICLv2 AUC | MaxEnt AUC | Time ratio |
|---|---|---|---|
| 500 | 0.954 | 0.876 | 149× |
| 1,000 | 0.953 | 0.878 | 33× |
| 2,000 | 0.942 | 0.909 | 13× |
| 5,000 | 0.926 | 0.920 | 5.3× |
| Full (6,649) | 0.924 | 0.919 | 4.9× |

- **TabICLv2 gets worse with more context** on spatially held-out data.
  - Minke whale: 0.984 → 0.923.
  - *Pictilabrus*: 0.985 → 0.919.
- **MaxEnt improves** with more context.
- Both trends fit the per-tercile table: TabICLv2's advantage is in small samples.
- Caveat: this is 6 species on a single fold. MaxEnt's low score at small
  context is mostly one species (*Auxis*, 0.53).

## Reading
1. **TabICLv2 is not *only* interpolating geographically.**
   - It sees no coordinates.
   - It is tested on spatially disjoint discs.
   - It still matches or slightly beats an untuned MaxEnt.
   - So it extracts environmental signal that transfers.
2. **"Interpolation" is not ruled out.** The environment is itself spatially
   autocorrelated, so this tests transfer to new places, not novel conditions.
   The context sweep, where more training rows lower held-out AUC, hints that
   with large contexts TabICLv2 leans on local, environment-space
   nearest-neighbour structure that transfers less well.
3. **The win is modest and fragile.**
   - The CI lower bound is about 0.
   - It comes mainly from small-sample species and two MaxEnt failures.
   - MaxEnt was run with default settings, without ENMeval-style beta tuning.
     Tuning would likely narrow the gap.
4. **Cost.** TabICLv2 is about 5× (full context) to over 100× (small context)
   slower than MaxEnt per fold.

## Reproduce
```bash
python3 extension/fetch_marinespeed.py                    # ~190 MB -> extension/data/
pip install --target ~/venvs/maxent-site elapid==1.0.3    # MaxEnt only; keeps main sklearn
bash extension/run_all.sh                                 # sweeps + 40 x 5 folds, ~4 h CPU
python3 extension/evaluate.py                             # results/summary.json, figures/auc_paired
python3 extension/make_sweep_figure.py                    # figures/sweep_time
```
Runners are resumable: they skip finished (species, fold, context, draw) keys.
`evaluate.py` recomputes every AUC from the stored test labels and scores.
