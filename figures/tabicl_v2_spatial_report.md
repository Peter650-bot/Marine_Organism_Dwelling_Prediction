# TabICLv2 vs RandomForest (full data) — spatial-block 5-fold CV

Positive class: HIGH dwelling. Decision threshold P(HIGH) = 0.5, no per-fold tuning. TabICL is not fitted: the whole training fold is passed as in-context data in a single forward pass.

## Summary

| species | model | AUC mean | AUC sd | AUC worst | F1 mean | F1 sd | F1 worst |
|---|---|---|---|---|---|---|---|
| Caretta caretta | TabICLv2 | 0.9208 | 0.0870 | 0.7670 | 0.8187 | 0.0879 | 0.6772 |
| Caretta caretta | RF-full | 0.8834 | 0.1535 | 0.6114 | 0.8153 | 0.1650 | 0.5292 |
| Cetorhinus maximus | TabICLv2 | 0.9585 | 0.0254 | 0.9293 | 0.8595 | 0.0952 | 0.7059 |
| Cetorhinus maximus | RF-full | 0.8735 | 0.0716 | 0.7570 | 0.7915 | 0.0743 | 0.7048 |
| Balaenoptera musculus | TabICLv2 | 0.9914 | 0.0066 | 0.9830 | 0.8982 | 0.0616 | 0.7952 |
| Balaenoptera musculus | RF-full | 0.9604 | 0.0233 | 0.9314 | 0.8203 | 0.1183 | 0.6301 |

## Paired difference (TabICLv2 − RF-full), over folds

| species | metric | Δ mean | 95% CI | p | wins |
|---|---|---|---|---|---|
| Caretta caretta | ROC-AUC | +0.0373 | [-0.0454, +0.1200] | 0.2786 | 4/5 |
| Caretta caretta | F1 | +0.0034 | [-0.1272, +0.1340] | 0.9457 | 3/5 |
| Cetorhinus maximus | ROC-AUC | +0.0850 | [+0.0220, +0.1480] | 0.0200 | 5/5 |
| Cetorhinus maximus | F1 | +0.0680 | [-0.0294, +0.1654] | 0.1248 | 4/5 |
| Balaenoptera musculus | ROC-AUC | +0.0310 | [+0.0094, +0.0526] | 0.0163 | 5/5 |
| Balaenoptera musculus | F1 | +0.0779 | [+0.0064, +0.1494] | 0.0390 | 5/5 |

## Caretta caretta — per fold

TabICL weights: unrecorded.

Reproduction check vs `tabicl_results.json`: `auc_tabicl_max_abs_diff=0`, `auc_rf_full_max_abs_diff=0`

| fold | AUC TabICLv2 | AUC RF-full | F1 TabICLv2 | F1 RF-full |
|---|---|---|---|---|
| 0 | 0.9755 | 0.9738 | 0.8333 | 0.9568 |
| 1 | 0.9476 | 0.9323 | 0.8940 | 0.8580 |
| 2 | 0.9692 | 0.9711 | 0.8873 | 0.8593 |
| 3 | 0.9444 | 0.9286 | 0.8015 | 0.8730 |
| 4 | 0.7670 | 0.6114 | 0.6772 | 0.5292 |
| **mean** | **0.9208** | **0.8834** | **0.8187** | **0.8153** |
| source | `figures/Caretta_caretta/tabicl_f1_spatial.json` | | | |

## Cetorhinus maximus — per fold

TabICL weights: v2. 36902 ocean cells, prevalence 0.263, 88 spatial blocks, 29531 context rows/fold, 300 balanced test cells/fold.

| fold | AUC TabICLv2 | AUC RF-full | F1 TabICLv2 | F1 RF-full |
|---|---|---|---|---|
| 0 | 0.9689 | 0.9168 | 0.9175 | 0.8656 |
| 1 | 0.9937 | 0.9447 | 0.9521 | 0.8730 |
| 2 | 0.9390 | 0.8716 | 0.7059 | 0.7633 |
| 3 | 0.9617 | 0.8774 | 0.8794 | 0.7509 |
| 4 | 0.9293 | 0.7570 | 0.8425 | 0.7048 |
| **mean** | **0.9585** | **0.8735** | **0.8595** | **0.7915** |
| source | `figures/Cetorhinus_maximus/tabicl_f1_spatial.json` | | | |

## Balaenoptera musculus — per fold

TabICL weights: v2. 67200 ocean cells, prevalence 0.084, 132 spatial blocks, 53760 context rows/fold, 300 balanced test cells/fold.

| fold | AUC TabICLv2 | AUC RF-full | F1 TabICLv2 | F1 RF-full |
|---|---|---|---|---|
| 0 | 0.9952 | 0.9817 | 0.7952 | 0.6301 |
| 1 | 0.9830 | 0.9314 | 0.9155 | 0.8643 |
| 2 | 0.9931 | 0.9512 | 0.8945 | 0.7955 |
| 3 | 0.9865 | 0.9510 | 0.9310 | 0.8696 |
| 4 | 0.9994 | 0.9868 | 0.9547 | 0.9419 |
| **mean** | **0.9914** | **0.9604** | **0.8982** | **0.8203** |
| source | `figures/Balaenoptera_musculus/tabicl_f1_spatial.json` | | | |

## Reading notes

Test subsamples are class-balanced, so the trivial all-HIGH baseline is F1 ~ 0.667 (not 0). With 5 spatial blocks a t-CI has 4 df -- indicative only, and it can extend past a bounded metric's maximum.
