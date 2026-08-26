# Results

Headline results under the two corrections:

1. **Label** — 50% volume contour of a weighted Gaussian KDE of sighting
   intensity (the genuine core-area isopleth), replacing the median of
   non-zero cells on a linearly-interpolated surface.
2. **Cross-validation** — contiguous leave-one-region-out (k-means, 5
   regions), replacing `GroupKFold` over 3° blocks, which scattered the
   blocks and left held-out cells ~0.5° from training data.

LaTeX for every table is in `tables.tex`.

## Headline table

| species | cells | prevalence | regions scored | separation (median °) | TabICLv2 AUC | RF-full AUC | TabICLv2 F1 | RF-full F1 |
|---|---|---|---|---|---|---|---|---|
| Caretta caretta | 13,683 | 0.292 | 4 / 5 | 2.40 | 0.9878 | 0.9089 | 0.8689 | 0.7924 |
| Cetorhinus maximus | 36,902 | 0.219 | 5 / 5 | 1.93 | 0.8543 | 0.8459 | 0.7861 | 0.7469 |
| Balaenoptera musculus | 42,476 | 0.052 | 3 / 5 | 3.80 | 0.9203 | 0.8964 | 0.7022 | 0.4710 |

## In-context learning at k=32

Mean ROC-AUC over held-out regions and exemplar draws.

| model | Caretta caretta | Cetorhinus maximus | Balaenoptera musculus |
|---|---|---|---|
| TabICLv2 | 0.8616 | 0.7894 | 0.8918 |
| TabLLM (Qwen2.5-72B) | 0.5553 | 0.6603 | 0.8929 |
| Random Forest | 0.8221 | 0.8023 | 0.8595 |
| GBT | 0.8354 | 0.7214 | 0.7890 |
| kNN | 0.6559 | 0.6998 | 0.7287 |
| LogReg | 0.3738 | 0.4230 | 0.7934 |

## Against a nested-CV-tuned Random Forest

`run_pipeline.py` fits one fixed RF configuration; `tune_rf.py`
selects hyperparameters by leave-one-region-out *within each outer
training fold*. Both baselines are kept so the change is auditable.

| species | metric | TabICLv2 | RF fixed | RF tuned | Δ vs tuned | 95% CI |
|---|---|---|---|---|---|---|
| Caretta caretta | AUC | 0.9878 | 0.9089 | 0.8992 | +0.0885 | [-0.102, +0.279] |
| Caretta caretta | F1 | 0.8689 | 0.7924 | 0.8040 | +0.0650 | [+0.012, +0.118] |
| Cetorhinus maximus | AUC | 0.8543 | 0.8459 | 0.8291 | +0.0252 | [-0.050, +0.100] |
| Cetorhinus maximus | F1 | 0.7861 | 0.7469 | 0.7299 | +0.0562 | [-0.023, +0.136] |
| Balaenoptera musculus | AUC | 0.9203 | 0.8964 | 0.9021 | +0.0182 | [-0.031, +0.068] |
| Balaenoptera musculus | F1 | 0.7022 | 0.4710 | 0.6922 | +0.0099 | [-0.492, +0.512] |

## Not re-run

- **Fine-tuning (zero-shot vs fine-tuned)** — runs on Modal and bills
  GPU credits. Its published values were produced under the superseded
  label and CV scheme and are not carried forward.
