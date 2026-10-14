# Local measurement: TabICLv2 kv_cache on this project's data (CPU)

Source: measured in this session (2026-10-03) with bench_kvcache.py (session scratchpad). Hardware: WSL2, 4 logical cores, 5.9 GB RAM + 12 GB swap, CPU-only torch 2.13, tabicl 2.1.1, checkpoint tabicl-classifier-v2-20260212, n_estimators=2, n_jobs=None (PyTorch default threads). Data: the paper's real folds (spatial k-means leave-one-region-out, replayed from run_pipeline.py), 6 features, 300-cell class-balanced test sets. One process per (species, mode). 'predict2' = a second, independently drawn 300-cell test set from the same held-out region predicted against the same fitted context (stands in for a new scenario/time slice). Times in seconds.

## Loggerhead (Caretta caretta, Mediterranean), 4 scoreable folds, context 10.3k-11.2k rows

| fold | context | mode | fit | predict1 | fit+predict1 | predict2 | AUC |
|---|---|---|---|---|---|---|---|
| 0 | 10908 | none | 2.0 | 136.1 | 138.2 | 231.2 | 0.99276 |
| 0 | 10908 | repr | 16.1 | 90.2 | 106.3 | 95.8 | 0.99276 |
| 0 | 10908 | kv | 269.2 | 13.4 | 282.6 | 6.4 | 0.99276 |
| 1 | 11073 | none | 4.0 | 115.4 | 119.4 | 110.0 | 0.96080 |
| 1 | 11073 | repr | 33.8 | 132.6 | 166.4 | 86.2 | 0.96080 |
| 1 | 11073 | kv | 139.6 | 3.2 | 142.8 | 2.6 | 0.96080 |
| 3 | 11239 | none | 1.2 | 112.7 | 113.9 | 104.2 | 0.99920 |
| 3 | 11239 | repr | 12.2 | 112.2 | 124.4 | 91.3 | 0.99920 |
| 3 | 11239 | kv | 113.4 | 3.7 | 117.1 | 3.1 | 0.99920 |
| 4 | 10345 | none | 1.7 | 91.5 | 93.2 | 101.4 | 0.99836 |
| 4 | 10345 | repr | 12.3 | 81.4 | 93.7 | 78.8 | 0.99836 |
| 4 | 10345 | kv | 114.7 | 2.9 | 117.7 | 2.6 | 0.99836 |

Means over 4 folds:

| mode | fit | predict1 | fit+predict1 | predict2 | peak RSS (GB, whole process) |
|---|---|---|---|---|---|
| none | 2.2 | 113.9 | 116.2 | 136.7 | 2.29 |
| repr | 18.6 | 104.1 | 122.7 | 88.0 | 1.87 |
| kv | 159.2 | 5.8 | 165.1 | 3.7 | 3.08 |

Full-basin prediction (all 13,683 cells) on fold 0: none 514.6 s (after a 2.0 s fit), repr 374.1 s (after a 16.1 s fit), kv 131.6 s (after a 269.2 s fit).

Fold 0 was slow in every mode (first fold of each process; likely warm-up / machine load); folds 1,3,4 are the cleaner comparison.

AUC is identical across modes on every fold (to 5 decimals); a synthetic smoke test gave max |p_cached - p_uncached| = 1.3e-4 -> caching is exact up to float noise.

## Blue whale (Balaenoptera musculus, California Current), fold 2 only, context 33,912 rows

| mode | fit | predict1 | fit+predict1 | predict2 | AUC | peak RSS GB |
|---|---|---|---|---|---|---|
| none (from earlier benchmark time.csv, same fold) | 1.8 | 794.2 | 796.0 | not measured (~= predict1 without cache) | 0.88689 | n/a |
| repr | 60.3 | 797.9 | 858.2 | 758.5 | 0.88689 | 3.23 |
| kv | 1142.4 | 89.1 | 1231.4 | 21.8 | 0.88689 | 5.01 |

kv on the whale fold did NOT run out of memory but peaked at 5.0 GB on a 5.9 GB machine (with swap available); a larger context or more ensemble members (default n_estimators=8 instead of 2) would not fit in RAM here.

## Comparison points from the earlier benchmark (time.csv, same folds, same machine)

Random Forest (200 trees, fixed config) fit+predict: loggerhead mean 1.9 s, blue whale 2.2 s. RF with full nested-CV tuning (32 configs x inner folds + refits): loggerhead mean 140 s, blue whale 110 s.

## Derived facts (arithmetic on the tables above)
- Loggerhead folds 1,3,4: single fit+predict no-cache 109 s vs kv 126 s (1.16x). Repeat predict no-cache 105 s vs kv 2.8 s (38x faster).
- Blue whale fold 2: single fit+predict no-cache 796 s vs kv 1231 s (1.55x). Repeat predict repr 758 s (~no-cache) vs kv 22 s (35x faster).
- Break-even, loggerhead (folds 1,3,4 means): kv becomes cheaper in cumulative wall-clock from predict call #2 on the same context. For 10 calls: no-cache 1056 s vs kv 151 s.
- Break-even, blue whale fold 2: kv becomes cheaper in cumulative wall-clock from predict call #2 on the same context. For 10 calls: no-cache 7622 s vs kv 1427 s.
- Even with kv, TabICLv2 fit+predict (117-1231 s) remains slower than a fixed RF (2 s) and comparable to or slower than a fully tuned RF (110-140 s). Caching does not change the TabICL-vs-RF single-task comparison; it changes the cost of *re-using* one species' context for many predictions.
