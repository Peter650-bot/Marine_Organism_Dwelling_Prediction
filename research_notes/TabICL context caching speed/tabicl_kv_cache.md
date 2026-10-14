# TabICL context caching (kv_cache) and inference speed

Scope: `tabicl` Python package (soda-inria; Jingang Qu, David Holzmüller et al.), TabICLv1 (ICML 2025, arXiv 2502.05564) and TabICLv2 (arXiv 2602.11139). Researched 2026-10-03. Latest release is **v2.2.0 (2026-09-02)**; the project machine has **v2.1.1 (2026-04-29)** installed. Source-code facts below were read directly from the tagged files on GitHub (raw.githubusercontent.com/soda-inria/tabicl/<tag>/...). "Local" numbers come from the coordinator's smoke test, not from an online source.

## 1. When was kv_cache introduced, and what do the official docs say about it?

### Takeaway
`kv_cache` came with the TabICLv2 code release, **v2.0.0 (2026-02-12)**. Both modes, `"kv"` and `"repr"`, and the `save(..., save_kv_cache=True)` persistence path were there from the first version. Before that (v0.0.x–v0.1.4, the TabICLv1 era) the package had no cache: `fit()` only fit the preprocessors, and every `predict()` ran the transformer over train+test. In **v2.0.3 (2026-03-02)**, `kv_cache` moved from a `fit()` argument to a constructor argument. The README pitches the cache for "repeated inference on the same training data".

### Cited Findings
- v2.0.0 release (2026-02-12): "Release the code of TabICLv2". — [GitHub releases](https://github.com/soda-inria/tabicl/releases)
- In v2.0.0, `kv_cache` was a **`fit()` parameter**. The fit docstring at that tag already lists `False`, `True`/`"kv"` and `"repr"`, with the same "~24x less memory than 'kv' for the ICL part" wording. (Read from `src/tabicl/sklearn/classifier.py` at tag v2.0.0.) — [classifier.py @ v2.0.0](https://github.com/soda-inria/tabicl/blob/v2.0.0/src/tabicl/sklearn/classifier.py)
- v2.0.3 release notes (2026-03-02), under "Breaking Changes": "`kv_cache` moved from `fit()` to `__init__()` following scikit-learn convention. `kv_cache` is now a constructor parameter for both `TabICLClassifier` and `TabICLRegressor`." The same release fixed a cache dtype bug: "When AMP is enabled, cached projections stored in float16 caused errors when loaded on CPU/MPS/CUDA without AMP. The cache is now auto-upcast to float32 during loading." Commit df09cb38 ("Change `kv_cache` as init parameter, fix KV cache dtype issues, and add MPS support"). — [GitHub releases](https://github.com/soda-inria/tabicl/releases)
- v2.2.0 (2026-09-02) cache-related fixes: PR #124 "fix get/set state model_kv_cache_" (pickling fix, merged 2026-06-05) and PR #121 "only set model_ when unpickling if pickled model was fitted". — [v2.2.0 release](https://github.com/soda-inria/tabicl/releases/tag/v2.2.0); [PR #124](https://github.com/soda-inria/tabicl/pull/124)
- README: "To speed up repeated inference on the same training data, enable KV caching. The cache is built during `fit` and reused across `predict` calls." Example given: `clf = TabICLClassifier(kv_cache=True); clf.fit(X_train, y_train)  # caches key-value projections for training data; clf.predict(X_test)  # fast: only processes test data by reusing the cached context`. — [README](https://github.com/soda-inria/tabicl)
- API docs (readthedocs): kv_cache "Controls caching of training data computations to speed up subsequent predict_proba/predict calls. The cache is built during fit()". False = no caching. True/"kv" = "Cache key-value projections from both column embedding and ICL transformer layers. Fast inference but memory-heavy for large training sets". "repr" = "Cache column embedding KV projections and row interaction outputs (representations). Uses ~24x less memory than 'kv' for the ICL part, at the cost of re-running the ICL transformer at predict time." — [API docs](https://tabicl.readthedocs.io/en/latest/api.html)
- v2.2.0 docstring addition on dtype: "The cache retains whatever dtype the model produced during fit() (float16 when AMP is active, float32 otherwise). If the cache is later loaded on a device without AMP, the tensors are automatically upcast to float32 ... With AMP enabled, reduced-precision caches are kept on CPU, CUDA, XPU, and MPS." — [classifier.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/classifier.py)
- Pre-cache behaviour (Issue #6, March 2025, v0.0.x). Maintainer D. Holzmüller: "`fit()` only fits preprocessors, while `predict()` performs a forward pass of the Transformer with the entire training+test data." A user with 500k×30 data saw fit take under 10 s but `predict` on 20 rows "costs a long time". — [Issue #6](https://github.com/soda-inria/tabicl/issues/6)
- The cache also works with **TabICLv1 checkpoints** in the v2 codebase. J. Qu (Issue #88, March 2026): "The current codebase should support kv cache for both TabICLv1 and TabICLv2", e.g. `TabICLClassifier(checkpoint_version="tabicl-classifier-v1-20250208.ckpt", kv_cache=True, device="cpu")`. He adds that v1 reproducibility in the v2 codebase is attempted "but no guarantee". Holzmüller notes the `n_estimators` default fell from 32 (v0.1.x) to 8 (v2.x). — [Issue #88](https://github.com/soda-inria/tabicl/issues/88)

### Inferences
- If you call `fit(..., kv_cache=...)` on v2.0.0–v2.0.2, it will not work on ≥2.0.3. On the local v2.1.1, pass it to the constructor, as the coordinator already does.
- "Repeated predictions on the same training set" is the only use case the docs endorse. With a single predict call per fit, the cache saves nothing, because building it already costs one train-side forward pass.

### Gaps
- No standalone CHANGELOG file entry announces the feature. The v2.0.0 release note is one line, so there is no "introduced in" note beyond the tag contents.
- I found no official benchmark tables that time cache vs no-cache.

## 2. What each cache mode stores, how it is used at predict time, and memory cost

### Takeaway
`"kv"` stores per-layer K/V projections of the training rows for the column-embedding ISABs and **all 12 ICL transformer layers**. At predict time, only the test rows go through the network, attending to the cached train K/V. `"repr"` stores the column-embedding K/V plus the row-interaction output (the 512-d train representations with y already embedded). Predict then skips the column and row stages for the training rows but still runs the full 12-block ICL transformer over train+test. ICL memory per training row per ensemble member works out to about 48 KB for kv vs about 2 KB for repr in fp32 (my estimate). The cache is built once per ensemble member and per normalization method.

### Cited Findings
- Module docstring of `_model/kv_cache.py`: "The caching strategy focuses on: 1. ColEmbedding: Cache K/V of the second attention layer of ISAB blocks 2. ICLearning: Cache K/V from training data at each layer of the ICL transformer". The container `TabICLCache` holds `col_cache`, `row_repr` ("Cached row representations"), `icl_cache`, `train_shape (batch_size, train_size, num_features)` and `num_classes`. It has `cache_size_mb()` ("Return the memory occupied by cached tensors in MB") and `cache_type()` returning 'kv', 'repr' or 'empty'. — [kv_cache.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_model/kv_cache.py)
- `TabICL.forward_with_cache` docstring: "'repr': Cache column embedding KV projections and row interaction outputs (representations with y_train baked in). Uses ~24x less memory for the ICL part, at the cost of re-running the ICL transformer." With `store_cache=True` it "Processes training data and stores cached values". With `use_cache=True` it "Uses cached values for training data". — [tabicl.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_model/tabicl.py)
- Cache building (`_build_kv_cache`). For each normalization method, the ensemble's training views are split into chunks of `batch_size`, each chunk goes through `forward_with_cache(X_train, y_train, use_cache=False, store_cache=True, cache_mode=...)` under `torch.no_grad()`, and the chunk caches are concatenated into `model_kv_cache_[norm_method]`. At predict time, `_batch_forward_with_cache` slices the cache to match each test batch. Docstring of `model_kv_cache_`: "Pre-computed KV caches for training data, keyed by normalization method." — [classifier.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/classifier.py)
- Restriction: "KV caching is not supported when the number of classes ... exceeds the max number of classes ... natively supported by the model" (raises ValueError). — [classifier.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/classifier.py)
- Column-embedding cache detail: in the ISAB, the inducing points "will only attend to training data in the first attention", and the cache stores "K/V of the second attention layer". — [layers.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_model/layers.py)
- Reported memory (Issue #92, 2026-04-01, user): "with a sample size of around 40,000 and about 250 features, the cache requires roughly 5GB of memory". Maintainer J. Qu's reply: "Setting `kv_cache="repr"` can largely reduce memory consumption at the cost of more computations." — [Issue #92](https://github.com/soda-inria/tabicl/issues/92)
- Persisted caches are moved to CPU on save ("KV cache — Saved by default (moved to CPU for device portability)"). — [base.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/base.py)
- Context from a third-party blog (C. Molnar, 2026-05-12). Citing the TabPFN-2.5 paper, it says TabPFN's comparable cache needs "6.1 KB of GPU memory and 48.8 KB of CPU memory per cell". That is a **TabPFN** figure, not TabICL. — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models)

### Inferences
- **Where "~24x" comes from (my derivation, consistent with the docstring).** The ICL transformer has 12 blocks, each with a K and a V projection of the 512-d row embedding: 12 × 2 × 512 values per training row. "repr" stores one 512-d vector per row instead, a 24× ratio.
- **Memory estimate (my arithmetic, assuming float32 and that the ICL K/V width equals the 512-d ICL dim).** For the ICL part, "kv" ≈ 12 × 2 × 512 × 4 B ≈ **48 KB per training row per ensemble member**. "repr" ≈ 512 × 4 B ≈ **2 KB**. Halve both under fp16 AMP. With the default n_estimators=8 and 3,000 training rows: kv ≈ 1.2 GB, repr ≈ 50 MB. At 40k rows, kv ≈ 2 GB per member in fp32 (~1 GB in fp16), the same order as the 5 GB in Issue #92 (whose n_estimators and dtype are unknown). Memory scales linearly with `n_train × n_estimators`.
- The column cache stores K/V of the second ISAB attention, which comes from the 128 inducing-point summaries. Its size should therefore depend on the number of features and estimators, **not on n_train**. This is inferred from the code comments; I did not measure it.
- **Why local "repr" predict is no faster than no-cache (2.3 s fit / 18.2 s predict vs 5.2 s / 18.5 s).** On CPU, predict cost is dominated by the 12-block ICL transformer over train+test, which "repr" still re-runs by design. The column and row stages it skips are cheap. So **"repr" saves memory relative to "kv", not time relative to no-cache, on this kind of data.**
- **Why local "kv" fit is slow (28.7 s vs 2.3 s).** With caching, the whole train-side transformer pass moves from `predict` into `fit`. A 28.7 s fit is still more than the 18.2 s no-cache predict. Plausible cause: the store-cache path does extra per-layer K/V materialisation and concatenation. Issue #155 notes that the store-cache column path keeps batch dims and loses the skip fast-path (see section 4). This is not measured. The break-even is roughly 2 predict calls on the same fit (28.7 + 1.6k vs 2.3 + 18.2k seconds for k calls → about 1.6 calls).

### Gaps
- No official per-row memory formula or measured cache sizes. The estimates above are mine. `TabICLCache.cache_size_mb()` could measure them exactly on the local install.
- No official explanation of why cache building is slower than an uncached forward pass.

## 3. What do the TabICL and TabICLv2 papers say about inference cost, masking, timings and caching?

### Takeaway
Both papers state the test→train-only attention pattern that makes caching exact: train rows attend to train rows, test rows attend only to train rows. Neither paper presents KV caching as a contribution (I found no mention of it). Their timings are for single fit+predict runs. TabICLv1 reports GPU (A100) timings only. TabICLv2 reports H100 and CPU speedups over TabPFN-2.5, plus disk/CPU offloading for million-row tables.

### Cited Findings
- **TabICLv1 (arXiv 2502.05564):** complexity O(m²n + n²): column embedding O(nkm) with k inducing vectors, row interaction O(m²n), ICL O(n²). Compare TabPFNv2's O(m²n + n²m). Masking: "embeddings in H_test can only attend to H_train". — [TabICL paper](https://arxiv.org/html/2502.05564)
- TabICLv1 timings (A100): "TabICL is 1.5× faster than TabPFNv2 on small datasets and 3-10× faster on large datasets"; about 1 s vs about 2 s at 1K samples × 10 features; about 20 s vs about 100 s at 10K × 100. "geometric mean training+inference time per 1K samples for TabICL is 1.1 seconds". Memory: 100K samples × 500 features ≈ "5 GB of GPU memory and 32 GB of RAM"; 500K × 500 "less than 14GB of GPU memory". "intermediate activations can be offloaded to CPU and disk as needed". FlashAttention and AMP used. No CPU-only results reported. — [TabICL paper](https://arxiv.org/html/2502.05564)
- **TabICLv2 (arXiv 2602.11139, 2026-02-11; README lists it as ICML 2026):** "test samples attend to training samples for prediction". Efficiency: "we reduce redundant computation by selectively computing Q/K/V projections" (row-wise, dataset-wise ICL and layer-norm reuse). Speed: "10.6× on an H100 GPU at 50K samples" vs TabPFN-2.5 with 100 features; "The efficiency gap is even more pronounced on CPU, reaching 11.8× at just 10K samples." Scale: disk offloading reduces "requirements to under 24 GB CPU and 50 GB GPU to process a table with 1M samples and 500 features within 450 seconds". Model ≈ 28M parameters. — [TabICLv2 paper](https://arxiv.org/html/2602.11139)
- README (v2.x): "On an H100 GPU, TabICLv2 can `fit` and `predict` a dataset with 50,000 samples and 100 features in under 10 seconds, which is 10x faster than TabPFN-2.5". Complexity is given as O(n² + nm²). — [README](https://github.com/soda-inria/tabicl)

### Inferences
- The test→train-only mask is exactly what makes `"kv"` caching lossless: train-row K/V never depend on test rows. This matches the coordinator's local check (predictions identical within 1.3e-4; the remainder is float noise).
- With a cache, predict cost becomes roughly linear in n_test × n_train (one test-to-train attention per layer), compared with ≈ (n_train + n_test)² × layers without it. This explains the 18.2 s → 1.6 s local drop.

### Gaps
- I could not pull the CPU model or thread count, or the absolute CPU seconds, behind the "11.8× at 10K" claim (the WebFetch summary did not surface them). Check TabICLv2 appendix H directly.
- No paper reports a fit/predict time split or cache-vs-no-cache timings.

## 4. GitHub issues, discussions and third-party reports on slow CPU inference, cache memory, and speed

### Takeaway
Only a handful of reports exist. Cache memory (Issue #92) and pre-cache slow predict (Issue #6) are the main ones. One independent blog measured about a 5.5× cache speedup and about 30× GPU-over-CPU speedup on a small dataset. The authors steer very large data away from TabICL, and the docstring says bf16 AMP on CPU is "much slower than fp32".

### Cited Findings
- Issue #6 (v0.0.x): very slow `predict` on 500k rows. Maintainer: "TabICL will probably not perform well either at this data size since it has never seen such large datasets during pre-training. It is probably better to use more classical tabular methods for this data size." — [Issue #6](https://github.com/soda-inria/tabicl/issues/6)
- Issue #92 (2026-04): user reports "The KV cache is incredibly helpful and significantly speeds up model inference" but about 5 GB for about 40k × 250. Maintainer recommends `kv_cache="repr"`. — [Issue #92](https://github.com/soda-inria/tabicl/issues/92)
- Issue #155 (2026-09-07, open; PR #156). In the released config (`target_aware=True`), the column-stage `skip_value` fast path for reserved CLS columns "is never taken". Measured on CPU with tabicl 2.2.0 (same code in 2.1.1): whole-model "speedup on CPU is ~1.05x–1.1x up to 25 features" once fixed, and logits are bit-identical. The cache path `_compute_embeddings_with_cache, store_cache=True` also destroys the sentinel, but "no FLOPs are lost there". — [Issue #155](https://github.com/soda-inria/tabicl/issues/155)
- C. Molnar, "How to make Tabular Foundation Model inference faster" (2026-05-12), UCI breast cancer (455 train rows × 31 cols). `kv_cache=True` cut prediction "from 0.5 seconds to 0.09 seconds". On Colab, CPU ≈ 15 s vs GPU ≈ 0.5 s ("30x improvement"). "GPU inference plus KV caching can improve prediction time by well over 100x". TabICL is described as "on average, the fastest TFM" per TabArena. — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models)
- Open PR #160 (2026-09-15): "Restore PyTorch thread count when prediction fails". This implies the estimator changes the global torch thread count during predict. — [PR #160](https://github.com/soda-inria/tabicl/pull/160)

### Inferences
- The authors have made no explicit "CPU is unsuitable" statement. The design and docs are GPU-oriented (AMP and FA3 auto-heuristics, H100 benchmarks), and CPU support is handled through `n_jobs` and the offloading machinery. The local 18 s for 3,000 + 300 rows with 2 estimators is in line with Molnar's CPU-vs-GPU gap.

### Gaps
- No GitHub Discussions on CPU speed turned up. Searches of issues for "slow" and "CPU" returned only the items above.

## 5. Other official inference knobs and their effect on speed

### Takeaway
On CPU, the knobs that matter are `n_estimators` (cost is roughly linear in it), `kv_cache`, `n_jobs` (torch threads), and `batch_size` (memory vs throughput). `use_amp` and `use_fa3` default to off on CPU ("auto"). FA3 is CUDA/Hopper-only. Forcing bf16 AMP on CPU was "much slower" in the authors' benchmarks. `offload_mode` controls where column-embedding outputs live and matters for memory, not CPU speed.

### Cited Findings
- `n_estimators` (default 8): "Number of estimators for ensemble predictions". README: more members improve accuracy but increase runtime. In v0.1.x the default was 32. — [API docs](https://tabicl.readthedocs.io/en/latest/api.html); [README](https://github.com/soda-inria/tabicl); [Issue #88](https://github.com/soda-inria/tabicl/issues/88)
- `batch_size` (default 8): "Batch size for inference. If None, all ensemble members are processed in a single batch". README: reduce to save memory. Cache building and cached predict are also chunked by `batch_size` (code). — [API docs](https://tabicl.readthedocs.io/en/latest/api.html); [classifier.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/classifier.py)
- `n_jobs` (default None): "Number of threads to use for PyTorch on CPU. None uses PyTorch default (physical CPU cores)." Introduced in v0.0.3 (PR #4). v0.1.3 fixed the "thread count calculation". v2.2.0 PR #107: "n_threads is set to the maximum of n_logical_cores and n_jobs, rather than minimum". — [API docs](https://tabicl.readthedocs.io/en/latest/api.html); [releases](https://github.com/soda-inria/tabicl/releases)
- `use_amp` (default "auto"): "Controls automatic mixed precision (AMP) for inference", enabled by a size heuristic (small: off; medium n<10240: AMP on; large n≥10240: AMP and FA3 on). v2.2.0 docstring: "On CPU, use_amp="auto" stays off (explicit use_amp=True still enables bfloat16 autocast, but local benchmarks found it much slower than fp32)". The heuristic "assumes that the training set is large relative to the test set and does not account for KV-cache scenarios. If it is suboptimal for your workload, set it explicitly." — [classifier.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/classifier.py)
- `use_fa3` (default "auto"): "Flash Attention 3 that can speed up inference for large datasets on NVIDIA Hopper GPUs like H100. Only effective on CUDA when FA3 is installed; a no-op on CPU / MPS / XPU." v2.2.0 PR #154: "Use grad scaler for float32 + FlashAttention-3". — [API docs](https://tabicl.readthedocs.io/en/latest/api.html); [v2.2.0 release](https://github.com/soda-inria/tabicl/releases/tag/v2.2.0)
- `offload_mode` (default 'auto'): "Controls where column-wise embedding outputs are stored during inference". Options are 'auto', 'gpu'/False, 'cpu'/True, and 'disk' (memory-mapped). v2.0.1 fixed CUDA-stream race conditions in cpu/disk offloading. v2.0.2 added "structured offloading fallback". v2.2.0 (PR #152) respects cgroup memory limits when detecting CPU memory. — [API docs](https://tabicl.readthedocs.io/en/latest/api.html); [releases](https://github.com/soda-inria/tabicl/releases)
- `inference_config`: "fine-grained control over the three transformers in TabICL" (added v0.0.3). `device=None` auto-selects CUDA → XPU → MPS → CPU. — [API docs](https://tabicl.readthedocs.io/en/latest/api.html); [README](https://github.com/soda-inria/tabicl)

### Inferences
- With no cache on CPU, the cheapest speed lever is a lower `n_estimators` (8 → 4 or 2 roughly halves or quarters the time, at some accuracy cost). `kv_cache="kv"` is the lever when the same training set is reused for many predict calls (CV folds reuse it only within a fold).
- Raising `batch_size` (or None) processes more ensemble members per forward pass. On CPU it is unlikely to beat a multithreaded fp32 GEMM by much, and it raises peak memory. Unmeasured.

### Gaps
- No official CPU benchmark of `n_jobs` scaling or `batch_size` effects.

## 6. Can a saved/pickled fitted model with its cache be reused across sessions?

### Takeaway
Yes, and this is officially supported. `est.save(path, save_model_weights=False, save_training_data=True, save_kv_cache=True)` has existed since v2.0.0. It pickles the fitted estimator **including the KV cache**, moved to CPU. `TabICLClassifier.load(path, device=...)` (or plain `pickle`/`joblib`) restores it, so `predict_proba` in a new session reuses the cached context without re-encoding the training set. You can even drop the raw training data and predict from the cache alone.

### Cited Findings
- `save()` docstring (v2.2.0): "The saved file can be loaded back using the class's load() method, pickle.load(), or joblib.load()". `save_model_weights` (default False): if False, "model weights are reloaded from the checkpoint on load, resulting in a smaller file". `save_training_data` (default True): if False, "training data is excluded for a smaller file and better data privacy. This is only allowed when save_kv_cache is True and a KV cache exists". `save_kv_cache` (default True): "If False, the KV cache is excluded, resulting in a smaller file but requiring a rebuild for fast inference." — [base.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/base.py)
- `__getstate__`: "KV cache — Saved by default (moved to CPU for device portability)". Training data "Can be excluded when KV cache is present, which is useful for data privacy since the KV cache encodes the training context and enables prediction without raw data." `model_` and `device_` are always excluded and rebuilt on load. Defaults apply to a direct `pickle.dump` too (cache kept). Version metadata (sklearn/torch/numpy/python) is embedded. — [base.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/base.py)
- `load(path, device=None)`: if a device is given, it moves the model, rebuilds the inference config and calls `_move_cache_to_device()`. Predicting without training data or a cache raises: "Cannot predict: this estimator was saved without training data and has no KV cache." — [base.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/base.py); [classifier.py @ v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/classifier.py)
- `save()` with `save_kv_cache` is present in `sklearn/base.py` at tags v2.0.0 and v2.0.3 (verified in the source). README: "When KV cache exists and is saved, you can set `save_training_data=False` to exclude cached training data". — [base.py @ v2.0.0](https://github.com/soda-inria/tabicl/blob/v2.0.0/src/tabicl/sklearn/base.py); [README](https://github.com/soda-inria/tabicl)
- Pickling bugs fixed only in **v2.2.0**: PR #124 "fix get/set state model_kv_cache_" (the author says "I missed the `model_kv_cache_` in the previous PR") and PR #121 / Issue #116 (unpickling / `check_is_fitted`). — [v2.2.0 release](https://github.com/soda-inria/tabicl/releases/tag/v2.2.0); [PR #124](https://github.com/soda-inria/tabicl/pull/124)
- v2.0.3: an fp16 (AMP) cache loaded on a non-AMP device is upcast to fp32 to avoid dtype errors. — [GitHub releases](https://github.com/soda-inria/tabicl/releases)

### Inferences
- The persisted file holds the full cache, so with "kv" its size is about cache memory (see section 2 estimates). With n_estimators=8 and thousands of training rows, that can reach GBs.
- **Version caveat for the local install (v2.1.1).** PR #124 says `model_kv_cache_` was mishandled in get/set state before v2.2.0, so cache round-tripping on 2.1.1 may be buggy. Upgrade to ≥2.2.0, or verify on 2.1.1 that a reloaded estimator still has `model_kv_cache_` and gives the same predictions with similar predict time. I did not read the pre-fix diff, so the exact failure mode on 2.1.1 is unknown.
- The cache is tied to one training set and its ensemble preprocessing. Any change (new fold, new features, new rows) needs a new fit and a new cache. Reuse across sessions therefore only amortizes repeated prediction on a fixed training set, e.g. predicting many grid cells or climate scenarios from one fitted model.

### Gaps
- No published timing of save/load or file sizes for cached models.
- Exact pre-2.2.0 pickling failure mode not confirmed (PR #124 body is one line).
