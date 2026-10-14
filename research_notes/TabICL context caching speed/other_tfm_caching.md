# Context caching and other inference-time shortcuts across tabular foundation models (TabPFN v2/2.5/3/3.5, TabDPT, Mitra, LimiX, TabFlex, AutoGluon) compared with TabICL

Scope note: written 2026-10-03. Package facts come from reading the source and README of the current PyPI wheels: `tabpfn` 9.1.0 and `tabicl` 2.2.0. I downloaded and unpacked both. Paper facts come from arXiv full-text PDFs that I downloaded and searched. Where a number comes from a figure's embedded text rather than prose, that is flagged. Hardware is given for every timing, and CPU and GPU are kept separate. A sibling file (`tabicl_kv_cache.md`) covers TabICL's cache in depth, so TabICL appears here only as the comparison anchor.

## Q1. TabPFN (PriorLabs `tabpfn`): what do the `fit_mode` options do, what is cached, what are the documented speedups and memory costs, CPU limits, and how do `inference_precision`, `memory_saving_mode` and `n_estimators` affect speed?

### Takeaway
TabPFN has four fit modes. Only `fit_with_cache` stores the transformer's key/value (KV) state for the training rows. The default `fit_preprocessors` caches only the preprocessed training data, so every `.predict()` still runs the full train+test forward pass. PriorLabs' own figures for the cache range widely. Their docs cookbook shows about 5.5x on one predict call (5,000 rows, hardware not stated). The TabPFN-3 report on an H100 shows 15–30x (22 ms vs 342–642 ms) and "1–3 orders of magnitude" at large n. Building the cache costs about the same as one uncached fit+predict, so caching only pays off when you predict more than once against the same fit. On CPU the package refuses training sets above 1,000 rows for v2/v2.5/v2.6 and above 5,000 rows for v3/v3.5 unless you override it.

### Cited Findings
**`fit_mode` semantics (tabpfn 9.1.0 docstring, `TabPFNClassifier.__init__`)**
- The signature is `fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"] = "fit_preprocessors"`. The docstring frames fitting this way: "the 'fitting' is technically the forward pass of the model". — [tabpfn source, classifier.py](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py)
- `"low_memory"`: "the data is preprocessed on-demand during inference when calling .predict() ... most memory-efficient mode but can be slower for large datasets because the data is (repeatedly) preprocessed on-the-fly. Ideal with low GPU memory and/or a single call to .fit() and .predict()." — [tabpfn classifier.py](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py)
- `"fit_preprocessors"` (default): "the data is preprocessed and cached once during the .fit() call. During inference, the cached preprocessing (of the training data) is used instead of re-computing it. Ideal with low GPU memory and multiple calls to .predict() with the same training data." — [tabpfn classifier.py](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py)
- `"fit_with_cache"`: preprocessing is cached as above, and "the transformer key-value cache is also initialized, allowing for much faster inference on the same data at a large cost of memory. Ideal with very high GPU memory and multiple calls to .predict() with the same training data." — [tabpfn classifier.py](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py)
- `"batched"`: iterates over data already preprocessed with `get_preprocessed_datasets`. It is "primarily used only for inference ... in Fine-Tuning" and is set internally by `fit_from_preprocessed()`. It is not a speed mode for normal use. — [tabpfn classifier.py](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py)
- What the KV-cache engine stores, per the docstring in `inference.py`:
  - "Each ensemble member (estimator) has its own KV cache".
  - At predict "only X_test is preprocessed" and the model is called with `x_is_test_only=True`.
  - `keep_cache_on_device=True` (the default) keeps the caches on the GPU. False moves them to CPU after building.
  - For TabPFN-3 checkpoints the default `kv_cache_precision` is `"int8"` (per-tensor symmetric quantisation, dequantised on the fly). `"fp8"`, `"auto"` and `"adaptive"` also exist. Architectures that cannot quantise fall back to `"auto"`.
  - [tabpfn source](https://github.com/PriorLabs/TabPFN)
- Large test sets under `fit_with_cache` are chunked with `TABPFN_MAX_BATCHED_TEST_ROWS` (default 32,768).
  - README: "Test rows are conditionally independent given the KV cache, so chunking is mathematically equivalent". Results may differ slightly because of floating-point non-associativity.
  - "the hardware is already saturated at that chunk size", so larger chunks bring no speedup.
  - [TabPFN README](https://github.com/PriorLabs/TabPFN)
- The out-of-memory (OOM) error text separates the two cases. For a large test set, batch it. For a "Large training set — batching won't help. Subsample your training data". — [tabpfn errors.py](https://github.com/PriorLabs/TabPFN)
- A fitted estimator, including its fit state, can be saved and reloaded with `save_fitted_tabpfn_model` / `load_fitted_tabpfn_model`. — [TabPFN README](https://github.com/PriorLabs/TabPFN)

**Documented speedups and memory for `fit_with_cache`**
- PriorLabs cookbook. Setup: synthetic data, 5,000 rows, 20 features, binary, 95/5 split, warm-up fit excluded. **Hardware and model version are not stated.**

  | Scenario | Default mode (fit / predict) | `fit_with_cache` (fit / predict) | Predict speedup |
  |---|---|---|---|
  | One predict | 1.46 s / 0.91 s | 1.36 s / 0.17 s | ~5.5x |
  | 10 repeated predicts | 1.14 s / 4.23 s total | 1.34 s / 1.57 s total | ~2.7x |

  — [PriorLabs docs: faster performance with cache](https://docs.priorlabs.ai/cookbook/faster_performance_with_cache)
- PriorLabs KV-cache page:
  - Recommended when "You make repeated predictions against the same fit, including repeated calls from explainability tools".
  - Memory is "Higher; scales with training-set size and number of estimators".
  - "A cache is valid only for the training data and model configuration used to create it".
  - The hosted API accepts at most 10,000 test rows per cached call, and caching is not compatible with "Thinking mode".
  - [PriorLabs docs: KV cache](https://docs.priorlabs.ai/capabilities/kv-cache)
- August 2026 changelog: "Up to 10× faster repeat inference with KV cache". This applies to `tabpfn-client` 0.4.2 and the hosted API. — [PriorLabs changelog Aug 2026](https://docs.priorlabs.ai/changelog/august-2026)
- TabPFN-3 report (12 May 2026), **H100 GPU**, single estimator, no preprocessing, ntrain = 50,000, ntest = 100. Values are read from the embedded text of Figure 7b.

  | Path | 10 features | 100 features |
  |---|---|---|
  | TabPFN-2.5 fit+predict | 1,224 ms | ~7.8 s (broken axis, approximate) |
  | TabPFN-3 cold fit+predict | 342 ms | 642 ms |
  | TabPFN-3 build cache | 341 ms | 642 ms |
  | TabPFN-3 cached predict | 22 ms | 22 ms |

  Prose from the report:
  - "cached-predict is one to three orders of magnitude faster than either the TabPFN-2.5 baseline or TabPFN-3's own cold 'fit+predict' path ... between 0.1 and 3 ms/test point for batches of 100".
  - "The fit-with-cache call costs essentially the same as the cold fit+predict at every measured shape".
  - [TabPFN-3 Technical Report, arXiv 2605.13986](https://arxiv.org/abs/2605.13986)
- TabPFN-3 memory. Test-row queries use a single KV head (multi-query attention), which gives "a KV-cache size of 7GiB per estimator for 1M rows datasets". That makes the default 8 estimators "usable on common GPUs". By inference, 8 estimators would need about 56 GiB at 1M rows. — [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)
- TabPFN-3 explainability: shapiq SHAP computation is ">120× efficiency gains" faster with the KV cache. That works out to "1.08 seconds" per test row for a 200k-row, 500-feature training table (GPU). — [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)
- TabPFN-3.5 report (14 Sep 2026):
  - Single-row online inference "is bound by the memory bandwidth of moving the cache (and the weights)".
  - Cached-predict times of 3.5 and 3 "align exactly", and TabPFN-3.5-Fast (4 estimators) is "around three times faster".
  - Grouped-query attention keeps the KV-cache size "largely unchanged compared to TabPFN-3".
  - [TabPFN-3.5 Technical Report, arXiv 2609.17895](https://arxiv.org/abs/2609.17895)

**CPU guidance and limits**
- README: "⚡ GPU Recommended ... (even older ones with ~8GB VRAM work well; 16GB needed for some large datasets). On CPU, only moderate datasets are feasible (TabPFN-3, TabPFN-3.5 and TabPFN-3.5-Fast allow up to 5000 samples; older versions up to 1000)." — [TabPFN README](https://github.com/PriorLabs/TabPFN)
- In the source, `cpu_sample_limit()` returns 1000 for `V2, V2_5, V2_6` and 5000 otherwise. The override is `TABPFN_ALLOW_CPU_LARGE_DATASET=true` or `ignore_pretraining_limits=True`. The README adds: "Note: large datasets can still be slow on CPU!" — [tabpfn inference_config.py / README](https://github.com/PriorLabs/TabPFN)
- v2.5 pretraining limits are 50,000 rows and 2,000 features. Above 500 features, 500 are subsampled per estimator. The class limit is 10 and cannot be ignored. — [tabpfn classifier.py docstring](https://github.com/PriorLabs/TabPFN)
- The commercial "Fast Inference Mode" distils TabPFN into an MLP or tree ensemble with "orders-of-magnitude lower latency". The TabPFN-3 report says the distilled artifact "runs on CPU at the sub-millisecond latency of a standard MLP or tree ensemble". — [TabPFN README](https://github.com/PriorLabs/TabPFN); [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)

**Other speed knobs**
- `inference_precision`:
  - `"auto"` (the default) chooses autocast by device. `"autocast"` forces mixed precision. A `torch.dtype` forces that precision.
  - The docstring says it "can dramatically affect the speed and reproducibility", and that higher precision comes "at the cost of a much higher inference time".
  - [tabpfn classifier.py](https://github.com/PriorLabs/TabPFN)
- `memory_saving_mode` (`"auto"` / True / False):
  - It batches internal computations and "can both avoid out-of-memory errors and improve fit+predict speed".
  - A source comment says the right setting "can have a big impact on fit+predict speed, sometimes greater than 2x" (PR #605).
  - The CPU heuristic turns it on when train+test cells exceed 200,000. The code marks this "TODO: Refine the CPU heuristic".
  - [tabpfn memory.py](https://github.com/PriorLabs/TabPFN); [PR #605](https://github.com/PriorLabs/TabPFN/pull/605)
- `n_estimators="auto"` resolves to `DEFAULT_N_ESTIMATORS = 8`. It is raised on wide data so every feature is seen by some estimator. Each estimator is one forward pass ("an ensemble of n_estimators-many 'prompts'"). — [tabpfn source](https://github.com/PriorLabs/TabPFN)
- `n_preprocessing_jobs` threads preprocessing across estimators. The old `n_jobs` "never had any effect". — [tabpfn classifier.py](https://github.com/PriorLabs/TabPFN)
- Christoph Molnar's blog (12 May 2026) shows the same ensemble effect for TabICL on the breast-cancer data:
  - 8 / 4 / 2 / 1 estimators took 0.42 / 0.22 / 0.11 / 0.06 s, which is roughly linear.
  - CPU ~15 s vs GPU ~0.5 s vs GPU with KV cache 0.09 s.
  - Exact hardware was not extracted.
  - [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models)

### Inferences
- The defaults (`fit_preprocessors`, `kv_cache=False` in TabICL) do not cache the context's forward pass. Both libraries make caching opt-in because it costs a lot of memory. Caching is a standard, documented feature in both, but it is not the default in either.
- Building the cache costs about one cold fit+predict (TabPFN-3 report). In a one-fit-one-predict loop such as one predict per CV fold, the expected gain is therefore close to nothing. The cookbook's "single predict" gain looks larger because its uncached default re-runs the forward pass at predict time. Gains come from repeated predicts: large prediction grids predicted in pieces, SHAP or permutation importance, or online scoring.
- The headline "1–3 orders of magnitude" numbers are GPU (H100) numbers at large n_train with small test batches. For moderate n on CPU, the only directly comparable documented number is the Nature paper's CPU caching figure (Q2).

### Gaps
- PriorLabs publishes no CPU timings for `fit_with_cache` in the current package. The cookbook does not state its hardware.
- There is no published per-estimator KV-cache memory formula for v2/v2.5 (only the TabPFN-3 figure of 7 GiB at 1M rows). The "6.1 KB GPU / 48.8 KB CPU per cell" figure that Molnar attributes to the TabPFN-2.5 paper could not be checked: the 2.5 PDF's text layer would not extract.

## Q2. TabPFN v2 Nature paper (Hollmann et al. 2025): what timings and hardware were reported against tuned baselines, and was caching discussed?

### Takeaway
The headline result is "In 2.8 s" TabPFN beats a 4-hour-tuned baseline ensemble, a 5,140x speedup for classification. It was measured with 8 CPU cores plus an RTX 2080 Ti GPU for TabPFN and 8 CPU cores for the baselines. That figure compares fit+predict against tuning. The paper does discuss train-state caching explicitly, with sizeable CPU speedups. It also concedes that per-sample latency is about 1,000x slower than CatBoost.

### Cited Findings
- "In 2.8 s, TabPFN outperforms an ensemble of the strongest baselines tuned for 4 h". The speedup is 5,140x for classification and 3,000x for regression; the regression average is 4.8 s. — [Hollmann et al., Nature 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)
- Hardware: "All methods were evaluated using eight CPU cores, with TabPFN additionally using a consumer-grade GPU (RTX 2080 Ti)." Baselines (XGBoost, CatBoost, LightGBM, RF, etc.) were tuned by random search with 5-fold CV under 30 s to 4 h budgets. Pretraining used 8 RTX 2080 GPUs for 2 weeks. — [Nature 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)
- Caching, verbatim: "On datasets with 10,000 training samples and 10 features, our optimized train-state caching results in inference speedups of around 300× on CPU (from 32 s to 0.1 s) and 6× on GPU. With 10× more features (100), the speedups increase to 800× on CPU and 30× speedup on GPU." This is from the section "An architecture designed for tables". — [Nature 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)
- Latency limitation, verbatim: "For a dataset with 10,000 rows and 10 columns, our model requires 0.2 s (0.6 s without GPU) to perform a prediction for one sample, whereas CatBoost (default) can do the same in 0.0002 s." — [Nature 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)
- Memory, verbatim: "less than 1,000 bytes per cell ... datasets with up to 50 million cells (for example, 5 million rows × 10 features) on a single H100 GPU". — [Nature 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)

### Inferences
- The 300x/800x CPU caching numbers are the best published evidence that caching helps most on CPU. Without a cache, the CPU must redo the O(n_train²) train-side attention on every call. On GPU that work is cheap, so caching gives only 6–30x there.
- The 5,140x figure compares a TFM fit+predict against hours of baseline tuning. It says nothing about TFM vs an untuned RF/GBDT, where TFMs are slower (see Q4).
- The paper measured caching at 10,000 training rows on CPU, but today's package blocks v2 on CPU above 1,000 rows by default. So the paper's CPU caching scenario now needs the override flag.

### Gaps
- The paper text does not give the test-set size behind "32 s to 0.1 s", the CPU model, or whether that caching path shipped in the v2 open-source release. I did not confirm when `fit_with_cache` first appeared in `tabpfn`.

## Q3. Other tabular foundation models (TFMs: TabDPT, Mitra, LimiX, TabFlex, TabPFN-3/3.5, AutoGluon): do they cache the context, use retrieval or kNN subsets, or use linear attention for speed? What speeds do they report?

### Takeaway
There are three families of shortcut:
- **KV caching of the full context.** TabPFN (`fit_with_cache`, from the v2 Nature paper through v3.5) and TabICL (`kv_cache`, with a cheaper `"repr"` mode).
- **Per-query retrieval of a kNN context.** TabDPT, LimiX's attention-guided retrieval, LocalPFN. This bounds context size but defeats cache reuse, because every query gets a different context.
- **Sub-quadratic attention or smaller models.** TabFlex (linear attention), LimiX-2M, TabPFN-3.5-Fast, plus distillation to an MLP or trees.

I found no caching in TabDPT or Mitra. AutoGluon wraps TFMs with row limits and bagging but documents no cache.

### Cited Findings
**TabICL (anchor, tabicl 2.2.0)**
- `kv_cache=False` is the default.
- `True`/`"kv"` caches the K/V projections of the column-embedding ISAB layer (induced set attention block) and of every in-context-learning (ICL) transformer layer: "Fast inference but memory-heavy for large training sets".
- `"repr"` caches column-embedding K/V plus row-interaction outputs and "Uses ~24x less memory than 'kv' for the ICL part, at the cost of re-running the ICL transformer at predict time".
- A saved model can carry the cache and drop the raw training data.
- Other speed knobs: `n_estimators=8`, `batch_size=8`, and `offload_mode` (`'auto'`/`'gpu'`/`'cpu'`/`'disk'`) for the column-embedding tensor.
- [tabicl source/README](https://github.com/soda-inria/tabicl)
- The TabPFN-3 report notes that TabICLv2's CPU/disk offloading "requires a large amount of CPU memory (250GB for a 1M × 500 table ...) or otherwise incurs substantial I/O overhead (... a 4x slowdown)". — [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)

**TabDPT (Layer 6)**
- "For each test query row ... we retrieve the top K closest rows from the training set to get context". Retrieval uses FAISS "in the raw feature space after preprocessing". Context sizes run from 128 to 2,048, with 2,048 as the default plus 8 ensemble members. Hardware: "Nvidia A100 GPUs with 40 GB". — [TabDPT, arXiv 2410.18164](https://arxiv.org/abs/2410.18164)
- Speed claims:
  - The context-2048 model is "at least one order of magnitude faster than dataset-specific tree and deep learning baselines". That comparison includes the baselines' tuning.
  - Its runtime is "comparable to TabPFN v2".
  - Without training cost, "classical tree-based pipelines can deliver substantially faster inference (e.g., ∼0.02–0.10s per 1,000 rows), often about an order of magnitude faster than TabDPT for inference alone".
  - [arXiv 2410.18164](https://arxiv.org/abs/2410.18164)
- The paper does not mention KV caching. — [arXiv 2410.18164](https://arxiv.org/abs/2410.18164)
- Liu & Ye ("TabPFN Unleashed"): kNN-per-query support sets (LocalPFN, TabDPT) "disrupt PFN-style batching, significantly reducing inference efficiency". — [arXiv 2502.02527](https://arxiv.org/abs/2502.02527)

**Mitra (Amazon / AutoGluon)**
- Pretrained on 8x A100 40 GB for about 60 h, on 45M synthetic datasets. In Mitra's own TabArena table (Table 17), Mitra (single) has a median train time of 457.2 s and inference of 34.6 s per 1K rows. That compares with TabICL 8.9 / 1.7, TabPFNv2 4.1 / 0.4 and RandomForest 0.4 / 0.1. Mitra's default includes in-context ensembling and fine-tuning, which I infer explains the large train time. No caching is described. — [Mitra, arXiv 2510.21204](https://arxiv.org/abs/2510.21204)
- AutoGluon 1.6.1 docs:
  - Mitra "runs seamlessly on both GPU and CPU".
  - Per-fit limits: Mitra and TabPFNv2 ≤10,000 rows / 500 features / 10 classes; TabICLv2 ≤500,000 rows / 2,000 features; TabDPT ≤100,000 rows / 2,500 features / 160 classes.
  - "several models are far slower near their upper bound than well below it".
  - No context-cache option is documented.
  - [AutoGluon TFM tutorial](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html)

**LimiX / LimiX-2M**
- LimiX: "attention-guided retrieval provides an efficient, optional ensemble and fine-tuning mechanism ... retrieve informative samples and features using their own attention scores". — [LimiX, arXiv 2509.03505](https://arxiv.org/abs/2509.03505)
- LimiX-2M was benchmarked on synthetic data (900 samples × 60 features, mean of 3 runs) on an **AMD EPYC 9354 CPU + RTX 4090 GPU**. It ran in "171.40 ms on GPU, outperforming TabPFN-v2 by ≈2× and TabICL by > 10×", and "maintains a substantial lead in CPU environments against heavy models like Mitra". — [LimiX-2M, arXiv 2606.04485](https://arxiv.org/abs/2606.04485)

**TabFlex (ICML 2025)**
- It replaces quadratic attention with linear attention, after finding that state-space models' causality "hinders ICL performance". It "processes the poker-hand dataset with over a million samples in just 5 seconds" and claims ">2x speedup compared to TabPFN and a 1.5x speedup over XGBoost". The abstract states no hardware. — [TabFlex, arXiv 2506.05584](https://arxiv.org/abs/2506.05584)

**TabPFN-3 / 3.5 (other shortcuts besides the cache)**
- TabPFN-3 is "up to 20x faster than TabPFN-2.5". Row-chunking is enabled when ntrain+ntest > 2048 and keeps activations on the GPU instead of offloading them. — [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)
- TabPFN-3.5-Fast is "up to 3× faster than TabPFN-3". — [arXiv 2609.17895](https://arxiv.org/abs/2609.17895)

### Inferences
- Full-context KV caching is now standard in the two leading open TFMs, TabPFN and TabICL, as an opt-in. Retrieval-based TFMs (TabDPT, LimiX retrieval mode) cannot reuse one cache across queries because each query has its own context. Their speed comes from capping context size instead.
- Distillation to an MLP or trees is the vendor-endorsed route to CPU-speed inference (sub-ms per row). This suggests PriorLabs itself does not treat KV caching as enough to match tree latency.

### Gaps
- I found no caching implementation or measurement for TabDPT, Mitra, LimiX or TabFlex. The absence in their papers was checked by text search, not by reading the code.
- I did not verify the TabFlex hardware, or whether TabDPT's released code shares retrieval across batches.

## Q4. What do independent benchmarks (TabArena, TALENT) report for train and inference times of TabICL and TabPFN vs Random Forest and GBDTs, especially on CPU?

### Takeaway
TabArena v0.1 ran trees on 8-core CPUs and TFMs on an L40S GPU. With defaults, the TFMs train in seconds per 1K rows, but they predict 6–30x slower than a default Random Forest. TabICL predicts at 1.48 s per 1K samples vs RF's 0.05 s, both under TabArena's 8-fold bagging. TFMs win clearly only against tuned-and-ensembled GBDTs, which take hundreds to thousands of seconds per 1K rows to train.

### Cited Findings
- TabArena v0.1 (NeurIPS 2025 Datasets & Benchmarks track, arXiv v4) setup:
  - CPU models ran on "AWS m6i.2xlarge (eight cores Intel Xeon CPU)".
  - "We run TabM, ModernNCA, and the foundation models on GPU", which was an "NVIDIA L40S with 48 GB VRAM and eight cores of an AMD EPYC 9334 CPU".
  - Each configuration is validated with 8-fold inner CV.
  - TabPFNv2 was limited to ≤10k training rows / 500 features; TabICL to classification ≤100k rows / 500 features.
  - [TabArena, arXiv 2506.16791](https://arxiv.org/abs/2506.16791)
- TabArena Table A.1, median times per 1K samples (train / predict, seconds). D = default, T = tuned, T+E = tuned + ensembled.

  | Model | Hardware | Train / 1K (s) | Predict / 1K (s) |
  |---|---|---|---|
  | RandomForest (D) | CPU | 0.43 | 0.05 |
  | ExtraTrees (D) | CPU | 0.25 | 0.05 |
  | XGBoost (D) | CPU | 1.94 | 0.12 |
  | LightGBM (D) | CPU | 1.96 | 0.14 |
  | CatBoost (D) | CPU | 6.83 | 0.08 |
  | **TabPFNv2 (D)** | GPU | 3.36 | 0.31 |
  | **TabICL (D)** | GPU | 6.63 | 1.48 |
  | TabDPT (D) | GPU | 22.53 | 8.55 |
  | RandomForest (T+E) | CPU | 373.18 | 0.77 |
  | CatBoost (T+E) | CPU | 1658.41 | 0.65 |
  | LightGBM (T+E) | CPU | 416.98 | 2.64 |
  | TabPFNv2 (T+E) | GPU | 3030.15 | 21.44 |
  | RealMLP (T+E) | GPU | 6564.71 | 10.26 |

  — [TabArena, arXiv 2506.16791](https://arxiv.org/abs/2506.16791)
- TabArena paper: "tuned EBM (EBM-T) and tuned CatBoost (CatBoost-T) shine at inference time". — [arXiv 2506.16791](https://arxiv.org/abs/2506.16791)
- PriorLabs' restatement of the current TabArena timing protocol: "every method is fitted as an 8-fold bagging ensemble with a final refit ... on one node with 24 CPU cores and one NVIDIA RTX Pro 6000 GPU". It adds that "8-fold bagging is not used for TFMs in general". — [TabPFN-3.5 report, arXiv 2609.17895](https://arxiv.org/abs/2609.17895)
- Mitra's TabArena table (single/default models, median per 1K rows, train / infer): TabICL 8.9 / 1.7; TabPFNv2 4.1 / 0.4; RandomForest 0.4 / 0.1; CatBoost 5.7 / 0.1; XGBoost 2.4 / 0.2. — [arXiv 2510.21204](https://arxiv.org/abs/2510.21204)
- Molnar: switching TabICL from CPU to GPU gave about a 30x speedup on breast cancer (~15 s → ~0.5 s). — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models)

### Inferences
- TFM predict times on TabArena are inflated by bagging. Each of the 8 bagged models re-runs the context forward pass at predict time, and TabArena does not appear to use KV caching. Even after dividing by 8, TabICL's GPU predict (~0.19 s per 1K) stays above RF's CPU predict (0.05 s per 1K, which also includes bagging).
- No major benchmark reports TFM times on CPU. TabArena, the Mitra table and the TabDPT paper all run TFMs on GPU. Any CPU comparison (like ours) has to rely on the Nature paper's CPU numbers, Molnar's ~30x CPU penalty, or our own measurements.

### Gaps
- I did not get to the TALENT benchmark's timing tables. I found no independent CPU-only TFM vs RF benchmark.
- It is unconfirmed whether TabArena's TFM wrappers use `fit_with_cache` or `kv_cache`; I believe they do not, but did not check the code.

## Q5 (synthesis). How standard is "cache the training set", and what speedups are realistic, compared with TabICL?

### Takeaway
Caching the context is now a documented, opt-in feature in TabPFN (`fit_with_cache`, described since the 2025 Nature paper) and TabICL (`kv_cache`). Retrieval TFMs do not use it. The realistic gain is large only when the same fit is predicted many times. In that case the documented gains are 5–30x on GPU at moderate n, 300–800x on CPU at 10k rows (Nature, v2), and up to 1–3 orders of magnitude on an H100 at large n (v3). For a single fit followed by a single predict, the gain is about zero.

### Cited Findings
- Building the cache costs "essentially the same as the cold fit+predict". — [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)
- Nature v2: 300x on CPU (32 s → 0.1 s) and 6x on GPU at 10k × 10; 800x on CPU and 30x on GPU at 10k × 100. — [Nature 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)
- Cookbook: 5.5x (one predict), 2.7x total (10 predicts), 5k rows, hardware unstated. — [PriorLabs cookbook](https://docs.priorlabs.ai/cookbook/faster_performance_with_cache)
- TabICL on GPU: 0.5 s → 0.09 s with KV cache (~5.5x), breast cancer. — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models)
- Even with a cache, per-row latency stays above trees. Nature: 0.2 s per sample on GPU vs CatBoost 0.0002 s. TabPFN-3 cached: 0.1–3 ms per test point on H100. — [Nature 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/); [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)

### Inferences
- For a spatial-block CV in which each fold refits on a new training set and predicts once, a KV cache gives no meaningful saving. The levers that do help are:
  - fewer estimators (cost is roughly linear in `n_estimators`);
  - a GPU (about 30x);
  - a smaller context (subsampling or kNN);
  - batching the test rows.
- The cache does pay off for a fixed final model used to predict a large map or grid in chunks, or under SHAP, permutation importance or partial-dependence loops. On CPU, the Nature evidence suggests the gain can be very large (two or more orders of magnitude). That is because without a cache the CPU repeats the quadratic train-side attention on every call.

### Gaps
- No public source gives TabICL vs TabPFN cached-predict timings measured side by side on the same CPU hardware. This would need a local measurement.
