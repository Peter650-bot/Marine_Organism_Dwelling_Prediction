# Cost structure of tabular in-context learning (TabICL / TabICLv2 / TabPFN) and speed-up strategies beyond caching

Scope note: research current to 2026-10-03. Peer-reviewed = arXiv papers published at NeurIPS/ICML/ICLR or Nature. Blog/docs/code = labelled as such. "Code" findings come from reading the installed `tabicl` 2.1.1 package source (`/home/peets/.local/lib/python3.10/site-packages/tabicl/`), which is the primary source for what the KV cache actually stores.

## 1. Cost structure: which stages scale with n_train, n_test, n_features, and what is cacheable?

### Takeaway
In TabICL (v1 and v2), test rows never influence training-row representations: the column-wise ISAB inducing points attend only to training rows, row-wise interaction is per-row, and in the 12-layer ICL transformer train rows attend only to train rows while test rows attend only to train rows. So the whole training-side computation is exactly cacheable. Without a cache, every predict call redoes train-to-train attention, which costs O(n_train^2) per layer per ensemble member. With a cache, prediction costs O(n_test * n_train) per layer for test-to-train attention, plus cheap per-row work that is linear in n_test. TabPFN v2/2.5 has the same train/test independence but costs more, because it never compresses away the feature axis.

### Cited Findings
**TabICL v1 architecture and complexity (peer-reviewed, ICML 2025)**
- The column-wise embedding (TF_col) is a Set Transformer with "k=128 inducing vectors, 4 attention heads in the MABs, and 3 ISAB blocks", d=128. The row-wise interaction (TF_row) is a "3-layer transformer with 8 attention heads", with "four learnable [CLS] tokens" per row whose outputs are concatenated to 4x128 = 512 dims. The dataset-wise ICL stage (TF_icl) is a "12-layer Transformer with 4 attention heads" — [TabICL, arXiv 2502.05564](https://arxiv.org/html/2502.05564)
- On train/test separation in ISAB: "In MAB1, inducing vectors act as queries and attend to training samples U_train ... In MAB2, inputs U (including both training and test samples) serve as queries and attend back to M ... Crucially, only train samples serve as keys and values in MAB1. This ensures that the outputs of ISAB depend solely on training data, thereby preventing data leakage." — [TabICL](https://arxiv.org/html/2502.05564)
- ICL mask: "The embeddings in H_train can attend to one another while those in H_test can only attend to H_train." — [TabICL](https://arxiv.org/html/2502.05564)
- Stated complexity: TF_col O(n k m) with k=128, TF_row O(m^2 n), and TF_icl O(n^2), for an overall "O(m^2 n + n^2)". TabPFNv2 "alternates column-wise and row-wise attentions without collapsing dimensions, resulting in complexity O(m^2 n + n^2 m)" — [TabICL](https://arxiv.org/html/2502.05564)
- Measured speed: "TabICL is 1.5x faster than TabPFNv2 on small datasets and 3-10x faster on large datasets". For example, 10,000 samples x 100 features takes "roughly 20 seconds versus 1 minute 40 seconds for TabPFNv2" (GPU). It handles up to 500K samples x 500 features. 100K x 500 needs "only 5 GB of GPU memory and 32 GB of RAM" with CPU/disk offloading of activations — [TabICL](https://arxiv.org/html/2502.05564)

**TabICLv2 (arXiv preprint, Feb 2026)**
- v2 keeps the "compression then ICL" three-stage design at O(n^2 + n m^2), compared with TabPFN's O(n^2 m + n m^2). It adds query-aware scalable softmax (QASSMax) in TF_col and TF_icl, plus "early target injection": target embeddings are added to training tokens before column-wise compression (E2[i,j] = E1[i,j] + Embed(y_i) for i in D_train only) — [TabICLv2, arXiv 2602.11139](https://arxiv.org/html/2602.11139)
- Speed: "For 100 features, TabICLv2 is faster than TabPFN-2.5 across all hardware ... 10.6x on an H100 GPU at 50K samples. The efficiency gap is even more pronounced on CPU, reaching 11.8x at just 10K samples." 1M samples x 500 features runs "under 24 GB CPU and 50 GB GPU ... within 450 seconds" with disk offloading — [TabICLv2](https://arxiv.org/html/2602.11139)
- The TabICLv2 paper itself does not discuss KV caching — [TabICLv2](https://arxiv.org/html/2602.11139)

**What the TabICL KV cache stores (code, tabicl 2.1.1)**
- `kv_cache.py` module docstring: "1. ColEmbedding: Cache K/V of the second attention layer of ISAB blocks; 2. ICLearning: Cache K/V from training data at each layer of the ICL transformer" — [tabicl source / GitHub](https://github.com/soda-inria/tabicl)
- `TabICLClassifier(kv_cache=...)` docstring: "True or 'kv': Cache key-value projections from both column embedding and ICL transformer layers. Fast inference but memory-heavy for large training sets." and "'repr': Cache column embedding KV projections and row interaction outputs (representations). Uses ~24x less memory than 'kv' for the ICL part, at the cost of re-running the ICL transformer at predict time." The cache is built during `fit()` — [tabicl source / GitHub](https://github.com/soda-inria/tabicl)
- In `forward_with_cache`, the cached path sets `X = X_test; y_train = None` and runs only test rows through col and row stages. In `kv` mode, test rows then attend to the cached per-layer train K/V. In `repr` mode, the cached train representations (with y_train "baked in") are concatenated with the test representations and the full ICL transformer is re-run — [tabicl source / GitHub](https://github.com/soda-inria/tabicl)
- The pretraining code path has an `embed_with_test` flag: "If True, inducing points attend to all samples (train + test). If False, inducing points only attend to training samples." It defaults to False, so inference uses the train-only ISAB — [tabicl source / GitHub](https://github.com/soda-inria/tabicl)
- README parameters: `n_estimators=8` ("number of ensemble members, more = better but slower"), `batch_size=8` ("ensemble members processed together"), `use_amp="auto"`, `use_fa3="auto"`, `offload_mode="auto"`, and `n_jobs` ("number of PyTorch threads for CPU inference") — [soda-inria/tabicl README](https://github.com/soda-inria/tabicl)

**TabPFN v2 (peer-reviewed, Nature 2025)**
- "Each layer first has an attention over features, followed by an attention over samples ... our inter-sample attention does not allow the test samples to attend to each other but only to the training data." — [Hollmann et al., Nature 637 (2025), PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)
- The Nature paper states compute "scale[s] quadratically with the number of samples (n) and the number of features (m), that is O(n^2 + m^2)" and memory scales as O(n*m). The TabICL authors give the per-layer total as O(m^2 n + n^2 m). These are consistent once you count that sample-attention runs once per feature column and feature-attention once per row; the Nature figure is per attention operation — [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/); [TabICL](https://arxiv.org/html/2502.05564)
- TabPFN-2.5 (tech report, Nov 2025) uses "18 layers for regression ... 24 layers for classification", alternating attention, feature group size 3, and "64 additional 'thinking' rows". It is "between 1x and 2.3x faster" than v2 despite being larger, and uses FlashAttention-3 — [TabPFN-2.5, arXiv 2511.08667](https://arxiv.org/html/2511.08667)

### Inferences
- Per ICL layer with hidden width d (512 in v1): **train-to-train** attention is O(n_train^2 * d) and is paid once per ensemble member per forward pass. **Test-to-train** attention is O(n_test * n_train * d). Everything else is O(n) in rows. TF_col costs O(n*k*m*d) and is linear in n because of the 128 inducing points. TF_row costs O(n*m^2*d) and is linear in n but quadratic in features. With n_estimators=8, all of this is multiplied by about 8. When n_test is much smaller than n_train (e.g. one CV fold of 30k train / 3k test), train-to-train dominates by roughly n_train/n_test = 10x.
- What depends only on training data (cacheable): the ISAB inducing-point summaries M (MAB1 outputs), the train rows' col/row representations, and every ICL layer's train K/V. What depends on test rows: their own col embedding (MAB2 queries against cached M), their row embedding, and their attention over the cached train K/V. No test row affects any training-row representation or any other test row, so (a) caching is exact, not an approximation, and (b) predicting test rows in arbitrary chunks gives the same output as one big batch, up to floating-point noise.
- Cache memory (my own arithmetic, not stated by the authors): the "~24x" in the `repr` docstring matches 12 layers x (K + V) = 24 tensors per train row versus 1 representation per row. If the per-row K/V width is 512, the `kv` cache holds about 12 x 2 x 512 x 4 B = ~49 KB per train row per ensemble member in fp32, or about 390 KB/row for 8 members. That is about 12 GB for 30k train rows. The `repr` cache is about 2 KB/row/member. Measure this before relying on it.
- `repr` mode avoids only the col/row stages for train rows. It still pays the O(n_train^2) ICL train-to-train cost on every predict call, so it helps mainly when m (features) is large. With few features (e.g. about 10 env covariates), `repr` saves little.

### Gaps
- I found no published per-stage FLOP/time breakdown for TabICL (e.g. % time in TF_col vs TF_row vs TF_icl on CPU) and no official cache-memory-per-cell figure for TabICL (TabPFN has one; see section 2).
- I did not verify from the TabICL code whether the per-estimator preprocessing (normalisation, outlier clipping, feature shuffles) is fitted on training rows only. That is assumed by sklearn convention and by the train-only ISAB design.

## 2. Why "caching moves cost into fit": when a KV cache helps and when it doesn't

### Takeaway
A KV cache does not remove the O(n_train^2) train-to-train computation. It runs it once, in `fit()`, and stores the result. It pays off only when the same context serves many `predict` calls or streaming test batches; the published speedups (TabPFN: ~300x on CPU) are predict-only. For a single fit-then-predict (as in a CV fold), total wall time is about the same or slightly worse, and memory rises sharply.

### Cited Findings
- TabPFN v2 (Nature): "On datasets with 10,000 training samples and 10 features, our optimized train-state caching results in inference speedups of around 300x on CPU (from 32 s to 0.1 s) and 6x on GPU. With 10x more features (100), the speedups increase to 800x on CPU and 30x speedup on GPU." — [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/) (peer-reviewed; these are prediction-time numbers after the train state is built)
- TabPFN `fit_mode` docstrings (code): `low_memory` preprocesses "on-demand during inference"; `fit_preprocessors` caches preprocessing at fit; `fit_with_cache` additionally initialises "the transformer key-value cache"; `kv_cache_precision` options include `"int8"`/`"fp8"` for memory — [PriorLabs/TabPFN classifier.py](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/classifier.py)
- TabPFN-2.5 report on the KV cache: "An experimental feature that trains and stores the model during .fit(), making subsequent .predict() calls fast by using a KV-Cache." Memory: "Classification models will consume approximately 6.1 KB of GPU memory and 48.8 KB of CPU memory per cell in the training dataset (regression models about 25% less)." Recommended only "for small training datasets"; distilled MLP/TreeEns is suggested for larger ones — [TabPFN-2.5, arXiv 2511.08667](https://arxiv.org/html/2511.08667)
- Blog (Molnar, 12 May 2026): KV caching gave about 5.6x faster predict (0.5 s to 0.09 s, breast-cancer data, GPU). The trade-off is a "slower .fit()" and it is "only beneficial with repeated .predict() calls". GPU plus KV cache compounds to ">100x" faster prediction versus CPU without cache — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models) (blog, not peer-reviewed)
- TabICL docstring caveat: the AMP/FA3 "auto" heuristic "assumes that the training set is large relative to the test set and does not account for KV-cache scenarios" — [tabicl source / GitHub](https://github.com/soda-inria/tabicl)
- TabICL README describes the cache as "cache training data KV projections for faster repeated inference" and notes a privacy side-benefit: a fitted model can be saved with `save_training_data=False` if the KV cache is kept — [soda-inria/tabicl](https://github.com/soda-inria/tabicl)

### Inferences
- Without a cache, each `predict` call recomputes the train side: about 8 x 12 x O(n_train^2) attention for TabICL. With `kv`, `fit` pays that once, and each `predict` pays about 8 x 12 x O(n_test * n_train) plus small linear terms.
- The cache wins when:
  - the same fitted model is queried repeatedly (e.g. predicting a large spatial grid, many map tiles, or time slices with one fitted context per species);
  - test data arrives in streams or batches;
  - the model is served.
- The cache does not win for one fit plus one predict on a fixed test fold. The total work is identical, because cache build plus a cached predict equals one uncached forward. Allocating and storing the cache adds overhead, and memory can explode: about 12 GB at 30k train rows x 8 estimators by the arithmetic in section 1. On CPU, memory bandwidth may then become the bottleneck.
- For spatial-block CV, each fold has a different context, so nothing carries over between folds. Caching helps only within a fold when predicting several test sets (e.g. the CV test fold plus a prediction map) from the same fitted context.

### Gaps
- No peer-reviewed amortization analysis (break-even number of predict calls) for TFMs was found; the break-even is implied by the cost model rather than measured in literature.
- No published TabICL-specific predict-time speedup figure for `kv` vs no-cache on CPU was found.

## 3. Context reduction: subsampling, retrieval/kNN contexts, learned contexts, linear attention

### Takeaway
Shrinking the context cuts the quadratic term: halving n_train cuts ICL attention cost about 4x. Random subsampling is the cheapest option but loses accuracy as data grow; TabPFN with random subsampling "plateaus and declines". kNN/retrieval contexts (LoCalPFN, TabDPT) and learned compressed contexts (TuneTables, in-context data distillation, MixturePFN) recover or exceed full-context accuracy, at the cost of per-query retrieval or per-dataset tuning. I found no published curve giving the exact accuracy loss at 1k-5k versus 30k rows for TabICL/TabICLv2; this has to be measured.

### Cited Findings
**Retrieval / local context**
- LoCalPFN (Thomas et al., NeurIPS 2024): each test point gets its own kNN context with "the minimum of (i) 10 times the square root of the dataset size, and (ii) 1000" neighbours, using distances on standardised features. In their Fig. 2 (adult-census, electricity, eeg-eye-state), TabPFN with random subsampling "plateaus and declines as dataset size increases", while "TabPFN-kNN can leverage larger datasets ... and shows improvements". AUC IQM on 95 TabZilla datasets: TabPFN-kNN 0.943 and LoCalPFN (kNN plus fine-tuning) 0.958; on 48 medium/large datasets, 0.955 and 0.968. TabPFN-kNN "runs surprisingly fast" — [LoCalPFN, arXiv 2406.05207](https://arxiv.org/html/2406.05207)
- TabDPT: uses faiss kNN retrieval with a 2048-row context at inference. "Removing training retrieval leads to a consistent drop in both classification and regression test accuracy." A 256-row context performs worse than 2048 but better than no retrieval. Fig. 3(b) shows the accuracy/latency trade-off as context grows. Runtime is "comparable to TabPFN v2" on A100-40GB — [TabDPT, arXiv 2410.18164](https://arxiv.org/html/2410.18164)

**Learned or compressed context**
- TuneTables (Feuer et al., NeurIPS 2024): prompt-tunes a learned context of 10 tokens (≤2000 samples) or 1000 tokens (>2000 samples). "TabPFN_s3000" (random 3,000-row context) "is best on small datasets, but when scaled past 3000 datapoints and 100 features ... significantly underperforms". TuneTables is "around 9x faster than TabPFNs3000, on average" at inference. Mean accuracy 0.860 vs CatBoost 0.856 on 98 TabZilla datasets, and 0.830 vs 0.823 on LargeScaleTables (up to 1.9M rows) — [TuneTables, arXiv 2402.11137](https://arxiv.org/html/2402.11137v3)
- In-context data distillation (Ma, Thomas, Yu, Caterini, 2024): optimises a small synthetic context for TabPFN, trading "quadratic memory complexity" for "a linear number of tuning steps". Evaluated on 48 large OpenML datasets and reported competitive with XGBoost. The abstract gives no numeric deltas — [arXiv 2402.06971](https://arxiv.org/abs/2402.06971) (preprint)
- MixturePFN (ICLR 2025): routes each test sample to one of K "in-context prompters", each fine-tuned on a kNN-defined cluster of training data. It is motivated by transformers' "quadratic time complexity with respect to context size" and by context distribution shift. It beat 19 baselines on 36 datasets (mean rank / Condorcet winner) — [MixturePFN, arXiv 2405.16156](https://arxiv.org/pdf/2405.16156); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3de60dc2ff7ac8866e805bbf457e2291-Abstract-Conference.html)
- "A Closer Look at TabPFN v2" (Ye, Liu, Chao, NeurIPS 2025): proposes test-time divide-and-conquer for large data. It splits training data into support and query sets and uses TabPFN v2 features extracted on the support set to train a linear model — [arXiv 2502.17361](https://arxiv.org/abs/2502.17361v1)

**Linear attention**
- TabFlex (Zeng et al., ICML 2025): replaces softmax with non-causal linear attention (O(n)); test rows attend only to train rows. Median 0.29 s vs 0.82 s per 1000 instances against TabPFN on 57 small datasets ("nearly identical performance"). Poker-hand (1.025M rows) took 4.88 s vs TabPFN 15.36 s vs XGBoost 504.52 s. Median AUC on hard large datasets was 0.76 vs TabPFN 0.72. GPU only, with no CPU timings — [TabFlex, arXiv 2506.05584](https://arxiv.org/html/2506.05584); [PMLR v267](https://proceedings.mlr.press/v267/zeng25b.html)

**Library-level subsampling and ensemble size**
- TabPFN docs: "For large datasets or limited memory, use per-estimator subsampling, e.g. `inference_config={'SUBSAMPLE_SAMPLES': 50_000}`" — [PriorLabs/TabPFN classifier.py](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/classifier.py)
- Blog (Molnar 2026), TabICL on breast-cancer data: 8 to 4 estimators took 0.42 s to 0.22 s; 8 to 1 took 0.42 s to 0.06 s (about 7x). Log-loss went 0.0671 to 0.0717. Halving the context is about a 4x saving on the quadratic term — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models) (blog; tiny dataset)
- The TabICLv2 paper reports top rankings "across all dataset sizes from 10^3 to 10^5" but gives "no explicit subsampling ablations" — [TabICLv2](https://arxiv.org/html/2602.11139)

### Inferences
- Rough cost model for TabICL with context n_c and n_test test rows: ICL cost scales with n_c^2 + n_test * n_c. Going from 30k to 5k context cuts train-to-train ICL work about 36x. Going to 1k cuts it about 900x. Linear stages shrink 6x and 30x respectively.
- Expected accuracy cost, from the literature pattern only:
  - small for "smooth" problems;
  - larger for problems needing fine local structure. LoCalPFN shows that underfitting with global contexts grows with n, and that kNN contexts fix it.
- For the spatial project, a 5k random subsample still covers geography broadly. But if retention adds little beyond geography (as in the earlier ablation), local spatial context matters most. A kNN context (in feature space or in lat/lon) is likely a better use of a 1-5k budget than random rows.
- Caveat for kNN contexts: each test point (or test cluster) needs its own forward pass, so you cannot share one KV cache. Cost becomes about n_test x O(k^2) unless test points are clustered and share a context, as in LoCalPFN's shared-context approximation or MixturePFN routing.
- Ensemble size (n_estimators) is a linear cost lever that is independent of context, and per the blog has small accuracy cost. Try it first on CPU.

### Gaps
- No published accuracy-vs-context-size curve exists specifically for TabICL/TabICLv2 at 1k/2k/5k/10k/30k. LoCalPFN Fig. 2 and TabDPT Fig. 3(b) are qualitative for this purpose (exact numbers not extracted), so the 30k to 1-5k accuracy delta must be measured on our own data.
- In-context data distillation and MixturePFN numeric accuracy/speed deltas were not extracted (abstract-level only).

## 4. CPU-specific inference practices for PyTorch transformers (and what's reported for TFMs)

### Takeaway
TFMs are reported to be 12-63x (docs) or about 30x (blog) slower on CPU than on GPU. TabICLv2's compression design makes it 11.8x faster than TabPFN-2.5 on CPU at 10k rows. Generic PyTorch CPU levers are thread count/affinity, NUMA binding, allocator, bf16 autocast, torch.compile, ONNX, and quantization. I found no published TFM-specific speedups for bf16-on-CPU, torch.compile, ONNX or int8 weight quantization. Two code-level findings matter: TabICL's `use_amp` only autocasts on CUDA, and its `n_jobs>0` logic actually sets threads to at least all logical cores.

### Cited Findings
- PriorLabs-related docs/aggregators: CPU in-context inference is "12-63x slower on CPU compared to GPU". CPUs are "only suitable for small datasets", and TabPFN-as-MLP/Tree is recommended for CPU deployment — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models) and the PriorLabs docs surfaced by search ([docs.priorlabs.ai](https://docs.priorlabs.ai/best-practices), which returned 404 on fetch, so the 12-63x figure is not independently verified)
- Blog: TabICL on breast cancer took 15 s on CPU vs 0.5 s on GPU (about 30x) — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models)
- TabICLv2 is 11.8x faster than TabPFN-2.5 on CPU at 10K samples with 100 features (peer-review status: arXiv preprint) — [TabICLv2](https://arxiv.org/html/2602.11139)
- TabPFN Nature: train-state caching gives about 300x on CPU at 10k x 10, and about 800x at 10k x 100. Caching matters far more on CPU than on GPU (6x/30x) — [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)
- PyTorch CPU tuning guide:
  - set `OMP_NUM_THREADS`, with `OMP_PROC_BIND=CLOSE`/`GOMP_CPU_AFFINITY`;
  - use `numactl --cpunodebind=N --membind=N` on multi-socket machines;
  - Intel OpenMP (`libiomp5`) with `KMP_AFFINITY=granularity=fine,compact,1,0`, `KMP_BLOCKTIME=1`;
  - preload jemalloc/tcmalloc.
  
  The guide gives no numeric CPU speedups — [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- **TabICL code, AMP on CPU:** inference wraps in `torch.autocast(device_type="cuda")` only `if self.use_amp and self.exe_device.type == "cuda"`. Cache tensors are upcast on CPU because "CPU and MPS do not support float16 attention". So `use_amp` does nothing on CPU, and there is no built-in bf16 CPU path — [tabicl source / GitHub](https://github.com/soda-inria/tabicl)
- **TabICL code, n_jobs:** the docstring says None means "the PyTorch default, which is the number of physical CPU cores", and negative values mean max(1, logical_cores + 1 + n_jobs). But for positive `n_jobs` the code sets `n_threads = max(n_logical_cores, self.n_jobs)`. So `n_jobs=4` on a 16-thread machine uses 16 threads, not 4. That looks like a min/max bug, and it can oversubscribe hyperthreads or collide with outer joblib parallelism — [tabicl source / GitHub](https://github.com/soda-inria/tabicl) (`_sklearn/classifier.py` ~L703-716, `_sklearn/regressor.py` ~L682-697)
- TabICL auto heuristic: AMP is off for "Small (n < 1024 & feat < 60)" and on above that, and FA3 is on for n ≥ 10240, but only on CUDA/Hopper — [tabicl source / GitHub](https://github.com/soda-inria/tabicl)
- TabPFN exposes `inference_precision` (a torch dtype, or "autocast"/"auto") and `memory_saving_mode` ("can both avoid out-of-memory errors and improve fit+predict speed by reducing memory pressure ... by automatically batching certain model computations") — [PriorLabs/TabPFN classifier.py](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/classifier.py)

### Inferences
- Practical CPU recipe for TabICL:
  1. Call `torch.set_num_threads(<physical cores>)` yourself after constructing the model, or leave `n_jobs=None`. Don't pass a positive `n_jobs`, because of the max() bug.
  2. Avoid nesting TabICL inside a joblib/n_jobs CV loop, which oversubscribes threads.
  3. Reduce `n_estimators` (8 to 2-4).
  4. Cap the context with subsampling or kNN.
  5. Predict test rows in large chunks; this is exact, per section 1.
  6. Run under `torch.inference_mode()`.
- A manual `torch.autocast("cpu", dtype=torch.bfloat16)` wrapper might help on CPUs with AMX/AVX-512-BF16 (recent Xeon / Zen 4+), but it is untested for TabICL. Probability calibration should be checked, because QASSMax and softmax over long contexts may be precision-sensitive.
- torch.compile, ONNX export and int8 dynamic quantization are generic options. The TFM forward has data-dependent shapes (n, m vary per dataset), which reduces compile/ONNX benefit. Expect less than for fixed-shape NLP models, but this is a guess.

### Gaps
- I found no peer-reviewed or issue-tracker report quantifying bf16/AMP-on-CPU, torch.compile, ONNX, or int8 quantization speedups for TabPFN or TabICL specifically.
- The source of the "12-63x CPU vs GPU" figure could not be fetched (PriorLabs docs returned 404); I could not determine its hardware or dataset sizes.

## 5. Distillation into a fast model (MLP / trees / hypernetwork-generated nets)

### Takeaway
Distilling a TFM into a dataset-specific MLP or tree ensemble gives RF/GBDT-class predict latency, and PriorLabs reports that the distilled models still beat tuned GBDTs on ≤10k-row benchmarks. You give up the ability to update context without retraining. MotherNet-style hypernetworks generate an MLP in one forward pass and lose noticeable accuracy versus TabPFN.

### Cited Findings
- TabPFN-2.5 distillation engine (tech report, Nov 2025): outputs "TabPFN-2.5-as-MLP" or "TabPFN-2.5-as-TreeEns", trained to mimic TabPFN-2.5 predictions on the given dataset. These "still outperform tree-based models while having much faster inference speed than TabPFN". Fig. 7 compares them against tuned LightGBM/XGBoost/CatBoost on internal ≤10k-sample benchmarks. Latency is described as "orders-of-magnitude lower", but no exact numbers were extracted — [TabPFN-2.5, arXiv 2511.08667](https://arxiv.org/html/2511.08667) (tech report, not peer-reviewed; the distillation engine is a PriorLabs commercial/API feature per the report)
- MotherNet (Mueller et al., arXiv 2312.08598): a hypernetwork that emits a trained MLP in "a single forward pass". On 30 small OpenML CC-18 datasets:
  - **Prediction time:** MotherNet (GPU) 0.006 s, TabPFN (GPU) 0.337 s, XGBoost 0.024 s, MLP 0.001 s.
  - **Fit time:** MotherNet 0.136 s, TabPFN 0.008 s, tuned XGBoost 4.912 s.
  - **Normalised ROC AUC (rank):** TabPFN 0.850 (3.73), **MLP-Distill (TabPFN distilled into an MLP) 0.802 (4.77)**, XGBoost 0.804 (5.93), MotherNet 0.780 (5.57).
  
  Evaluated on datasets ≤2000 samples — [MotherNet, arXiv 2312.08598](https://arxiv.org/html/2312.08598)
- Blog view: surrogate distillation "loses TFM flexibility" and fits "constrained devices or fixed-context scenarios" — [Mindful Modeler](https://mindfulmodeler.substack.com/p/making-tabular-foundation-models)

### Inferences
- In MotherNet's benchmark, distilling TabPFN into an MLP cost about 0.05 normalised-AUC versus TabPFN itself (0.850 to 0.802) and landed at XGBoost level. PriorLabs claims a smaller gap for TabPFN-2.5 distillation. Treat the vendor claim cautiously until independently replicated.
- A DIY version for TabICL/TabICLv2 should work:
  1. Fit TabICL on the training fold.
  2. Score a large synthetic or augmented sample (e.g. jittered training rows, or the full prediction grid).
  3. Fit an RF/LightGBM/MLP to the soft probabilities.
  
  The TFM cost is paid once on the scored set, and prediction runs at RF speed. If the model in our own project is mainly needed to predict a large spatial grid, that is the natural fit.
- If the TFM never beats RF under spatial CV (per the earlier TabICL benchmark memory), distilling it into a tree ensemble may simply converge to roughly RF accuracy. In that case, distillation is a speed fix with no accuracy upside.

### Gaps
- No independent (non-PriorLabs) replication of TabPFN-2.5-as-MLP/TreeEns accuracy or latency numbers was found. No published "TabICL-to-tree" distillation exists.
- TabPFGen (TabPFN as a generative model for data augmentation) was not researched in depth. It is a data-generation method, not a latency-reduction method, so it is out of scope for speed.
