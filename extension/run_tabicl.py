#!/usr/bin/env python3
"""
run_tabicl.py -- TabICLv2 on MarineSPEED (presences vs background, environmental
predictors only). Same settings as the main study (run_pipeline.py):
n_estimators=2, seed 42, the v2 checkpoint, 4 CPU threads, no KV cache.

    python3 extension/run_tabicl.py              # 40 species x 2 fold kinds x 5 folds
    python3 extension/run_tabicl.py --sweep      # context-size sweep for the timing figure
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner  # noqa: E402

THREADS, SEED = 4, 42
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models", "tabicl-classifier-v2-20260212.ckpt")


def main():
    import numpy as np
    import torch
    torch.set_num_threads(THREADS)
    from tabicl import TabICLClassifier

    def make():
        return TabICLClassifier(n_estimators=2, device="cpu", random_state=SEED,
                                model_path=CKPT)

    # first-call overhead (checkpoint load, kernel warm-up) kept out of the timings
    rng = np.random.default_rng(0)
    w = make()
    w.fit(rng.normal(size=(500, 8)), np.r_[np.ones(50), np.zeros(450)])
    w.predict_proba(rng.normal(size=(50, 8)))

    def fit_predict(Xtr, ytr, Xte):
        clf = make()
        t = time.perf_counter(); clf.fit(Xtr, ytr); fit_s = time.perf_counter() - t
        t = time.perf_counter(); p = clf.predict_proba(Xte)[:, 1]
        return p, fit_s, time.perf_counter() - t

    runner.run("tabicl", fit_predict)


if __name__ == "__main__":
    main()
