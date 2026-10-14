#!/usr/bin/env python3
"""
run_maxent.py -- MaxEnt (elapid 1.0.3, Python Maxent) on MarineSPEED, the same
rows as run_tabicl.py. Needs elapid's isolated site directory:

    PYTHONPATH=~/venvs/maxent-site python3 extension/run_maxent.py [--sweep]

maxent.jar defaults: regularisation multiplier 1, cloglog output, clamping,
and feature classes chosen by the number of training presences
(>=80 LQPHT, 15-79 LQH, 10-14 LQ, <10 L; Phillips & Dudik 2008). Background
points are weighted as in elapid's default (class_weights=100).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner  # noqa: E402

THREADS, SEED = 4, 42


def feature_classes(n_pres):
    return "lqpht" if n_pres >= 80 else "lqh" if n_pres >= 15 else "lq" if n_pres >= 10 else "l"


def main():
    import elapid

    def fit_predict(Xtr, ytr, Xte):
        m = elapid.MaxentModel(feature_types=feature_classes(int(ytr.sum())),
                               beta_multiplier=1.0, transform="cloglog", clamp=True,
                               n_cpus=THREADS, random_state=SEED)
        t = time.perf_counter(); m.fit(Xtr, ytr); fit_s = time.perf_counter() - t
        t = time.perf_counter(); p = m.predict(Xte)
        return p, fit_s, time.perf_counter() - t

    runner.run("maxent", fit_predict)


if __name__ == "__main__":
    main()
