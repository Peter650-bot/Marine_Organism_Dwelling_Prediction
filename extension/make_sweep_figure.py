#!/usr/bin/env python3
"""
make_sweep_figure.py -- TabICLv2 vs MaxEnt on MarineSPEED as the context grows;
the counterpart of ../figures/context_sweep_time.pdf (same style).

    figures/sweep_time.{png,pdf}   wall-clock fit+predict time per fold (log-log),
                                   mean ROC-AUC printed at each dot

Data: results/sweep_{tabicl,maxent}.jsonl from `run_*.py --sweep` (spatial
fold 1 of the sweep species; 3 stratified context draws per subsampled size,
1 at full context; identical rows and test sets for both models).
Aggregation: draws averaged within a species, then species averaged. The band
spans the fastest to slowest species.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
from evaluate import FIG, MAXENT, _save, load  # noqa: E402
from make_context_sweep_figure import (INK, INK_2, INK_3, LABEL_BOX, LEG_KW,  # noqa: E402
                                       SURFACE, TABICL, style)

MODELS = [("tabicl", "TabICLv2", TABICL, "s"), ("maxent", "MaxEnt", MAXENT, "o")]
LABEL_OFFSET = {"tabicl": (0, 12), "maxent": (0, -13)}


def summarise():
    s, species = {}, None
    for key, *_ in MODELS:
        recs = load(key, prefix="sweep_")
        sp_set = sorted({r["species"] for r in recs})
        species = sp_set if species is None else species
        assert sp_set == species, "both models must cover the same sweep species"
        for c in sorted({r["context"] for r in recs}, key=lambda c: 10 ** 9 if c == "full" else c):
            t, a, n = [], [], []
            for sp in species:
                rows = [r for r in recs if r["species"] == sp and r["context"] == c]
                assert rows, f"missing {key} {sp} ctx={c}"
                t.append(np.mean([r["total_s"] for r in rows]))
                a.append(np.mean([r["auc"] for r in rows]))
                n.append(np.mean([r["n_context"] for r in rows]))
            s[key, c] = {"t": np.mean(t), "t_min": np.min(t), "t_max": np.max(t),
                         "auc": np.mean(a), "n": np.mean(n), "t_sp": t, "auc_sp": a}
    ctxs = sorted({c for _, c in s}, key=lambda c: 10 ** 9 if c == "full" else c)
    return s, ctxs, species


def report(s, ctxs):
    print(f"{'context':>8s} {'model':7s} {'time_s':>8s} {'min':>7s} {'max':>7s} {'AUC':>6s}")
    for c in ctxs:
        for key, *_ in MODELS:
            v = s[key, c]
            print(f"{v['n']:8,.0f} {key:7s} {v['t']:8.2f} {v['t_min']:7.2f} "
                  f"{v['t_max']:7.2f} {v['auc']:6.3f}")
        print(f"{'':8s} TabICLv2 / MaxEnt time ratio: {s['tabicl', c]['t'] / s['maxent', c]['t']:.1f}x")


def make_time(s, ctxs, species, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for key, label, colour, mark in MODELS:
        v = [s[key, c] for c in ctxs]
        x = np.array([d["n"] for d in v])
        ax.fill_between(x, [d["t_min"] for d in v], [d["t_max"] for d in v],
                        color=colour, alpha=0.10, lw=0, zorder=2)
        ax.plot(x, [d["t"] for d in v], color=colour, lw=2.0, marker=mark,
                ms=7, mec=SURFACE, mew=1.2, zorder=4)
        dx, dy = LABEL_OFFSET[key]
        for xi, d in zip(x, v):
            ax.annotate(f"{d['auc']:.3f}", (xi, d["t"]), xytext=(dx, dy),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=7.8, color=colour, fontweight="bold",
                        bbox=LABEL_BOX, zorder=6)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    ax.set_ylabel("wall-clock mean time per fold (s, log)", fontsize=10, color=INK_2)

    ax.set_xscale("log")
    xt = [s["tabicl", c]["n"] for c in ctxs]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([f"{v:,.0f}" + ("\n(full)" if c == "full" else "")
                        for v, c in zip(xt, ctxs)])
    ax.set_xlim(xt[0] / 1.35, xt[-1] * 1.35)
    ax.set_xlabel(f"context size: training rows given to the model (log scale)\n"
                  f"MarineSPEED, mean of {len(species)} species (spatial fold 1)",
                  fontsize=9, color=INK_2)
    style(ax)
    handles = [Line2D([], [], color=c, lw=2.0, marker=m, ms=7, mec=SURFACE, mew=1.2)
               for _, _, c, m in MODELS]
    leg = ax.legend(handles + [Patch(color=INK_3, alpha=0.18, lw=0)],
                    [n for _, n, _, _ in MODELS] + ["Range across species"],
                    title="Labels = ROC-AUC", loc="upper left", **LEG_KW)
    leg.get_title().set_color(INK_2)
    fig.tight_layout()
    _save(fig, "sweep_time", dpi)
    plt.close(fig)


if __name__ == "__main__":
    s, ctxs, species = summarise()
    report(s, ctxs)
    make_time(s, ctxs, species)
