#!/usr/bin/env python3
"""
make_context_sweep_figure.py -- TabICLv2 vs Random Forest on the loggerhead
turtle as the context grows (500 rows -> the full ~10.9k-row training fold).

    figures/context_sweep_time.png / .pdf   wall-clock mean time per fold
                                            (log-log), mean ROC-AUC at each dot
    figures/context_sweep_auc.png / .pdf    mean ROC-AUC with bootstrap 95% CI

Data: context_sweep.jsonl from bench_context_sweep.py. Both figures plot
TabICLv2 and the tuned RF; the fixed RF is in the printed table only.

Aggregation: draws are averaged within a fold, then folds are averaged
(4 scoreable folds). The time band spans the fastest to slowest fold.

Bootstrap (B = 2000, seed 0): each replicate resamples, within every fold,
the positive and the negative test cells separately (so the 150/150 balance is
kept) and, at subsampled contexts, the 3 context draws. The same resample is
applied to every model and context size, so the intervals are paired. The CI
reflects test-cell and context-draw sampling. With only 4 regions it does not
cover region-to-region variation; the per-fold spread is printed for that.
"""

import json
import os

import numpy as np

SWEEP = "context_sweep.jsonl"
FIG_ROOT = "figures"
OUT_TIME, OUT_AUC = "context_sweep_time", "context_sweep_auc"
TABICL, RF = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e4e3df", "#fcfcfb"
LABEL_BOX = dict(boxstyle="square,pad=0.12", fc=SURFACE, ec="none", alpha=0.9)
B, BOOT_SEED = 2000, 0
MODELS = [("tabicl", "TabICLv2", TABICL, "s"),
          ("rf_tuned", "Tuned RF (nested-CV search + fit)", RF, "o"),
          ("rf_fixed", "Fixed RF (one fit, no search)", INK_3, "^")]


def load(path=SWEEP):
    with open(path) as fh:
        recs = [json.loads(line) for line in fh]
    for r in recs:
        r["ctx_key"] = 0 if r["context"] == "full" else int(r["context"])
    return recs


def pairwise(p, y):
    """C[i, j] = 1{p_pos_i > p_neg_j} + 0.5 * tie, so AUC = mean(C)."""
    p, y = np.asarray(p), np.asarray(y)
    pp, pn = p[y == 1], p[y == 0]
    return (pp[:, None] > pn[None, :]) + 0.5 * (pp[:, None] == pn[None, :])


def summarise(recs):
    """Per (model, context): fold-level means, overall mean, bootstrap CI."""
    folds = sorted({r["fold"] for r in recs})
    ctxs = sorted({r["ctx_key"] for r in recs}, key=lambda c: c or 10 ** 9)
    rng = np.random.default_rng(BOOT_SEED)

    # one shared resample per fold: multinomial weights over pos / neg cells
    ref = {f: next(r for r in recs if r["fold"] == f) for f in folds}
    wts = {}
    for f in folds:
        y = np.asarray(ref[f]["y_test"])
        n_p, n_n = int(y.sum()), int((y == 0).sum())
        wts[f] = (rng.multinomial(n_p, np.full(n_p, 1 / n_p), size=B) / n_p,
                  rng.multinomial(n_n, np.full(n_n, 1 / n_n), size=B) / n_n)
    max_draws = max(r["draw"] for r in recs) + 1
    draw_pick = {f: rng.integers(0, max_draws, size=(B, max_draws)) for f in folds}

    out = {}
    for key, *_ in MODELS:
        for c in ctxs:
            rows = [r for r in recs if r["model"] == key and r["ctx_key"] == c]
            if not rows:
                continue
            t_fold, a_fold, a_boot, n_ctx = [], [], np.zeros(B), []
            for f in folds:
                fr = sorted((r for r in rows if r["fold"] == f), key=lambda r: r["draw"])
                assert fr, f"missing {key} ctx={c} fold={f}"
                assert all(r["test_idx"] == ref[f]["test_idx"] for r in fr)
                t_fold.append(np.mean([r["total_s"] for r in fr]))
                a_fold.append(np.mean([r["auc"] for r in fr]))
                n_ctx.append(np.mean([r["n_context"] for r in fr]))
                wp, wn = wts[f]
                per_draw = np.stack([np.einsum("bi,ij,bj->b", wp,
                                               pairwise(r["p_test"], r["y_test"]), wn)
                                     for r in fr], axis=1)           # B x n_draws
                pick = draw_pick[f][:, :len(fr)] % len(fr)
                a_boot += np.take_along_axis(per_draw, pick, axis=1).mean(axis=1)
            a_boot /= len(folds)
            out[key, c] = {"t": np.mean(t_fold), "t_min": np.min(t_fold),
                           "t_max": np.max(t_fold), "auc": np.mean(a_fold),
                           "lo": np.percentile(a_boot, 2.5),
                           "hi": np.percentile(a_boot, 97.5),
                           "boot": a_boot, "auc_folds": a_fold, "t_folds": t_fold,
                           "n": float(np.mean(n_ctx)), "draws": len(rows) // len(folds)}
    return out, ctxs, folds


def report(s, ctxs):
    print(f"{'context':>8s} {'model':9s} {'draws':>5s} {'time_s':>8s} {'min':>7s} "
          f"{'max':>7s} {'AUC':>6s} {'95% CI':>15s}   per-fold AUC")
    for c in ctxs:
        for key, *_ in MODELS:
            if (key, c) not in s:
                continue
            v = s[key, c]
            print(f"{v['n']:8,.0f} {key:9s} {v['draws']:5d} {v['t']:8.1f} {v['t_min']:7.1f} "
                  f"{v['t_max']:7.1f} {v['auc']:6.3f} [{v['lo']:.3f}, {v['hi']:.3f}]   "
                  + " ".join(f"{a:.3f}" for a in v["auc_folds"]))
    print("\npaired TabICLv2 - tuned RF, mean AUC difference (bootstrap 95% CI):")
    for c in ctxs:
        if ("tabicl", c) in s and ("rf_tuned", c) in s:
            d = s["tabicl", c]["boot"] - s["rf_tuned", c]["boot"]
            diff = s["tabicl", c]["auc"] - s["rf_tuned", c]["auc"]
            print(f"{s['tabicl', c]['n']:8,.0f}  {diff:+.3f} "
                  f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", which="major", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", colors=INK_2, labelsize=8.5, length=3)
    ax.tick_params(axis="x", which="minor", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


# where each model's AUC label sits relative to its dot (points)
LABEL_OFFSET = {"tabicl": (0, -13), "rf_tuned": (0, 9)}
PLOT_MODELS = [m for m in MODELS if m[0] != "rf_fixed"]   # fixed RF: table only
SHORT_NAMES = ["TabICLv2", "Tuned RF"]
LEG_KW = dict(frameon=False, fontsize=8.5, labelcolor=INK, handlelength=2.2,
              title_fontsize=8.5, alignment="left")


def _series(s, ctxs, key):
    cs = [c for c in ctxs if (key, c) in s]
    return np.array([s[key, c]["n"] for c in cs]), [s[key, c] for c in cs]


def _context_axis(ax, s, ctxs, folds):
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.set_xscale("log")
    xt = [s["tabicl", c]["n"] for c in ctxs]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([f"{v:,.0f}" + ("\n(full)" if c == 0 else "")
                        for v, c in zip(xt, ctxs)])
    ax.set_xlim(xt[0] / 1.35, xt[-1] * 1.35)
    ax.set_xlabel(f"context size: training rows given to the model (log scale)\n"
                  f"loggerhead turtle, Mediterranean, mean of {len(folds)} held-out folds",
                  fontsize=9, color=INK_2)
    style(ax)


def _save(fig, name, dpi):
    os.makedirs(FIG_ROOT, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_ROOT, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {path}")


def make_time(s, ctxs, folds, dpi):
    """Mean wall-clock time per fold (log-log), mean AUC printed at each dot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for key, label, colour, mark in PLOT_MODELS:
        x, v = _series(s, ctxs, key)
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
    ax.set_ylim(2, 300)
    ax.set_ylabel("wall-clock mean time per fold (s, log)", fontsize=10, color=INK_2)
    _context_axis(ax, s, ctxs, folds)
    handles = [Line2D([], [], color=c, lw=2.0, marker=m, ms=7, mec=SURFACE, mew=1.2)
               for _, _, c, m in PLOT_MODELS]
    leg = ax.legend(handles + [Patch(color=INK_3, alpha=0.18, lw=0)],
                    SHORT_NAMES + ["Range across folds"], title="Labels = ROC-AUC",
                    loc="lower right", **LEG_KW)
    leg.get_title().set_color(INK_2)
    fig.tight_layout()
    _save(fig, OUT_TIME, dpi)
    plt.close(fig)


def make_auc(s, ctxs, folds, dpi):
    """Mean ROC-AUC per context size with its bootstrap 95% CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for key, label, colour, mark in PLOT_MODELS:
        x, v = _series(s, ctxs, key)
        xa = x * {"tabicl": 0.975, "rf_tuned": 1.025}[key]
        ax.vlines(xa, [d["lo"] for d in v], [d["hi"] for d in v], color=colour,
                  lw=1.4, zorder=3)
        ax.plot(xa, [d["auc"] for d in v], color=colour, lw=1.6, marker=mark,
                ms=6, mec=SURFACE, mew=1.0, zorder=4)
        for xi, d in zip(xa, v):
            ax.annotate(f"{d['auc']:.3f}", (xi, d["auc"]),
                        xytext=(9, 9) if key == "tabicl" else (9, -10),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=7.8, color=colour, bbox=LABEL_BOX, zorder=6)
    ax.axhline(1.0, color=INK_3, lw=1.0, ls=":", zorder=1)
    ax.set_ylim(top=1.006)
    ax.set_ylabel("ROC-AUC on held-out region\n(bootstrap 95% CI)",
                  fontsize=10, color=INK_2)
    _context_axis(ax, s, ctxs, folds)
    handles = [Line2D([], [], color=c, lw=1.6, marker=m, ms=6, mec=SURFACE, mew=1.0)
               for _, _, c, m in PLOT_MODELS]
    ax.legend(handles + [Line2D([], [], color=INK_2, lw=0, marker="|", ms=12, mew=1.4)],
              SHORT_NAMES + ["95% bootstrap CI"], loc="center", ncol=3,
              bbox_to_anchor=(0.5, 0.62), **LEG_KW)
    fig.tight_layout()
    _save(fig, OUT_AUC, dpi)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    s, ctxs, folds = summarise(load())
    report(s, ctxs)
    make_time(s, ctxs, folds, a.dpi)
    make_auc(s, ctxs, folds, a.dpi)
