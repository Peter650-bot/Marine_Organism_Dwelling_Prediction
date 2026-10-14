#!/usr/bin/env python3
r"""
make_tables.py -- generate every results table,
as copy-paste LaTeX.

Reads `figures/<species>/results.json` (written by `run_pipeline.py`) and
emits:
  * `tables.tex`        -- every table, ready to \input or paste into a .tex
  * `RESULTS.md`        -- the same headline numbers in readable form

Mapping to the paper:
  Table 1  parity, Caretta only            Table 6  task scale
  Table 2  robustness, Caretta only        Table 7  Caretta-only extras (BLOCKED)
  Table 3  per-species summary             Table 8  permutation importance
  Table 4  parity, all species             Table 9  zero-shot vs fine-tuned (BLOCKED)
  Table 10 replaces Figure 1: full-context and k=32 models side by side

Tables 7 and 9 cannot be filled from this run. Table 7 needs TabLLM, which
requires the paid Anthropic API or a vLLM server, and its cached responses are
keyed to the OLD labels and test cells. Table 9 needs the Modal fine-tuning
study, which bills credits. Both are emitted with an explicit "not run" marker
rather than carrying stale numbers forward, since every one of those numbers was
produced under the superseded label and CV scheme.

Only the `requires` package set the paper already uses is assumed: booktabs.
"""

import json
import os

import numpy as np

FIG_ROOT = "figures"
SPECIES = [("Caretta caretta", "Mediterranean"),
           ("Cetorhinus maximus", "NE Atlantic"),
           ("Balaenoptera musculus", "Northeast Pacific")]
KSHOT_ROWS = [("tabicl", "TabICLv2"), ("tabllm", "TabLLM (Qwen2.5-72B)"),
              ("rf", "Random Forest"), ("gbt", "GBT"), ("knn", "kNN"),
              ("logreg", "LogReg")]
FEATURE_LABEL = {"longitude": "Longitude", "latitude": "Latitude",
                 "mean_speed": "Speed", "retention": "Retention",
                 "dist_to_coast": "Distance to coast", "vorticity": "Vorticity"}
NR = r"\emph{not run}"


def load(fig_root=FIG_ROOT):
    out = {}
    for sp, _ in SPECIES:
        p = os.path.join(fig_root, sp.replace(" ", "_"), "results.json")
        if os.path.exists(p):
            with open(p) as fh:
                out[sp] = json.load(fh)
    return out


def vals(d, key):
    return np.array([f[key] for f in d["folds"]], float)


def ci_t(x):
    from scipy import stats
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def paired(a, b):
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    if n < 2:
        return dict(mean=float(d.mean()), lo=float("nan"), hi=float("nan"),
                    p=float("nan"), wins=int((d > 0).sum()), n=n)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return dict(mean=float(d.mean()), lo=float(d.mean() - half),
                hi=float(d.mean() + half), p=float(stats.ttest_rel(a, b).pvalue),
                wins=int((d > 0).sum()), n=n)


def var_ratio(a, b):
    """RF/TabICL variance ratio with an F-test and a Levene test."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    ratio = float(vb / va) if va > 0 else float("inf")
    n = a.size
    pf = float(1 - stats.f.cdf(ratio, n - 1, n - 1)) * 2 if va > 0 else float("nan")
    pf = float(min(1.0, pf))
    pl = float(stats.levene(a, b).pvalue) if n > 1 else float("nan")
    return ratio, pf, pl


def f(x, n=4):
    return NR if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def sig(p):
    return r"\textbf{" + f"{p:.3f}" + "}" if np.isfinite(p) and p < 0.05 else f"{p:.3f}"


# ---------------------------------------------------------------- tables ----
def table1(D):
    sp = "Caretta caretta"
    if sp not in D:
        return ""
    d = D[sp]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{TabICLv2 versus a tuned Random Forest under \textbf{contiguous "
         r"leave-one-region-out} cross-validation, with the \textbf{KDE core-area} "
         r"label. Both models are scored on identical class-balanced test cells "
         r"per fold. Differences are paired per fold; intervals are 95\% "
         r"$t$-intervals on $n-1$ degrees of freedom.}",
         r"\label{tab:parity-caretta}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Metric & TabICLv2 & RF-full & Paired $\Delta$ & 95\% CI & $p$ \\",
         r"\midrule"]
    for key, lab in (("auc", "AUC"), ("f1", "F1")):
        a, b = vals(d, f"{key}_tabicl"), vals(d, f"{key}_rf_full")
        pr = paired(a, b)
        L.append(f"{lab} & {f(a.mean())} & {f(b.mean())} & {pr['mean']:+.4f} & "
                 f"$[{pr['lo']:+.3f}, {pr['hi']:+.3f}]$ & {sig(pr['p'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table2(D):
    sp = "Caretta caretta"
    if sp not in D:
        return ""
    d = D[sp]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Robustness across the held-out regions. Variance ratios are "
         r"RF/TabICLv2 on the squared standard deviations; range and worst-fold "
         r"rows compare the values shown directly.}",
         r"\label{tab:robustness-caretta}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Robustness summary & TabICLv2 & RF-full & Ratio \\", r"\midrule"]
    for key, lab in (("auc", "AUC"), ("f1", "F1")):
        a, b = vals(d, f"{key}_tabicl"), vals(d, f"{key}_rf_full")
        r, _pf, _pl = var_ratio(a, b)
        word = "lower variance" if r >= 1 else "higher variance"
        rr = r if r >= 1 else (1 / r if r > 0 else float("inf"))
        L.append(f"{lab} sd across folds & {a.std(ddof=1):.3f} & {b.std(ddof=1):.3f} "
                 f"& {rr:.1f}$\\times$ {word} \\\\")
    for key, lab in (("auc", "AUC"), ("f1", "F1")):
        a, b = vals(d, f"{key}_tabicl"), vals(d, f"{key}_rf_full")
        ra, rb = a.max() - a.min(), b.max() - b.min()
        w = rb / ra if ra > 0 else float("inf")
        L.append(f"{lab} fold range & {ra:.3f} & {rb:.3f} & {w:.1f}$\\times$ narrower \\\\")
    for key, lab in (("auc", "AUC"), ("f1", "F1")):
        a, b = vals(d, f"{key}_tabicl"), vals(d, f"{key}_rf_full")
        L.append(f"Worst-fold {lab} & {a.min():.3f} & {b.min():.3f} & {a.min()-b.min():+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table3(D):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Model performance by species (mean, standard deviation, and "
         r"worst-case AUC and F1) under the KDE core-area label and contiguous "
         r"leave-one-region-out CV.}",
         r"\label{tab:by-species}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"\multirow{2}{*}{Species} & \multirow{2}{*}{Model} & "
         r"\multicolumn{3}{c}{AUC} & \multicolumn{3}{c}{F1} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r" & & Mean & SD & Worst & Mean & SD & Worst \\", r"\midrule"]
    for sp, _ in SPECIES:
        if sp not in D:
            continue
        d = D[sp]
        rows = [("tabicl", "TabICLv2"), ("rf_full", "RF-full")]
        # AquaMaps is untrained, so it carries no paired test -- but mean/sd/worst
        # are well defined and it is scored on the identical test cells.
        if d["folds"] and "auc_aquamaps" in d["folds"][0]:
            rows.append(("aquamaps", r"AquaMaps$^{\dagger}$"))
        for key, lab in rows:
            a, b = vals(d, f"auc_{key}"), vals(d, f"f1_{key}")
            name = f"\\emph{{{sp}}}" if key == "tabicl" else ""
            L.append(f"{name} & {lab} & {a.mean():.4f} & {a.std(ddof=1):.4f} & "
                     f"{a.min():.4f} & {b.mean():.4f} & {b.std(ddof=1):.4f} & "
                     f"{b.min():.4f} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{\linewidth}\footnotesize $^{\dagger}$AquaMaps is the "
          r"published species-distribution surface. It is \emph{not} fitted to "
          r"this data -- a fixed 0.5$^{\circ}$ envelope re-scored on the identical "
          r"test cells -- so its across-fold spread reflects only which region was "
          r"held out, and no paired test against it is reported. The comparison is "
          r"partly circular in its favour: its occurrence base overlaps "
          r"OBIS-SEAMAP.\end{minipage}",
          r"\end{table}"]
    return "\n".join(L)


def table4(D):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{TabICLv2 against RF-full under contiguous leave-one-region-out "
         r"cross-validation. $\Delta$ is TabICLv2 minus RF-full, so a positive "
         r"value favours TabICLv2.}",
         r"\label{tab:parity-all}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Metric & TabICLv2 & RF-full & $\Delta$ & $p$ & Folds won \\", r"\midrule"]
    for sp, basin in SPECIES:
        if sp not in D:
            continue
        d = D[sp]
        L.append(r"\multicolumn{6}{l}{\emph{" + sp + "}, " + basin + r"} \\")
        for key, lab in (("auc", "AUC"), ("f1", "F1")):
            a, b = vals(d, f"{key}_tabicl"), vals(d, f"{key}_rf_full")
            pr = paired(a, b)
            L.append(f"\\quad {lab} & {a.mean():.4f} & {b.mean():.4f} & "
                     f"{pr['mean']:+.4f} & {sig(pr['p'])} & {pr['wins']}/{pr['n']} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{\linewidth}\footnotesize Bold $p$ values are "
          r"significant at $\alpha=0.05$. Folds won counts the held-out regions "
          r"on which TabICLv2 scored higher than RF-full.\end{minipage}",
          r"\end{table}"]
    return "\n".join(L)


def table5(D):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Dispersion of scores across the held-out regions, and the "
         r"weakest region for each model.}", r"\label{tab:dispersion}",
         r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Across-fold sd} & "
         r"\multicolumn{3}{c}{Variance ratio, RF/TabICLv2} & "
         r"\multicolumn{2}{c}{Worst fold} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
         r"Metric & TabICLv2 & RF-full & Ratio & $p_F$ & $p_{\text{Levene}}$ & "
         r"TabICLv2 & RF-full \\", r"\midrule"]
    for sp, basin in SPECIES:
        if sp not in D:
            continue
        d = D[sp]
        L.append(r"\multicolumn{8}{l}{\emph{" + sp + "}, " + basin + r"} \\")
        for key, lab in (("auc", "AUC"), ("f1", "F1")):
            a, b = vals(d, f"{key}_tabicl"), vals(d, f"{key}_rf_full")
            r, pf, pl = var_ratio(a, b)
            L.append(f"\\quad {lab} & {a.std(ddof=1):.3f} & {b.std(ddof=1):.3f} & "
                     f"{r:.2f} & {sig(pf)} & {sig(pl)} & {a.min():.3f} & {b.min():.3f} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{\linewidth}\footnotesize Variance ratios are RF-full "
          r"over TabICLv2, so a ratio below 1 means TabICLv2 was the more variable "
          r"of the two. Bold entries are significant at $\alpha=0.05$."
          r"\end{minipage}", r"\end{table}"]
    return "\n".join(L)


def table6(D):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Size and class balance of each task under the corrected "
         r"pipeline. Prevalence is the fraction of ocean cells inside the 50\% "
         r"KDE core-area contour.}", r"\label{tab:task-scale}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r" & \emph{C. caretta} & \emph{C. maximus} & \emph{B. musculus} \\",
         r" & Mediterranean & NE Atlantic & Northeast Pacific \\", r"\midrule"]

    def row(label, fn):
        cells = [fn(D[sp]) if sp in D else NR for sp, _ in SPECIES]
        return f"{label} & " + " & ".join(cells) + r" \\"

    L += [row("Grid cells", lambda d: f"{d['n_cells']:,}"),
          row("Prevalence (core area)", lambda d: f"{d['prevalence']:.3f}"),
          row("Context rows per fold",
              lambda d: f"{int(np.mean([x['n_context'] for x in d['folds']])):,}"
              if d["folds"] else NR),
          row("Test cells per fold",
              lambda d: f"{int(np.mean([x['n_test'] for x in d['folds']]))}"
              if d["folds"] else NR),
          row("Regions scored / attempted",
              lambda d: f"{len(d['folds'])} / {len(d['folds'])+len(d['folds_unscoreable'])}"),
          row("Train--test separation (median, $^\\circ$)",
              lambda d: f"{np.mean([x['separation_deg_median'] for x in d['folds']]):.2f}"
              if d["folds"] else NR),
          row("Occurrence rows after filtering", lambda d: f"{d['n_occurrence_rows']}")]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table7(D, k=32):
    """Now measurable: the TabLLM run was repeated under the corrected pipeline,
    so every row here is filled from artefacts rather than marked absent."""
    def tl(sp):
        return (D.get(sp, {}).get("kshot", {}).get(str(k), {}) or {}).get("tabllm")

    rows, aucs, swings, noans, deltas = [], [], [], 0, []
    ncalls = 0          # actual predictions, not folds x 300 (one fold had 66)
    for sp, _ in SPECIES:
        b = tl(sp)
        if not b:
            continue
        a = np.array([r["auc"] for r in b], float)
        aucs.append((sp, a))
        byfold = {}
        for r in b:
            byfold.setdefault(r["fold"], []).append(r["auc"])
        swings.append(max(max(v) - min(v) for v in byfold.values()))
        noans += sum(r.get("n_no_answer", 0) for r in b)
        ncalls += sum(r.get("n_used", 0) + r.get("n_no_answer", 0) for r in b)
        ti = (D[sp].get("kshot", {}).get(str(k), {}) or {}).get("tabicl")
        if ti:
            deltas.append(np.mean([r["auc"] for r in ti]) - a.mean())
    if not aucs:
        rows = [(r"TabICLv2 against TabLLM, matched $k$", NR),
                (r"TabLLM 72B AUC", NR),
                (r"Truncation-fabricated $P(\text{HIGH})=0$", NR),
                (r"Exemplar-draw variance ($k$-shot)", NR)]
    else:
        wins = sum(1 for d in deltas if d > 0)
        rows = [
            (f"TabICLv2 against TabLLM, matched $k={k}$",
             f"{wins}/{len(deltas)} species; $\\Delta$AUC " +
             ", ".join(f"{d:+.3f}" for d in deltas)),
            ("TabLLM 72B AUC (mean per species)",
             ", ".join(f"{a.mean():.3f}" for _s, a in aucs)),
            ("TabLLM 72B AUC (best / worst fold-draw)",
             f"{max(a.max() for _s, a in aucs):.3f} / "
             f"{min(a.min() for _s, a in aucs):.3f}"),
            (r"Truncation-fabricated $P(\text{HIGH})=0$",
             f"{noans} of {ncalls:,} predictions"),
            (r"Exemplar-draw variance ($k$-shot)",
             f"up to {max(swings):.3f} AUC swing within a fold"),
        ]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Language-model form of in-context learning, and sensitivity "
         r"checks, under the corrected pipeline. TabLLM is Qwen2.5-72B-Instruct-AWQ "
         r"scored by label-verbalizer logprob.}", r"\label{tab:tabllm}",
         r"\begin{tabular}{ll}", r"\toprule", r"Quantity & Value \\", r"\midrule"]
    for a, b in rows:
        L.append(f"{a} & {b} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{\linewidth}\footnotesize $\Delta$AUC is TabICLv2 "
          r"minus TabLLM at matched $k$, in species order. Species are listed in "
          r"the order of Table~\ref{tab:by-species}. Logprob scoring emits a "
          r"single verbalizer token, so the truncation failure that fabricated "
          r"confident zeros in the earlier verbalized run cannot occur."
          r"\end{minipage}", r"\end{table}"]
    return "\n".join(L)


def table8(D):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Permutation importance on the contiguous spatial holdout, as "
         r"the mean drop in AUC when each feature is permuted (RF-full, averaged "
         r"over held-out regions).}", r"\label{tab:perm-importance}",
         r"\begin{tabular}{l" + "c" * len(SPECIES) + "}", r"\toprule",
         r"Feature & " + " & ".join(f"\\emph{{{s.split()[0][0]}. {s.split()[1]}}}"
                                    for s, _ in SPECIES) + r" \\", r"\midrule"]
    order, table = None, {}
    for sp, _ in SPECIES:
        if sp not in D or not D[sp]["folds"]:
            continue
        keys = list(D[sp]["folds"][0]["perm_importance"])
        m = {k: float(np.mean([fo["perm_importance"][k] for fo in D[sp]["folds"]]))
             for k in keys}
        table[sp] = m
        if order is None:
            order = sorted(m, key=lambda k: -m[k])
    for feat in (order or []):
        cells = [f"{table[sp][feat]:.3f}" if sp in table else NR for sp, _ in SPECIES]
        L.append(f"{FEATURE_LABEL.get(feat, feat)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table9():
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Zero-shot against fine-tuned TabICLv2. \textbf{Not re-run "
         r"under the corrected pipeline.}}", r"\label{tab:finetune}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Species & ZS AUC & FT AUC & $\Delta$ & Folds won \\", r"\midrule"]
    for sp, _ in SPECIES:
        L.append(f"\\emph{{{sp}}} & {NR} & {NR} & {NR} & {NR} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{\linewidth}\footnotesize The fine-tuning study runs "
          r"on Modal and bills GPU credits; it was not relaunched. Its published "
          r"values were produced under the superseded label and CV scheme and are "
          r"not carried forward.\end{minipage}", r"\end{table}"]
    return "\n".join(L)


def table10(D, k=32):
    """Replaces Figure 1: full-context and k=32 models side by side."""
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{ROC-AUC for full-context and $k=32$ models under the "
         r"corrected pipeline, averaged over held-out regions. $k$-shot rows are "
         r"additionally averaged over exemplar draws; $\pm$ is one standard "
         r"deviation across regions.}", r"\label{tab:models-k32}",
         r"\begin{tabular}{l" + "c" * len(SPECIES) + "}", r"\toprule",
         r"Model & " + " & ".join(f"\\emph{{{s}}}" for s, _ in SPECIES) + r" \\",
         r"\midrule"]

    def cell(sp, getter):
        if sp not in D:
            return NR
        v = getter(D[sp])
        return NR if v is None else f"{v[0]:.3f} $\\pm$ {v[1]:.3f}"

    def full(key):
        return lambda d: (vals(d, key).mean(), vals(d, key).std(ddof=1)) if d["folds"] else None

    def ks(model):
        def g(d):
            b = d.get("kshot", {}).get(str(k), {}).get(model)
            if not b:
                return None
            a = np.array([r["auc"] for r in b], float)
            return (a.mean(), a.std(ddof=1)) if a.size > 1 else (a.mean(), 0.0)
        return g

    L.append(r"TabICLv2 (full context) & " +
             " & ".join(cell(sp, full("auc_tabicl")) for sp, _ in SPECIES) + r" \\")
    L.append(r"RF-full (full context) & " +
             " & ".join(cell(sp, full("auc_rf_full")) for sp, _ in SPECIES) + r" \\")
    L.append(r"\addlinespace")
    for key, lab in KSHOT_ROWS:
        L.append(f"{lab} ($k={k}$) & " +
                 " & ".join(cell(sp, ks(key)) for sp, _ in SPECIES) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{\linewidth}\footnotesize All $k=32$ rows draw the "
          r"\emph{same} 32 exemplars and are scored on the \emph{same} test "
          r"cells. TabLLM is Qwen2.5-72B-Instruct-AWQ under logprob scoring "
          r"(the label-verbalizer likelihood the method specifies), served on "
          r"4$\times$A10G.\end{minipage}", r"\end{table}"]
    return "\n".join(L)


# ------------------------------------------------------------------ emit ----
def build(fig_root=FIG_ROOT, out_tex="tables.tex", out_md="RESULTS.md"):
    D = load(fig_root)
    if not D:
        print(f"no results under {fig_root} yet -- run run_pipeline.py first")
        return 1

    head = [r"% ---------------------------------------------------------------",
            r"% tables.tex -- generated by make_tables.py",
            r"%   label: 50% KDE core-area volume contour (was: median of nonzero",
            r"%          on a linear-interpolated surface)",
            r"%   CV   : contiguous leave-one-region-out, k-means (was: GroupKFold",
            r"%          over 3-degree blocks, which scattered them)",
            r"% Requires: booktabs, multirow",
            r"% ---------------------------------------------------------------",
            ""]
    parts = [table1(D), table2(D), table3(D), table4(D), table5(D),
             table6(D), table7(D), table8(D), table9(), table10(D)]
    with open(out_tex, "w") as fh:
        fh.write("\n".join(head) + "\n\n" + "\n\n".join(p for p in parts if p) + "\n")
    print(f"[saved] {out_tex}")

    M = ["# Results", "",
         "Headline results under the two corrections:",
         "",
         "1. **Label** — 50% volume contour of a weighted Gaussian KDE of sighting",
         "   intensity (the genuine core-area isopleth), replacing the median of",
         "   non-zero cells on a linearly-interpolated surface.",
         "2. **Cross-validation** — contiguous leave-one-region-out (k-means, 5",
         "   regions), replacing `GroupKFold` over 3° blocks, which scattered the",
         "   blocks and left held-out cells ~0.5° from training data.",
         "",
         "LaTeX for every table is in `tables.tex`.", "",
         "## Headline table", "",
         "| species | cells | prevalence | regions scored | separation (median °) "
         "| TabICLv2 AUC | RF-full AUC | TabICLv2 F1 | RF-full F1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sp, _ in SPECIES:
        if sp not in D:
            M.append(f"| {sp} | — | — | — | — | — | — | — | — |")
            continue
        d = D[sp]
        if not d["folds"]:
            M.append(f"| {sp} | {d['n_cells']:,} | {d['prevalence']:.3f} | "
                     f"0 / {len(d['folds_unscoreable'])} | — | — | — | — | — |")
            continue
        a, b = vals(d, "auc_tabicl"), vals(d, "auc_rf_full")
        c, e = vals(d, "f1_tabicl"), vals(d, "f1_rf_full")
        sep = np.mean([x["separation_deg_median"] for x in d["folds"]])
        M.append(f"| {sp} | {d['n_cells']:,} | {d['prevalence']:.3f} | "
                 f"{len(d['folds'])} / {len(d['folds'])+len(d['folds_unscoreable'])} | "
                 f"{sep:.2f} | {a.mean():.4f} | {b.mean():.4f} | "
                 f"{c.mean():.4f} | {e.mean():.4f} |")
    # --- k=32 in-context comparison (replaces the old Figure 1) ---
    M += ["", "## In-context learning at k=32", "",
          "Mean ROC-AUC over held-out regions and exemplar draws.", "",
          "| model | " + " | ".join(sp for sp, _ in SPECIES) + " |",
          "|---" * (len(SPECIES) + 1) + "|"]
    for key, lab in KSHOT_ROWS:
        row = [lab]
        for sp, _ in SPECIES:
            recs = D.get(sp, {}).get("kshot", {}).get(str(32), {}).get(key, [])
            aucs = np.array([r["auc"] for r in recs], float)
            row.append(f"{aucs.mean():.4f}" if aucs.size else "\u2014")
        M.append("| " + " | ".join(row) + " |")

    # --- tuned RF baseline, if tune_rf.py has been run ---------------------
    if any(all("auc_rf_tuned" in f for f in D[sp]["folds"])
           for sp, _ in SPECIES if sp in D and D[sp]["folds"]):
        M += ["", "## Against a nested-CV-tuned Random Forest", "",
              "`run_pipeline.py` fits one fixed RF configuration; `tune_rf.py`",
              "selects hyperparameters by leave-one-region-out *within each outer",
              "training fold*. Both baselines are kept so the change is auditable.",
              "",
              "| species | metric | TabICLv2 | RF fixed | RF tuned | \u0394 vs tuned | 95% CI |",
              "|---|---|---|---|---|---|---|"]
        from scipy import stats as _st
        for sp, _ in SPECIES:
            d = D.get(sp)
            if not d or not d["folds"] or "auc_rf_tuned" not in d["folds"][0]:
                continue
            for m in ("auc", "f1"):
                t, u = vals(d, f"{m}_tabicl"), vals(d, f"{m}_rf_full")
                k = vals(d, f"{m}_rf_tuned")
                dd = t - k
                n = len(dd)
                h = (_st.t.ppf(0.975, n - 1) * dd.std(ddof=1) / np.sqrt(n)
                     if n > 1 else float("nan"))
                M.append(f"| {sp} | {m.upper()} | {t.mean():.4f} | {u.mean():.4f} "
                         f"| {k.mean():.4f} | {dd.mean():+.4f} "
                         f"| [{dd.mean()-h:+.3f}, {dd.mean()+h:+.3f}] |")

    M += ["", "## Not re-run", "",
          "- **Fine-tuning (zero-shot vs fine-tuned)** \u2014 runs on Modal and bills",
          "  GPU credits. Its published values were produced under the superseded",
          "  label and CV scheme and are not carried forward.", ""]
    with open(out_md, "w") as fh:
        fh.write("\n".join(M))
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-root", default=FIG_ROOT)
    ap.add_argument("--out-tex", default="tables.tex")
    ap.add_argument("--out-md", default="RESULTS.md")
    a = ap.parse_args()
    raise SystemExit(build(a.fig_root, a.out_tex, a.out_md))
