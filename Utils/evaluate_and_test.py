import argparse
import glob
import os
import re
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

DATASETS = ["DS1", "DS2", "DS3"]
BASELINES = ["GGI-DDI", "SSF-DDI", "LSFC", "DPSP", "GNN-DDI", "MSTE", "MDF-SA-DDI",
             "NNPS", "DDIMDL", "RF", "KNN", "LR", "DT"]
ABLATIONS = {"NoDisc": "No discriminator", "NoAEDisc": "No autoencoder, no discriminator"}
ORDER_DOUBLED = {"ADEP", "NoDisc", "NoAEDisc", "ADEP_fullcov"}
METRICS = ["ACC", "Macro-F1", "Macro-Precision", "Macro-Recall", "Macro-AUROC", "Macro-AUPRC"]
N_CLASSES = {"DS1": 65, "DS2": 100, "DS3": 185}


# ------------------------------------------------------------------ loading
def load(results, ds, method, fold):
    base = os.path.join(results, f"{ds}_{method}_fold{fold}")
    if not os.path.exists(base + "_prob.npy"):
        return None
    P = np.load(base + "_prob.npy").astype(np.float64)
    y = np.load(base + "_true.npy")
    inv = None
    if method in ORDER_DOUBLED:
        n = len(y) // 2
        assert np.array_equal(y[:n], y[n:]), f"{base}: the two halves have different labels"
        inv = float(np.mean(P[:n].argmax(1) == P[n:].argmax(1)))
        P, y = (P[:n] + P[n:]) / 2, y[n:]
    return P, y, inv


# ------------------------------------------------------------------ metrics
def metrics(y, P, C, classes=None):
    """Macro metrics; if `classes` is given, macro-average only over those classes."""
    yhat = P.argmax(1)
    labels = np.arange(C) if classes is None else np.array(sorted(classes))
    present = [c for c in labels if (y == c).any()]          # AUROC/AUPRC need positives
    Y = label_binarize(y, classes=np.arange(C))
    out = {
        "ACC": accuracy_score(y, yhat) if classes is None else np.mean(yhat[np.isin(y, labels)] == y[np.isin(y, labels)]),
        "Macro-F1": f1_score(y, yhat, labels=labels, average="macro", zero_division=0),
        "Macro-Precision": precision_score(y, yhat, labels=labels, average="macro", zero_division=0),
        "Macro-Recall": recall_score(y, yhat, labels=labels, average="macro", zero_division=0),
        "Macro-AUROC": np.mean([roc_auc_score(Y[:, c], P[:, c]) for c in present]),
        "Macro-AUPRC": np.mean([average_precision_score(Y[:, c], P[:, c]) for c in present]),
    }
    return {k: 100 * v for k, v in out.items()}


# ------------------------------------------------------------------ tests
def sign_randomization(d):
    """Exact one-sided p: P(mean of sign-flipped d >= observed mean)."""
    obs = d.mean()
    perms = [np.mean(np.array(s) * d) for s in product([-1, 1], repeat=len(d))]
    return float(np.mean(np.array(perms) >= obs - 1e-12))


def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def compare(a, b):
    d = np.asarray(a) - np.asarray(b)
    t_p = stats.ttest_rel(a, b).pvalue if np.any(d != d[0]) else (0.0 if d[0] != 0 else 1.0)
    lo, hi = bootstrap_ci(d)
    return {"mean_delta": d.mean(), "ci_low": lo, "ci_high": hi, "t_p": float(t_p),
            "sign_p": sign_randomization(d), "folds_won": int((d > 0).sum())}


def fmt(x):
    return f"{np.mean(x):.2f}$\\pm${np.std(x, ddof=1):.2f}"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ main
def main(results, out):
    os.makedirs(out, exist_ok=True)
    methods = sorted({re.match(r".*?_(.+)_fold\d+_prob\.npy", os.path.basename(f)).group(1)
                      for f in glob.glob(os.path.join(results, "*_prob.npy"))})
    print("methods found:", methods)

    rows, inv_rows, minority_rows = [], [], []
    for ds in DATASETS:
        C = N_CLASSES[ds]
        for k in range(1, 6):
            counts_f = os.path.join(results, f"{ds}_ADEP_fold{k}_train_class_counts.csv")
            minority = None
            if os.path.exists(counts_f):
                cnt = pd.read_csv(counts_f, index_col=0).iloc[:, 0].reindex(range(C), fill_value=0)
                n_min = int(np.ceil(0.25 * C))
                minority = set(cnt.sort_values(kind="stable").index[:n_min])
            for m in methods:
                r = load(results, ds, m, k)
                if r is None:
                    continue
                P, y, inv = r
                rows.append({"dataset": ds, "method": m, "fold": k, **metrics(y, P, C)})
                if inv is not None:
                    inv_rows.append({"dataset": ds, "method": m, "fold": k, "identical_pct": 100 * inv})
                if minority is not None and m in ("ADEP", "NoDisc"):
                    # Macro P/R/F1/AUROC/AUPRC averaged over minority classes only, computed on the
                    # FULL test fold, so false positives from majority-class pairs still count against
                    # minority precision. ACC/errors: test pairs whose true class is a minority class.
                    mask = np.isin(y, list(minority))
                    mm = metrics(y, P, C, classes=minority)
                    mm["errors"] = int((P[mask].argmax(1) != y[mask]).sum())
                    mm["support"] = int(mask.sum())
                    minority_rows.append({"dataset": ds, "method": m, "fold": k, **mm})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metrics_by_fold.csv"), index=False)
    if inv_rows:
        inv = pd.DataFrame(inv_rows).groupby(["dataset", "method"])["identical_pct"].agg(["mean", "std"])
        inv.to_csv(os.path.join(out, "order_invariance.csv"))
        print("\nOrder invariance (% test pairs, identical prediction for both orders):\n", inv)

    def series(ds, m, metric):
        return df[(df.dataset == ds) & (df.method == m)].sort_values("fold")[metric].values

    # ---- benchmark tables
    for ds in DATASETS:
        lines = []
        for m in ["ADEP"] + BASELINES:
            if not ((df.dataset == ds) & (df.method == m)).any():
                continue
            lines.append(m + " & " + " & ".join(fmt(series(ds, m, x)) for x in METRICS) + r" \\")
        open(os.path.join(out, f"table_{ds}.tex"), "w").write("\n".join(lines) + "\n")

    # ---- ablation table
    lines = []
    for abl, name in ABLATIONS.items():
        lines.append(f"% ---- {name}")
        for ds in DATASETS:
            if not ((df.dataset == ds) & (df.method == abl)).any():
                continue
            lines.append(f"{ds} & Full ADEP & " + " & ".join(fmt(series(ds, "ADEP", x)) for x in METRICS) + r" \\")
            lines.append(f" & {name} & " + " & ".join(fmt(series(ds, abl, x)) for x in METRICS) + r" \\" + "\n\\hline")
    open(os.path.join(out, "table_ablation.tex"), "w").write("\n".join(lines) + "\n")

    # ---- statistics
    st = []
    for ds in DATASETS:
        for metric in METRICS:
            fam = []
            for comp in BASELINES:
                if ((df.dataset == ds) & (df.method == comp)).any():
                    fam.append({"dataset": ds, "metric": metric, "comparator": comp, "family": "baselines",
                                **compare(series(ds, "ADEP", metric), series(ds, comp, metric))})
            for r, h in zip(fam, holm([r["t_p"] for r in fam]) if fam else []):
                r["holm_p"] = h
            st += fam
        for abl in ABLATIONS:
            if not ((df.dataset == ds) & (df.method == abl)).any():
                continue
            fam = [{"dataset": ds, "metric": metric, "comparator": abl, "family": "ablation",
                    **compare(series(ds, "ADEP", metric), series(ds, abl, metric))} for metric in METRICS]
            for r, h in zip(fam, holm([r["t_p"] for r in fam])):
                r["holm_p"] = h
            st += fam
    st = pd.DataFrame(st)
    st.to_csv(os.path.join(out, "stats.csv"), index=False)
    with open(os.path.join(out, "S3_rows.tex"), "w") as f:
        for _, r in st.iterrows():
            f.write(f"{r.dataset} & {r.metric} & {r.comparator} & {r.mean_delta:+.2f} & "
                    f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}] & {pfmt(r.t_p)} & {pfmt(r.holm_p)} & "
                    f"{pfmt(r.sign_p)} & {r.folds_won}/5 \\\\\n")

    # ---- minority analysis (rarest quartile of classes by training-fold frequency)
    if minority_rows:
        mdf = pd.DataFrame(minority_rows)
        mdf.to_csv(os.path.join(out, "minority_by_fold.csv"), index=False)
        mcols = METRICS + ["errors"]
        res, tex = [], []
        for ds in DATASETS:
            a = mdf[(mdf.dataset == ds) & (mdf.method == "ADEP")].sort_values("fold")
            b = mdf[(mdf.dataset == ds) & (mdf.method == "NoDisc")].sort_values("fold")
            if a.empty or b.empty:
                continue
            fam = []
            for c in mcols:
                # errors: lower is better -> flip so positive favors ADEP
                x, z = (a[c].values, b[c].values) if c != "errors" else (-a[c].values, -b[c].values)
                fam.append({"dataset": ds, "metric": "Minority " + c, **compare(x, z)})
            for r, h in zip(fam, holm([r["t_p"] for r in fam])):
                r["holm_p"] = h
            res += fam
            sup = f"{a.support.mean():.0f}"
            tex.append(f"{ds} & Full ADEP & {sup} & " + " & ".join(fmt(a[c].values) if c != "errors" else f"{a[c].mean():.1f}" for c in mcols) + r" \\")
            tex.append(f" & No discriminator & {sup} & " + " & ".join(fmt(b[c].values) if c != "errors" else f"{b[c].mean():.1f}" for c in mcols) + r" \\" + "\n\\hline")
        pd.DataFrame(res).to_csv(os.path.join(out, "minority_stats.csv"), index=False)
        open(os.path.join(out, "minority.tex"), "w").write("\n".join(tex) + "\n")

    print(f"\nDone. Files written to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_numbers")
    a = ap.parse_args()
    main(a.results, a.out)
