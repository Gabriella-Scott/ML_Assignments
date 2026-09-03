"""Evaluate selected kNN and decision tree models."""
import tune
import preprocess as pp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


OUT_DIR = Path("output/phase4")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

EVAL_N = 227_673
N_SPLITS = 10
RANDOM_STATE = pp.RANDOM_STATE

NAMES = {0: "Normal", 1: "Reconnaissance", 2: "Backdoor", 3: "DoS",
         4: "Exploits", 5: "Analysis", 6: "Fuzzers", 7: "Worms",
         8: "Shellcode", 9: "Generic"}
LABELS = sorted(NAMES)


def scores(y_true, y_pred):
    """Calculate metrics for one fold."""
    kw = dict(labels=LABELS, zero_division=0)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", **kw),
        "precision_macro": precision_score(y_true, y_pred, average="macro", **kw),
        "recall_macro": recall_score(y_true, y_pred, average="macro", **kw),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", **kw),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


# Cross-validation

def evaluate(name, pipe, X, y, cv):
    per_fold, per_class, timing = [], [], [] # lists to store metrics for each fold
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int) # confusion matrix to accumulate results across folds

    # evaluate the model on each fold of the cross-validation
    for i, (tr, te) in enumerate(cv.split(X, y), start=1):
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr], y.iloc[tr]) # fit the model on train data
        t_fit = time.perf_counter() - t0 

        t0 = time.perf_counter() 
        pred = pipe.predict(X.iloc[te]) # predict on test data
        t_pred = time.perf_counter() - t0

        s = scores(y.iloc[te], pred) # calculate metrics for this fold.
        s["fold"] = i  # add fold number to metrics
        per_fold.append(s)
        timing.append({"fold": i, "fit_seconds": t_fit,
                       "predict_seconds": t_pred})

        # Calculate per-class metrics for this fold
        f1c = f1_score(y.iloc[te], pred, average=None, labels=LABELS,
                       zero_division=0)
        rec = recall_score(y.iloc[te], pred, average=None, labels=LABELS,
                           zero_division=0)
        per_class.append(pd.DataFrame({"fold": i, "cls": LABELS,
                                       "f1": f1c, "recall": rec}))

        cm += confusion_matrix(y.iloc[te], pred, labels=LABELS)
        print(f"  [{name}] fold {i}/{cv.get_n_splits()} "
              f"macro F1 {s['f1_macro']:.4f}  "
              f"fit {t_fit:.1f}s  predict {t_pred:.1f}s")

    return (pd.DataFrame(per_fold).assign(model=name),
            pd.concat(per_class).assign(model=name),
            pd.DataFrame(timing).assign(model=name),
            cm)


def summarise(folds):
    """Summarise metrics across folds."""
    metrics = [c for c in folds.columns if c not in ("fold", "model")]
    out = (folds.groupby("model")[metrics]
           .agg(["mean", "std"]).round(4))
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


# Performance ceiling

def ceiling(X, y):
    key = X.fillna("NA").astype(str).agg("|".join, axis=1)
    pairs = pd.DataFrame({"key": key, "y": y}
                         ).value_counts().reset_index(name="n")
    majority = pairs.sort_values("n", ascending=False).drop_duplicates("key")
    oracle = key.map(majority.set_index("key")["y"])
    return pd.Series(f1_score(y, oracle, average=None, labels=LABELS,
                              zero_division=0), index=LABELS)


# Paired comparison

def compare(folds, metric="f1_macro"):
    """Compare model scores by fold."""
    wide = folds.pivot(index="fold", columns="model", values=metric)
    a, b = wide.columns
    d = wide[a] - wide[b]
    k = len(d)

    # Correct for overlapping training folds.
    corr = 1 / k + 1 / (k - 1)
    t = d.mean() / np.sqrt(corr * d.var(ddof=1))
    p = 2 * stats.t.sf(abs(t), df=k - 1)

    t_naive, p_naive = stats.ttest_rel(wide[a], wide[b])
    return pd.DataFrame([{
        "metric": metric, "model_a": a, "model_b": b,
        "mean_a": round(wide[a].mean(), 4), "mean_b": round(wide[b].mean(), 4),
        "mean_difference": round(d.mean(), 4),
        "std_difference": round(d.std(ddof=1), 4),
        "folds_a_wins": int((d > 0).sum()),
        "t_corrected": round(t, 3), "p_corrected": round(p, 5),
        "t_uncorrected": round(t_naive, 3), "p_uncorrected": round(p_naive, 5),
    }])


# Figures

def plot_confusion(cms, stem="confusion_matrices"):
    fig, axes = plt.subplots(1, len(cms), figsize=(6.5 * len(cms), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, cms.items()):
        norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels([NAMES[c] for c in LABELS], rotation=45, ha="right",
                           fontsize=8)
        ax.set_yticklabels([NAMES[c] for c in LABELS], fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}, row-normalised")
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if norm[i, j] >= 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center",
                            va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"  -> figures/{stem}.png")


def plot_per_class(tbl):
    """Plot per-class F1 and ceiling scores."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(LABELS))
    models = sorted(tbl["model"].unique())
    w = 0.8 / (len(models) + 1)

    for i, m in enumerate(models):
        g = tbl[tbl["model"] == m].set_index("cls").loc[LABELS]
        ax.bar(x + i * w, g["f1_mean"], w, yerr=g["f1_std"], capsize=2,
               label=m)

    g = tbl[tbl["model"] == models[0]].set_index("cls").loc[LABELS]
    ax.bar(x + len(models) * w, g["ceiling_f1"], w, label="ceiling",
           color="0.7")

    ax.set_xticks(x + w)
    ax.set_xticklabels([NAMES[c] for c in LABELS], rotation=45, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_f1.png", dpi=200)
    plt.close(fig)
    print("  -> figures/per_class_f1.png")


def save(df, stem, index=False):
    df.to_csv(TAB_DIR / f"{stem}.csv", index=index)
    print(f"  -> tables/{stem}.csv")


# Script entry point

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(pp.DATA_PATH))
    ap.add_argument("--eval-n", type=int, default=EVAL_N)
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--best", default="output/phase3/best_params.json")
    args = ap.parse_args()

    for d in (FIG_DIR, TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)

    best = json.load(open(args.best))
    print("Loading and cleaning")
    Xt, yt, Xe, ye, numeric, nominal, _ = pp.prepare_pools(
        Path(args.data), best["cv"]["subsample_n"])

    # Sample the evaluation pool.
    X, y = pp.stratified_subsample(Xe, ye, args.eval_n)
    print(f"\nevaluation sample {X.shape}")
    print(f"class counts: {y.value_counts().sort_index().to_dict()}")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True,
                         random_state=RANDOM_STATE)

    models = {"kNN": tune.rebuild_knn(numeric, nominal, best, n_jobs=args.jobs),
              "Tree": tune.rebuild_tree(numeric, nominal, best)}

    folds, classes, times, cms = [], [], [], {}
    for name, pipe in models.items():
        print(f"\n{name}: {pipe.named_steps['clf']}")
        f, c, t, cm = evaluate(name, pipe, X, y, cv)
        folds.append(f)
        classes.append(c)
        times.append(t)
        cms[name] = cm

    folds = pd.concat(folds, ignore_index=True)
    classes = pd.concat(classes, ignore_index=True)
    times = pd.concat(times, ignore_index=True)

    save(folds, "per_fold_scores")
    save(summarise(folds), "summary")
    print()
    print(summarise(folds).to_string(index=False))

    print("\nComputing ceiling on the evaluation sample")
    ceil = ceiling(X, y)
    per_class = (classes.groupby(["model", "cls"])[["f1", "recall"]]
                 .agg(["mean", "std"]).round(4))
    per_class.columns = [f"{m}_{s}" for m, s in per_class.columns]
    per_class = per_class.reset_index()
    per_class["ceiling_f1"] = per_class["cls"].map(ceil).round(4)
    per_class["fraction_of_ceiling"] = (per_class["f1_mean"] /
                                        per_class["ceiling_f1"]).round(3)
    per_class["class"] = per_class["cls"].map(NAMES)
    save(per_class, "per_class_scores")
    print(per_class.to_string(index=False))

    cmp = compare(folds)
    save(cmp, "paired_comparison")
    print()
    print(cmp.to_string(index=False))

    cost = times.groupby("model")[
        ["fit_seconds", "predict_seconds"]].mean().round(3)
    save(cost.reset_index(), "runtime")
    print()
    print(cost.to_string())

    for name, cm in cms.items():
        save(pd.DataFrame(cm, index=[NAMES[c] for c in LABELS],
                          columns=[NAMES[c] for c in LABELS]),
             f"confusion_{name.lower()}", index=True)

    plot_confusion(cms)
    plot_per_class(per_class)


if __name__ == "__main__":
    main()
