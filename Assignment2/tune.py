"""Tune kNN and decision tree settings."""
import preprocess as pp
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


OUT_DIR = Path("output/phase3")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

RANDOM_STATE = pp.RANDOM_STATE

N_SPLITS = 5  # Five folds keep rare classes in validation.

# Score missing predictions as zero
SCORING = {
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}
REFIT = "f1_macro"


# Grid search helpers

def make_cv(y, n_splits=N_SPLITS):
    """Build stratified folds for the data."""
    smallest = int(pd.Series(y).value_counts().min())  # Count of rarest class
    # num of folds cannot exceed rarest class count
    k = max(2, min(n_splits, smallest))
    if k < n_splits:  # Warn if we have to reduce folds due to rare classes
        print(f"  WARNING: rarest class has {smallest} instances, "
              f"using {k} folds")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)


def run_search(name, pipe, grid, X, y, n_jobs=1, train_scores=False):
    """Run a grid search and return results."""
    cv = make_cv(y)
    # Total num of fits = num param combos * num folds
    n_fits = int(np.prod([len(v) for v in grid.values()])) * cv.get_n_splits()
    print(f"\n[{name}] {n_fits} fits")

    search = GridSearchCV(pipe, grid, scoring=SCORING, refit=REFIT, cv=cv,
                          n_jobs=n_jobs, return_train_score=train_scores,
                          error_score="raise", verbose=1)
    t0 = time.perf_counter()
    search.fit(X, y)  # run grid search
    elap = time.perf_counter() - t0

    res = tidy(search, grid)
    best = res.iloc[0]  # get best row
    print(f"[{name}] {elap:.1f}s, best {REFIT} "
          f"{best['mean_f1_macro']:.4f} +/- {best['std_f1_macro']:.4f}")
    print(f"[{name}] {search.best_params_}")
    return search, res


def tidy(search, grid):
    """Keep useful grid search columns."""
    cv = pd.DataFrame(search.cv_results_)
    cols = {}
    for p in grid:  # for each param in grid, get the param values and map to labels
        cols[p.split("__")[-1]] = cv[f"param_{p}"].map(label)
    for m in SCORING:  # for each scoring metric, get the mean and std test scores
        cols[f"mean_{m}"] = cv[f"mean_test_{m}"]
        cols[f"std_{m}"] = cv[f"std_test_{m}"]
        if f"mean_train_{m}" in cv:
            cols[f"train_{m}"] = cv[f"mean_train_{m}"]
    cols["fit_seconds"] = cv["mean_fit_time"]
    return (pd.DataFrame(cols)
            .sort_values(f"mean_{REFIT}", ascending=False)
            .reset_index(drop=True))


def label(v):
    """Format parameter values for output."""
    if isinstance(v, pp.CorrelationFilter):
        return v.threshold
    if v is None or isinstance(v, (str, int, float, bool,
                                   np.integer, np.floating)):
        return v
    return type(v).__name__


def save(df, stem):
    df.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    print(f"  -> tables/{stem}.csv")


# kNN tuning

def knn_stage_a(X, y, numeric, nominal, n_jobs, quick):
    """Tune kNN preprocessing settings"""
    pipe = pp.build_knn_pipeline(
        numeric, nominal,
        KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=n_jobs))  # prepare pipeline with default kNN classifier

    scalers = [StandardScaler(), RobustScaler(),
               QuantileTransformer(output_distribution="normal",
                                   n_quantiles=1000,
                                   random_state=RANDOM_STATE)]  # set of scalers to try
    # Include no filtering as a baseline
    filters = ["passthrough", 0.99, 0.95, 0.90]
    if quick:
        scalers, filters = scalers[:2], ["passthrough", 0.95]

    grid = {"prep__num__scale": scalers,
            "prep__num__decorrelate": [f if f == "passthrough"
                                       else pp.CorrelationFilter(threshold=f)
                                       for f in filters]}
    # run grid search for stage A
    search, res = run_search("kNN stage A", pipe, grid, X, y)
    save(res, "knn_stage_a_representation")
    return search.best_params_, res, grid


def thr_of(f):
    return "none" if isinstance(f, str) else f.threshold


def knn_stage_b(X, y, numeric, nominal, rep, n_jobs, quick):
    """Tune kNN classifier settings."""
    pipe = pp.build_knn_pipeline(numeric, nominal,
                                 KNeighborsClassifier(n_jobs=n_jobs)) # prepare pipeline with kNN classifier
    pipe.set_params(**rep) # set the parameters of the pipeline to best parameters from stage A

    ks = [1, 3, 5, 7, 9, 11, 15, 21, 31] # set of k values to try
    metrics = ["euclidean", "manhattan"] 
    if quick:
        ks, metrics = [1, 5, 15], ["euclidean"]

    grid = {"clf__n_neighbors": ks,
            "clf__weights": ["uniform", "distance"],
            "clf__metric": metrics}
    search, res = run_search("kNN stage B", pipe, grid, X, y,
                             train_scores=True)
    save(res, "knn_stage_b_classifier")
    plot_k_curve(res)
    return search, res, grid


# Decision tree tuning

def tree_stage_a(X, y, numeric, nominal, n_jobs, quick):
    """Tune tree representation settings."""
    pipe = pp.build_tree_pipeline(
        numeric, nominal,
        DecisionTreeClassifier(random_state=RANDOM_STATE))

    encoders = [OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)]
    grid = {"prep__nom__encode": encoders,
            "clf__criterion": ["gini", "entropy"],
            # Include unweighted training.
            "clf__class_weight": [None, "balanced"]}
    if quick:
        grid["clf__criterion"] = ["gini"]

    search, res = run_search("tree stage A", pipe, grid, X, y, n_jobs=n_jobs)
    save(res, "tree_stage_a_representation")
    return search.best_params_, res, grid


def tree_stage_b(X, y, numeric, nominal, rep, n_jobs, quick):
    """Tune tree classifier settings."""
    pipe = pp.build_tree_pipeline(
        numeric, nominal,
        DecisionTreeClassifier(random_state=RANDOM_STATE))
    pipe.set_params(**rep)

    depths = [5, 10, 15, 20, 25, 30, None] # set of max_depth values to try
    leaves = [1, 2, 5, 10, 20] # set of min_samples_leaf values to try
    alphas = [0.0, 1e-5, 1e-4, 1e-3] # set of ccp_alpha values to try
    if quick:
        depths, leaves, alphas = [5, 15, None], [1, 10], [0.0, 1e-4]

    grid = {"clf__max_depth": depths,
            "clf__min_samples_leaf": leaves,
            "clf__ccp_alpha": alphas}
    search, res = run_search("tree stage B", pipe, grid, X, y, n_jobs=n_jobs,
                             train_scores=True) # run grid search for stage B
    save(res, "tree_stage_b_classifier")
    plot_tree_curves(res)
    return search, res, grid


# Figures

def plot_k_curve(res):
    """Plot macro F1 by neighbour count."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for (w, m), g in res.groupby(["weights", "metric"]):
        g = g.sort_values("n_neighbors")
        ax.errorbar(g["n_neighbors"], g["mean_f1_macro"],
                    yerr=g["std_f1_macro"], marker="o", capsize=3,
                    label=f"{w}, {m}")
    ax.set_xscale("log")
    ax.set_xticks(sorted(res["n_neighbors"].unique()))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("k")
    ax.set_ylabel("macro F1 (mean over folds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "knn_k_curve.png", dpi=200)
    plt.close(fig)
    print("  -> figures/knn_k_curve.png")


def plot_tree_curves(res):
    """Plot tree tuning results."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    d = res[(res["ccp_alpha"] == 0.0) & (res["min_samples_leaf"] == 1)].copy()
    d["depth_num"] = d["max_depth"].fillna(
        pd.to_numeric(d["max_depth"], errors="coerce").max() + 5)
    d = d.sort_values("depth_num")
    axes[0].plot(d["depth_num"], d["train_f1_macro"], marker="o",
                 label="training")
    axes[0].errorbar(d["depth_num"], d["mean_f1_macro"],
                     yerr=d["std_f1_macro"], marker="s", capsize=3,
                     label="validation")
    axes[0].set_xlabel("max_depth (rightmost point is unlimited)")
    axes[0].set_ylabel("macro F1")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    a = res[res["max_depth"].isna() & (res["min_samples_leaf"] == 1)].copy()
    a = a.sort_values("ccp_alpha")
    axes[1].errorbar(a["ccp_alpha"].replace(0.0, 1e-6), a["mean_f1_macro"],
                     yerr=a["std_f1_macro"], marker="s", capsize=3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("ccp_alpha (leftmost point is 0)")
    axes[1].set_ylabel("macro F1")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "tree_tuning_curves.png", dpi=200)
    plt.close(fig)
    print("  -> figures/tree_tuning_curves.png")


# Summary tables

def stage_rows(model, stage, res, grid):
    """Create report rows for one search stage."""
    top = res.iloc[0]
    rows = []
    for p, vals in grid.items():
        short = p.split("__")[-1]
        rows.append([model, stage, short,
                     ", ".join(show(label(v)) for v in vals),
                     show(top[short]),
                     round(top["mean_f1_macro"], 4),
                     round(top["std_f1_macro"], 4)])
    return rows


def show(v):
    """Format empty values for output."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "None"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def summarise(rows):
    """Create the tuning summary table."""
    df = pd.DataFrame(rows, columns=["model", "stage", "parameter",
                                     "values searched", "selected",
                                     "best mean macro F1", "std"])
    save(df, "tuning_summary")
    return df


# Rebuild selected models

SCALERS = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "QuantileTransformer": lambda: QuantileTransformer(
        output_distribution="normal", n_quantiles=1000,
        random_state=RANDOM_STATE),
}
ENCODERS = {
    "OrdinalEncoder": lambda: OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1),
    "OneHotEncoder": lambda: OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False),
}


def rebuild_knn(numeric, nominal, best, n_jobs=1):
    """Build kNN with selected settings"""
    b = best["knn"] 
    pipe = pp.build_knn_pipeline(
        numeric, nominal,
        KNeighborsClassifier(n_neighbors=b["clf__n_neighbors"],
                             weights=b["clf__weights"],
                             metric=b["clf__metric"], n_jobs=n_jobs))
    thr = best["knn_corr_threshold"] 
    pipe.set_params(
        prep__num__scale=SCALERS[b["prep__num__scale"]](),
        prep__num__decorrelate=("passthrough" if thr == "none"
                                else pp.CorrelationFilter(threshold=thr)))
    return pipe


def rebuild_tree(numeric, nominal, best):
    """Build the tree with selected settings."""
    b = best["tree"]
    pipe = pp.build_tree_pipeline(
        numeric, nominal,
        DecisionTreeClassifier(criterion=b["clf__criterion"],
                               class_weight=b["clf__class_weight"],
                               max_depth=b["clf__max_depth"],
                               min_samples_leaf=b["clf__min_samples_leaf"],
                               ccp_alpha=b["clf__ccp_alpha"],
                               random_state=RANDOM_STATE))
    pipe.set_params(prep__nom__encode=ENCODERS[b["prep__nom__encode"]]())
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(pp.DATA_PATH))
    ap.add_argument("--subsample", type=int, default=pp.SUBSAMPLE_N)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="shrunken grids, for smoke testing only")
    args = ap.parse_args()

    for d in (FIG_DIR, TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning")
    Xt, yt, Xe, ye, numeric, nominal, log = pp.prepare_pools(
        Path(args.data), args.subsample)
    print(f"\ntuning pool {Xt.shape}, evaluation pool {Xe.shape}")
    print(f"tuning class counts: {yt.value_counts().sort_index().to_dict()}")

    best = {}
    summary = []

    rep_knn, res_a, grid_a = knn_stage_a(Xt, yt, numeric, nominal,
                                         args.jobs, args.quick)
    sb, res_b, grid_b = knn_stage_b(Xt, yt, numeric, nominal, rep_knn,
                                    args.jobs, args.quick)
    best["knn"] = {**{k: label(v) for k, v in rep_knn.items()},
                   **sb.best_params_}
    best["knn_corr_threshold"] = thr_of(rep_knn["prep__num__decorrelate"])
    summary += stage_rows("kNN", "A representation", res_a, grid_a)
    summary += stage_rows("kNN", "B classifier", res_b, grid_b)

    rep_tree, res_ta, grid_ta = tree_stage_a(Xt, yt, numeric, nominal,
                                             args.jobs, args.quick) # tune tree representation settings
    st, res_tb, grid_tb = tree_stage_b(Xt, yt, numeric, nominal, rep_tree,
                                       args.jobs, args.quick) # tune tree classifier settings
    best["tree"] = {**{k: label(v) for k, v in rep_tree.items()},
                    **st.best_params_} # save best parameters for tree model
    summary += stage_rows("Tree", "A representation", res_ta, grid_ta)
    summary += stage_rows("Tree", "B classifier", res_tb, grid_tb)

    summarise(summary)

    best["cv"] = {"n_splits": N_SPLITS, "refit": REFIT,
                  "random_state": RANDOM_STATE,
                  "subsample_n": args.subsample}
    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump(best, f, indent=2, default=str)
    print(f"\nbest configuration -> {OUT_DIR}/best_params.json")
    print(json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main()
