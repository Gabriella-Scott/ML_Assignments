"""Build model-specific preprocessing pipelines."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)

DATA_PATH = Path("data/networkTraffic.csv")
OUT_DIR = Path("output/phase2")

TARGET = "attack_cat"
ID_COL = "id"
NOMINAL = ["proto", "state", "service"]
MISSING_MARKER = "?"

# Sparse-row limit from phase 1.
ROW_MISSING_CUTOFF = 1

# Correlation filter settings.
CORR_DROP_THRESHOLD = 0.95
CORR_GRID = [0.90, 0.95, 0.99]

SUBSAMPLE_N = 30_000  # None uses all rows
RANDOM_STATE = 42


# Shared cleaning

def load_raw(path=DATA_PATH):
    """Read CSV and map '?' to missing."""
    df = pd.read_csv(path, na_values=[MISSING_MARKER])
    assert df.columns[-1] == TARGET, f"last column is not {TARGET}"
    return df


def clean_common(df, row_cutoff=ROW_MISSING_CUTOFF, verbose=True):
    """apply shared cleaning steps."""
    log = {"n_start": len(df), "n_cols_start": df.shape[1]}
    df = df.copy()

    # Rows without labels.
    log["dropped_missing_target"] = int(df[TARGET].isna().sum())
    df = df[df[TARGET].notna()]

    # Identifier column.
    log["dropped_id"] = ID_COL in df.columns
    df = df.drop(columns=[ID_COL], errors="ignore")

    # Rows with many missing features.
    row_na = df.drop(columns=[TARGET]).isna().sum(axis=1)
    log["row_missing_cutoff"] = row_cutoff
    log["max_row_missing"] = int(row_na.max())
    log["dropped_sparse_rows"] = int((row_na > row_cutoff).sum())
    df = df[row_na <= row_cutoff]

    # Constant features
    const = [c for c in df.columns
             if c != TARGET and df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=const)
    log["dropped_single_valued"] = const

    df[TARGET] = df[TARGET].astype(int)
    log["n_end"], log["n_cols_end"] = len(df), df.shape[1]

    if verbose:
        for k, v in log.items():
            print(f"  {k}: {v}")
    return df, log


def split_X_y(df):
    return df.drop(columns=[TARGET]), df[TARGET]


def feature_groups(X):
    """Split columns into numeric and nominal groups"""
    nominal = [c for c in NOMINAL if c in X.columns]
    numeric = [c for c in X.columns if c not in nominal]
    return numeric, nominal


def stratified_subsample(X, y, n, seed=RANDOM_STATE):
    """Take a stratified sample."""
    # Same sample for both models
    if n is None or n >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True)
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y,
                                    random_state=seed)
    return Xs.reset_index(drop=True), ys.reset_index(drop=True)


def tuning_split(X, y, n=SUBSAMPLE_N, seed=RANDOM_STATE):
    """Create stratified tuning and evaluation pools."""
    if n is None or n >= len(X):
        # Small development data
        print("  WARNING: pools overlap, dataset smaller than subsample size")
        Xr, yr = X.reset_index(drop=True), y.reset_index(drop=True)
        return Xr, yr, Xr, yr

    Xt, Xe, yt, ye = train_test_split(X, y, train_size=n, stratify=y,
                                      random_state=seed)
    return (Xt.reset_index(drop=True), yt.reset_index(drop=True),
            Xe.reset_index(drop=True), ye.reset_index(drop=True))


def prepare_pools(path=DATA_PATH, subsample_n=SUBSAMPLE_N, verbose=True):
    """Load, clean, and split data."""
    df, log = clean_common(load_raw(path), verbose=verbose)
    X, y = split_X_y(df)  # Split features and target
    # Split into tuning and evaluation pools
    Xt, yt, Xe, ye = tuning_split(X, y, subsample_n)
    numeric, nominal = feature_groups(Xt)
    log["n_tune"], log["n_eval"] = len(Xt), len(Xe)
    log["n_numeric"], log["n_nominal"] = len(numeric), len(nominal)
    return Xt, yt, Xe, ye, numeric, nominal, log


# knn pipeline

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from highly correlated pairs."""

    def __init__(self, threshold=CORR_DROP_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Fit using training data only.
        A = np.asarray(X, dtype=float)
        n = A.shape[1]
        self.n_features_in_ = n

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.nan_to_num(
                np.atleast_2d(np.corrcoef(A, rowvar=False))))

        # Average absolute correlation.
        redundancy = (corr.sum(axis=1) - np.diag(corr)) / max(n - 1, 1)

        # Strongest pairs first
        i, j = np.triu_indices(n, k=1)  # Upper triangle indices
        over = corr[i, j] >= self.threshold  # pairs exceeding threshold
        # sort by correlation value
        pairs = sorted(zip(corr[i, j][over], i[over], j[over]), reverse=True)

        keep = np.ones(n, dtype=bool)
        for _, a, b in pairs:
            if not (keep[a] and keep[b]):
                continue
            # drop the more redundant feature
            keep[b if redundancy[b] >= redundancy[a] else a] = False

        self.keep_ = keep
        self.dropped_idx_ = np.flatnonzero(~keep)
        return self

    def transform(self, X):
        # Return only the kept features.
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        # Return the names of the kept features
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features)[self.keep_]


def build_knn_preprocessor(numeric_cols, nominal_cols, scaler=None,
                           corr_threshold=CORR_DROP_THRESHOLD):
    """Build kNN preprocessing."""
    # Scaler tuned in phase 3
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("decorrelate", CorrelationFilter(threshold=corr_threshold)),
        ("scale", StandardScaler() if scaler is None else scaler),
    ])

    nominal = Pipeline([
        # keep missing values as a level.
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([("num", numeric, numeric_cols),
                              ("nom", nominal, nominal_cols)],
                             remainder="drop")


def build_knn_pipeline(numeric_cols, nominal_cols, classifier, **kw):
    """Combine kNN preprocessing and classifier."""
    return Pipeline([("prep", build_knn_preprocessor(numeric_cols, nominal_cols, **kw)),
                     ("clf", classifier)])


# Decision tree pipeline

def build_tree_preprocessor(numeric_cols, nominal_cols, nominal_encoding="ordinal"):
    """Encode tree input values."""
    numeric = Pipeline([
        # Required by scikit-learn.
        ("impute", SimpleImputer(strategy="median")),
    ])

    encoder = (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
               if nominal_encoding == "ordinal"
               else OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", encoder),
    ])

    return ColumnTransformer([("num", numeric, numeric_cols),
                              ("nom", nominal, nominal_cols)],
                             remainder="drop")


def build_tree_pipeline(numeric_cols, nominal_cols, classifier, **kw):
    return Pipeline([("prep", build_tree_preprocessor(numeric_cols, nominal_cols, **kw)),
                     ("clf", classifier)])


# Report decision data

DECISIONS = [
    dict(
        issue="Missing target values",
        knn_action="Drop instance",
        knn_reason="A label cannot be imputed without fabricating supervision, "
                   "and an unlabelled instance cannot cast a vote.",
        tree_action="Drop instance",
        tree_reason="Impurity is computed over labels, so an unlabelled "
                    "instance contributes nothing to any split.",
        ref="topic3, topic4",
    ),
    dict(
        issue="Unique identifier (id)",
        knn_action="Drop",
        knn_reason="Every pair of instances is maximally separated on this "
                   "axis, so it adds an almost constant term to every distance "
                   "and dilutes the informative features.",
        tree_action="Drop",
        tree_reason="Information gain is biased toward features with many "
                    "distinct values, and a unique feature is the limiting "
                    "case: a split on it isolates single instances, reaching "
                    "zero training error with no generalisation.",
        ref="topic4",
    ),
    dict(
        issue="Single-valued features",
        knn_action="Drop",
        knn_reason="Contributes zero to every pairwise distance, so removal "
                   "cannot change any neighbourhood.",
        tree_action="Drop",
        tree_reason="Zero information gain at every node, so it is never "
                    "selected as a split.",
        ref="topic3, topic4",
    ),
    dict(
        issue="Instances with many missing features",
        knn_action="No instances removed",
        knn_reason="Topic 3 slide 17 notes the weighted-distance correction "
                   "only works when few values are absent, so a cutoff was "
                   "evaluated. No instance in this dataset has more than one "
                   "missing value, so the rule removes nothing and was not "
                   "applied. The issue named in the specification was present "
                   "in the assignment 1 data, not this version.",
        tree_action="No instances removed",
        tree_reason="Same evidence, and the same decision, so both models see "
                    "an identical instance set.",
        ref="topic3",
    ),
    dict(
        issue="Feature ranges spanning many orders of magnitude",
        knn_action="Quantile transformation (selected by tuning)",
        knn_reason="In an unscaled distance sum the widest feature dominates "
                   "the total by many orders of magnitude, so the distance is "
                   "effectively computed on one feature and the rest are noise. "
                   "Topic 3 slide 24 permits z-score normalisation only when "
                   "the original distribution is normal, and these features are "
                   "heavily right-skewed, so standard, robust and quantile "
                   "scaling were compared during tuning. Quantile won by about "
                   "three standard deviations over standard scaling; quantile "
                   "against robust sits inside the noise, so quantile is "
                   "reported as selected rather than as better.",
        tree_action="No transformation",
        tree_reason="A split is a threshold test on a single feature, giving "
                    "an axis-parallel partition. Any strictly monotone "
                    "rescaling maps each threshold onto an equivalent one and "
                    "leaves the partition, and therefore the tree, unchanged.",
        ref="topic3, wickramarachchi2016",
    ),
    dict(
        issue="Missing numeric values",
        knn_action="Median imputation",
        knn_reason="The features are heavily right-skewed, so the mean sits "
                   "away from the bulk of the data and an imputed instance "
                   "would be placed in a sparse region with distorted "
                   "neighbours. The median stays inside the bulk.",
        tree_action="Median imputation",
        tree_reason="Topic 4 slide 35 notes tree induction can handle missing "
                    "values natively through surrogate splits, but the "
                    "implementation used does not provide these. The "
                    "transformation serves the library, not the algorithm.",
        ref="topic4, sklearn_tree",
    ),
    dict(
        issue="Missing nominal values (service)",
        knn_action="Encode as its own level",
        knn_reason="In UNSW-NB15 a blank service records that no service was "
                   "predominant, so it is an unobserved category rather than a "
                   "lost value. Mode imputation would invent traffic that was "
                   "never observed. This affects 54.8 percent of instances, so "
                   "the choice is consequential rather than cosmetic.",
        tree_action="Encode as its own level",
        tree_reason="Same reading of the semantics, and it lets the tree split "
                    "on absence directly where that is informative.",
        ref="moustafa2015",
    ),
    dict(
        issue="Nominal encoding",
        knn_action="One-hot",
        knn_reason="Protocol names have no order, so ordinal codes would make "
                   "the distance between two protocols depend on "
                   "alphabetisation. One-hot gives every distinct pair of "
                   "levels an equal distance, matching the Overlap measure of "
                   "Topic 3 slide 13. Cost: dimensionality grows and the "
                   "nominal features are implicitly down-weighted relative to "
                   "scaled numeric ones.",
        tree_action="One-hot (selected by tuning, reversing the expectation)",
        tree_reason="Expected ordinal to win, because one-hot turns one k-level "
                    "feature into k binary features and fragments the "
                    "available impurity reduction. Tuning reversed this: "
                    "one-hot 0.514 against ordinal 0.499. Wang and Witten give "
                    "the mechanism: the best split for a k-valued enumerated "
                    "attribute is one of the k-1 positions obtained by ordering "
                    "the levels by average class value, whereas OrdinalEncoder "
                    "orders alphabetically. With 133 protocol levels the "
                    "arbitrary ordering costs more than the fragmentation.",
        ref="topic3, rodden2000, wang1997",
    ),
    dict(
        issue="Strongly correlated features",
        knn_action="No filter applied (hypothesis tested and rejected)",
        knn_reason="Expected a filter to help. Duplicated information enters "
                   "the distance sum "
                   "twice, doubling the implicit weight of that quantity "
                   "relative to every other feature. That weighting was never "
                   "chosen, so it is an artefact. Eighteen pairs exceed 0.90, "
                   "with is_ftp_login/ct_ftp_cmd at 0.999, dbytes/dloss at "
                   "0.997 and sbytes/sloss at 0.996. Of each pair the feature "
                   "with the higher mean absolute correlation to the remaining "
                   "features is dropped, so the retained one carries the more "
                   "distinct information. Kelleher et al. pp. 264-266 warn that "
                   "a filter judges each feature in isolation, so the threshold "
                   "was tuned as a wrapper instead of asserted. The wrapper "
                   "rejected it: macro F1 fell monotonically from 0.4465 with "
                   "no filter to 0.4437 at 0.99, 0.4376 at 0.95 and 0.4373 at "
                   "0.90, under every scaler. The filter removes 3 numeric "
                   "features at 0.99, 10 at 0.95 and 11 at 0.90, out of a 192 "
                   "column tuning space. Those features carry more information "
                   "than the duplicate weighting artefact costs, so no filter "
                   "is applied.",
        tree_action="Keep",
        tree_reason="The tree selects one best split per node, so a duplicate "
                    "is an alternative never needed twice and the same "
                    "partition remains available. Caveat: redundancy divides "
                    "impurity-based feature importances between the pair, so "
                    "importances must be read with that in mind.",
        ref="topic3, topic4, kelleher2015",
    ),
    dict(
        issue="Large positive outliers",
        knn_action="No removal, controlled through the scaler choice",
        knn_reason="Removing them would discard genuine attack traffic, which "
                   "is the signal being classified. Topic 3 notes majority "
                   "voting limits their influence at moderate k. Their effect "
                   "on standardisation is handled by comparing StandardScaler "
                   "against rank and quantile based alternatives during tuning.",
        tree_action="No treatment",
        tree_reason="Topic 4 slide 34: an outlier is isolated in its own leaf, "
                    "that leaf is pruned, and the instance is absorbed into a "
                    "parent where it sits in the minority. This holds only "
                    "under pruning, so it is tied to the ccp_alpha tuning.",
        ref="topic3, topic4",
    ),
    dict(
        issue="Skew class distribution (534:1)",
        knn_action="Distance weighting plus macro-averaged metrics",
        knn_reason="KNeighborsClassifier exposes no class_weight, so distance "
                   "weighting is used to stop a distant majority-class "
                   "neighbour outvoting a close minority-class one. Macro "
                   "averaging then gives every class equal weight in the "
                   "reported score, which is the appropriate choice when "
                   "minority-class performance matters.",
        tree_action="class_weight='balanced' plus macro-averaged metrics",
        tree_reason="Topic 4 slide 34 predicts minority leaves are pruned and "
                    "their instances absorbed into the majority class. "
                    "Reweighting the impurity calculation makes minority "
                    "classes resist that. Checked against per-class recall.",
        ref="topic4, kelleher2015",
    ),
    dict(
        issue="Dataset size (257,673 instances)",
        knn_action="Stratified subsample",
        knn_reason="kNN defers all computation to query time, so a grid search "
                   "over k, metric, weighting and scaler is intractable at full "
                   "size. A class-proportional subsample preserves the class "
                   "ratios that the evaluation depends on.",
        tree_action="Identical stratified subsample",
        tree_reason="The tree does not need the reduction, but applying the "
                    "same sample to both models keeps the comparison a "
                    "comparison of algorithms rather than of sample sizes.",
        ref="kelleher2015",
    ),
]


def build_decision_table():
    """Build the decision table."""
    return pd.DataFrame(DECISIONS, columns=["issue", "knn_action", "knn_reason",
                                            "tree_action", "tree_reason", "ref"])


def save_outputs(out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    build_decision_table().to_csv(out_dir / "preprocessing_decisions.csv",
                                  index=False)


# Script entry point

def prepare(path=DATA_PATH, subsample_n=SUBSAMPLE_N, verbose=True):
    """Load, clean, and sample data."""
    df, log = clean_common(load_raw(path), verbose=verbose)
    X, y = split_X_y(df)
    X, y = stratified_subsample(X, y, subsample_n)
    numeric, nominal = feature_groups(X)
    log["n_numeric"], log["n_nominal"] = len(numeric), len(nominal)
    return X, y, numeric, nominal, log


if __name__ == "__main__":
    X, y, numeric, nominal, log = prepare()
    print(f"\nX: {X.shape}  numeric: {len(numeric)}  nominal: {nominal}")
    print(f"classes: {sorted(y.unique())}")
    # report the effect of correlation filtering on the kNN feature space
    for thr in CORR_GRID:
        prep = build_knn_preprocessor(numeric, nominal, corr_threshold=thr)
        cols = prep.fit_transform(X, y).shape[1]
        dropped = prep.named_transformers_[
            "num"].named_steps["decorrelate"].dropped_idx_
        print(f"corr {thr}: kNN space {cols} cols, dropped "
              f"{[numeric[i] for i in dropped]}")

    tree_cols = build_tree_preprocessor(
        numeric, nominal).fit_transform(X, y).shape[1]
    print(f"tree feature space: {tree_cols} columns")

    save_outputs()
    print(f"decisions -> {OUT_DIR}/preprocessing_decisions.csv")
