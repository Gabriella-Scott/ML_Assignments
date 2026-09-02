"""Explore networkTraffic.csv."""
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Settings
DATA_PATH = Path("data/networkTraffic.csv")
FEATURES_PATH = Path("data/features.csv")
ATTACK_MAP_PATH = Path("data/attack_category_map.csv")

OUT_DIR = Path("output/phase1")
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

TARGET = "attack_cat"
ID_COL = "id"
NOMINAL = ["proto", "state", "service"]
MISSING_MARKER = "?"

EXPECTED_N_DESCRIPTIVE = 43

# Analysis thresholds
CORR_STRONG = 0.90
CORR_PERFECT = 0.999
OUTLIER_IQR_MULT = 1.5
ROW_MISSING_CUTOFF = 5

# Helpers


def numeric_descriptive(df):
    """Get numeric input columns."""
    # Exclude ID and target columns.
    return (df.select_dtypes(include=np.number)
              .drop(columns=[ID_COL, TARGET], errors="ignore"))


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# Loading


def load_data(path=DATA_PATH):
    """Read CSV and map '?' to NaN."""
    df = pd.read_csv(path, na_values=[MISSING_MARKER])
    assert df.columns[-1] == TARGET, f"Last column is not {TARGET}"
    n_desc = df.shape[1] - 1
    assert n_desc == EXPECTED_N_DESCRIPTIVE, \
        f"Expected {EXPECTED_N_DESCRIPTIVE} descriptive columns, got {n_desc}"
    return df


def load_metadata():
    """Load feature and class metadata."""
    feat_dict = pd.read_csv(FEATURES_PATH)
    # Match data column case.
    feat_dict["Name"] = feat_dict["Name"].str.strip(
    ).str.lower()  # strip whitespace and lowercase

    amap = pd.read_csv(ATTACK_MAP_PATH)
    attack_map = dict(zip(amap["Mapping"].astype(int),
                          amap["Attack Category Name"]))
    return feat_dict, attack_map


def verify_schema(df, feat_dict):
    """Compare data columns with metadata."""
    # ID and rate are absent from metadata.
    dat_cols = set(df.columns)
    dict_cols = set(feat_dict["Name"])
    return {
        "undocumented": sorted(dat_cols - dict_cols),
        "documented_but_absent": sorted(dict_cols - dat_cols),
    }

# Missing values


def missingness_by_feature(df):
    """Count missing values per column."""
    n = df.isna().sum()
    out = pd.DataFrame({
        "feature": n.index,
        "n_missing": n.values,
        "pct_missing": n.values / len(df) * 100,
        "dtype": df.dtypes.astype(str).values,
    })
    return (out[out["n_missing"] > 0]
            .sort_values("n_missing", ascending=False)
            .reset_index(drop=True))


def missingness_by_row(df, cutoff=ROW_MISSING_CUTOFF):
    """Count missing values per row."""
    row_na = df.isna().sum(axis=1)
    cnts = (row_na.value_counts().sort_index()
            .rename_axis("n_missing").reset_index(name="n_rows"))
    high_missing_idx = row_na[row_na > cutoff].index
    return cnts, high_missing_idx


def missingness_correlation(df):
    """Correlate missing-value flags."""
    mask = df.isna().astype(int)
    mask = mask.loc[:, mask.sum() > 0]
    return mask.corr()


def target_missingness(df):
    """Count rows with no target label."""
    idx = df.index[df[TARGET].isna()]
    return len(idx), idx


# Structure

def cardinality_report(df):
    """Count unique values per column."""
    # Keep missing counts for context.
    n_unique = df.nunique(dropna=True)
    n_missing = df.isna().sum()
    out = pd.DataFrame({
        "feature": n_unique.index,
        "n_unique_observed": n_unique.values,
        "n_missing": n_missing.reindex(n_unique.index).values,
        "unique_ratio": n_unique.values / len(df),
    })
    out["is_identifier"] = out["unique_ratio"] >= 1.0
    # Flag constant observed values.
    out["is_single_valued"] = out["n_unique_observed"] <= 1
    return out.sort_values("n_unique_observed", ascending=False).reset_index(drop=True)


def nominal_summary(df, nominal=NOMINAL):
    """Summarise nominal feature levels."""
    rows = []
    for col in nominal:
        vc = df[col].value_counts(dropna=False)
        rows.append({
            "feature": col,
            "n_levels_observed": df[col].nunique(dropna=True),
            "n_missing": int(df[col].isna().sum()),
            "top_level": vc.index[0],
            "top_level_pct": vc.iloc[0] / len(df) * 100,
            "levels": vc.to_dict(),
        })
    return pd.DataFrame(rows)


# Feature scale

def scale_profile(df):
    """Calculate numeric feature ranges"""
    num = numeric_descriptive(df)
    prof = num.describe().T  # transpose to have features as rows
    prof["range"] = prof["max"] - prof["min"]
    # Avoid log10(0)
    prof["log_range"] = np.where(prof["range"] > 0,
                                 np.log10(prof["range"].where(
                                     prof["range"] > 0)),
                                 np.nan)
    return prof.sort_values("log_range", ascending=False, na_position="last")


def plot_scale_disparity(profile, path=FIG_DIR / "scale_disparity.pdf"):
    """Plot feature ranges on a log scale."""
    p = profile.dropna(subset=["log_range"])
    fig, ax = plt.subplots(figsize=(6, max(4, len(p) * 0.18)))
    ax.barh(p.index, p["log_range"])
    ax.set_xlabel("log10 of feature range (max - min)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, path)


# Correlation

def correlation_matrix(df, method="pearson"):
    """Calculate numeric feature correlations."""
    return numeric_descriptive(df).corr(method=method)


def strong_pairs(corr, threshold=CORR_STRONG):
    """find strongly correlated feature pairs."""
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(
        bool))  # get upper triangle
    pairs = (upper.stack()
             .rename("correlation")
             .reset_index())  # convert upper triangle to long format
    pairs.columns = ["feature_a", "feature_b", "correlation"]
    # add absolute correlation column
    pairs["abs_corr"] = pairs["correlation"].abs()
    # filter for strong correlations
    pairs = pairs[pairs["abs_corr"] >= threshold]
    # flag near-perfect correlations
    pairs["is_perfect"] = pairs["abs_corr"] >= CORR_PERFECT
    return (pairs.sort_values("abs_corr", ascending=False)
            .drop(columns="abs_corr").reset_index(drop=True))


def plot_correlation_heatmap(corr, path=FIG_DIR / "correlation_heatmap.pdf"):
    """Plot the correlation matrix."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr.index, fontsize=5)
    save_fig(fig, path)


# Outliers

def outlier_report(df, mult=OUTLIER_IQR_MULT):
    """Find values outside Tukey fences."""
    num = numeric_descriptive(df)
    rows = []
    for col in num.columns:
        s = num[col].dropna()  # drop missing values for col
        q1, q3 = s.quantile(0.25), s.quantile(0.75)  # calc Q1 and Q3
        iqr = q3 - q1
        # Zero IQR -> fences unusable
        degenerate = iqr == 0  # flag degenerate case
        n_low = int((s < q1 - mult * iqr).sum())
        n_high = int((s > q3 + mult * iqr).sum())
        rows.append({
            "feature": col, "q1": q1, "q3": q3, "iqr": iqr,
            "n_low": n_low, "n_high": n_high, "n_total": n_low + n_high,
            "pct_high": n_high / len(s) * 100 if len(s) else np.nan,
            "skewness": float(s.skew()),
            "iqr_degenerate": degenerate,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["iqr_degenerate", "n_high"],
                           ascending=[True, False]).reset_index(drop=True)


def pick_outlier_examples(out_rep, n=6):
    """Choose features for outlier plots"""
    return out_rep[~out_rep["iqr_degenerate"]]["feature"].head(n).tolist()


def plot_outlier_examples(df, features, path=FIG_DIR / "outlier_boxplots.pdf"):
    """Plot selected feature boxplots."""
    if not features:
        return
    fig, axes = plt.subplots(
        1, len(features), figsize=(2.2 * len(features), 4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, features):
        ax.boxplot(df[col].dropna(), sym=".")
        ax.set_yscale("symlog")
        ax.set_title(col, fontsize=8)
        ax.set_xticks([])
    save_fig(fig, path)


# Target

def class_distribution(df, label_map):
    """Count rows per attack class."""
    cnts = (df[TARGET].dropna().astype(int)
            .value_counts()
            .rename_axis("code").reset_index(name="count")
            .sort_values("code"))  # sort by code for consistency
    cnts["class"] = cnts["code"].map(label_map)
    assert cnts["class"].notna().all(), "unmapped target codes present"
    cnts["pct"] = cnts["count"] / cnts["count"].sum() * 100
    # calculate imbalance ratio
    imbalance = cnts["count"].max() / cnts["count"].min()
    return cnts[["code", "class", "count", "pct"]], imbalance


def plot_class_distribution(dist, path=FIG_DIR / "class_distribution.pdf"):
    """Plot attack class counts."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(dist["class"], dist["count"])
    ax.set_xlabel("Attack category")
    ax.set_ylabel("Number of instances")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    if dist["count"].max() / dist["count"].min() > 10:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, path)


# Output

def save_table(frame, name):
    """Save a table as CSV."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Load data and metadata
    df = load_data()
    feat_dict, attack_map = load_metadata()
    schema = verify_schema(df, feat_dict)
    # save tables and plots
    miss_feat = missingness_by_feature(df)
    save_table(miss_feat, "missingness_by_feature")
    # Save missingness by row and get high-missing rows for reporting.
    miss_rows, high_missing_idx = missingness_by_row(df)
    save_table(miss_rows, "missingness_by_row")
    save_table(missingness_correlation(
        df).reset_index(), "missingness_correlation")
    n_target_missing, _ = target_missingness(df)
    # save cardinality, nominal summary, and scale profile
    card = cardinality_report(df)
    save_table(card, "cardinality_report")
    nom = nominal_summary(df)
    save_table(nom.drop(columns=["levels"]), "nominal_summary")
    profile = scale_profile(df)
    save_table(profile.reset_index(names="feature"), "scale_profile")
    plot_scale_disparity(profile)
    # save correlation matrix and strong pairs
    corr = correlation_matrix(df)
    save_table(corr.reset_index(names="feature"), "correlation_matrix")
    plot_correlation_heatmap(corr)
    pairs = strong_pairs(corr)
    save_table(pairs, "strong_pairs")

    # Also check rank correlation.
    sp_pairs = strong_pairs(correlation_matrix(df, method="spearman"))
    save_table(sp_pairs, "strong_pairs_spearman")
    out_rep = outlier_report(df)
    save_table(out_rep, "outlier_report")
    plot_outlier_examples(df, pick_outlier_examples(out_rep))

    dist, imbalance = class_distribution(df, attack_map)
    save_table(dist, "class_distribution")
    plot_class_distribution(dist)

    print(f"Rows: {len(df):,}  Cols: {df.shape[1]}")
    print(f"Undocumented columns: {schema['undocumented']}")
    print(f"Features with missing values: {len(miss_feat)}")
    print(f"Rows with >{ROW_MISSING_CUTOFF} missing: {len(high_missing_idx)}")
    print(f"Target missing: {n_target_missing}")
    print(
        f"Identifiers: {card.loc[card['is_identifier'], 'feature'].tolist()}")
    sv = card.loc[card["is_single_valued"], ["feature", "n_missing"]]
    print(f"Single-valued: {list(sv.itertuples(index=False, name=None))}")
    print(f"Pearson pairs >= {CORR_STRONG}: {len(pairs)} "
          f"({int(pairs['is_perfect'].sum())} near-perfect)")
    print(f"Spearman pairs >= {CORR_STRONG}: {len(sp_pairs)}")
    print(f"Classes observed: {len(dist)}  imbalance ratio: {imbalance:.1f}x")


if __name__ == "__main__":
    main()
