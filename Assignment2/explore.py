"""
Data exploration for networkTraffic.csv
"""
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# config
DATA_PATH = Path("data/networkTraffic.csv")
FEATURES_PATH = Path("data/features.csv")
ATTACK_MAP_PATH = Path("data/attack_category_map.csv")

OUT_DIR = Path("output/phase1")
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

TARGET = "attack_cat"
ID_COL = "id"
NOMINAL = ["proto", "state", "service"]  # per spec
MISSING_MARKER = "?"

EXPECTED_N_DESCRIPTIVE = 43  # spec, section 3.1

# thresholds, tune once you see full data
CORR_STRONG = 0.90
CORR_PERFECT = 0.999
OUTLIER_IQR_MULT = 1.5

# loading


def load_data(path=DATA_PATH):
    """Read the CSV, mapping '?' to NaN so pandas infers numeric dtypes."""
    df = pd.read_csv(path, na_values=[MISSING_MARKER])
    assert df.columns[-1] == TARGET, f"Last column is not {TARGET}"
    assert df.shape[1] - \
        1 == EXPECTED_N_DESCRIPTIVE, f"Expected {EXPECTED_N_DESCRIPTIVE} descriptive columns, got {df.shape[1] - 1}"
    # note: id counts as one of the 43

    return df


def load_metadata():
    """Load the feature dictionary and the attack category label map."""
    feat_dict = pd.read_csv(FEATURES_PATH)
    # features.csv uses Sload/Dload/Spkts/Dpkts/Sjit/Djit; data uses lowercase
    feat_dict["Name"] = feat_dict["Name"].str.strip().str.lower()

    attack_map_df = pd.read_csv(ATTACK_MAP_PATH)
    attack_map = dict(
        zip(attack_map_df["Mapping"], attack_map_df["Attack Category Name"]))
    return feat_dict, attack_map


def verify_schema(df, feat_dict):
    """Cross-check dataset columns against the supplied feature dictionary."""
    # features.csv documents 41 descriptive + target
    # data has 43 descriptive, so 'id' and 'rate' are undocumented
    data_cols = set(df.columns)
    dict_cols = set(feat_dict["Name"])
    undocumented = data_cols - dict_cols
    missing_in_data = dict_cols - data_cols

    print(f"Columns in data but not in feature dictionary: {undocumented}")
    print(f"Columns in feature dictionary but not in data: {missing_in_data}")

    # worth a sentence in the report: 'rate' is undocumented, so its meaning
    # must be inferred rather than cited

    raise NotImplementedError

# missingness


def missingness_by_feature(df):
    """Count and percentage of NaN per column, descending."""
    n_missing = df.isna().sum()
    result = pd.DataFrame({
        "feature": n_missing.index,
        "n_missing": n_missing.values,
        "pct_missing": n_missing.values / len(df) * 100,
        "dtype": df.dtypes.values,
    })
    # expect 'service' to dominate, it is missing in the original dataset
    return result[result["n_missing"] > 0].sort_values("n_missing", ascending=False).reset_index(drop=True)
 
 
def missingness_by_row(df):
    """Distribution of missing counts per instance."""
    # spec hints some instances have too many missing features
    # TODO: row_na = df.isna().sum(axis=1), return value_counts
    # TODO: flag rows above some cutoff, these are drop candidates in phase 2
    raise NotImplementedError
 
 
def missingness_correlation(df):
    """Correlate the missingness indicator matrix."""
    # spec says missing values are weakly correlated to one another
    # weak correlation supports treating missingness as MCAR-ish, which in turn
    # justifies simple imputation instead of a model-based scheme
    # TODO: mask = df.isna().astype(int), drop all-zero columns, corr()
    raise NotImplementedError
 
 
def target_missingness(df):
    """Count instances with a missing target."""
    # these cannot be imputed, a guessed label is fabricated supervision
    # TODO: return count and index of rows where TARGET is NaN
    raise NotImplementedError

# Structure

def cardinality_report(df):
    """Unique-value counts, to expose constants and identifiers."""
    # TODO: per column record n_unique, and n_unique / n_rows
    # ratio == 1.0 flags an identifier, n_unique == 1 flags a constant
    # constants carry zero information gain and zero distance contribution
    raise NotImplementedError
 
 
def nominal_summary(df, nominal=NOMINAL):
    """Level counts and frequencies for the nominal features."""
    # cardinality here drives the encoding decision in phase 2
    # high cardinality punishes one-hot encoding for trees
    # TODO: for each, value_counts(dropna=False) and n_levels
    raise NotImplementedError

# scale
def scale_profile(df):
    """Per-feature range, spread and order of magnitude."""
    # the point is the spread across features, not within one
    # kNN distance is dominated by whichever feature has the widest range
    # TODO: numeric describe().T plus a log10(max - min) column
    # TODO: sort by that column so the disparity is visible in one glance
    raise NotImplementedError
 
 
def plot_scale_disparity(profile, path=FIG_DIR / "scale_disparity.pdf"):
    """Horizontal bar chart of feature ranges on a log axis."""
    # TODO: barh of log10 range per feature
    # this figure is the evidence for 'scaling is required for kNN'
    raise NotImplementedError
 
# redundancy
def correlation_matrix(df):
    """Pearson correlation over numeric descriptive features."""
    # TODO: drop ID_COL and TARGET, then corr()
    # consider Spearman as well, many features are heavily skewed and Pearson
    # understates monotone but non-linear relationships
    raise NotImplementedError
 
 
def strong_pairs(corr, threshold=CORR_STRONG):
    """Extract upper-triangle pairs above a correlation threshold."""
    # TODO: mask with np.triu, stack, filter on abs value, sort descending
    # expect sbytes/sloss, dbytes/dloss, is_ftp_login/ct_ftp_cmd near 1.0
    # perfect pairs are pure redundancy, one of each pair carries no new info
    raise NotImplementedError
 
 
def plot_correlation_heatmap(corr, path=FIG_DIR / "correlation_heatmap.pdf"):
    """Heatmap of the correlation matrix."""
    # TODO: imshow with a diverging colormap, vmin=-1, vmax=1
    # keep tick labels small, there are ~40 features
    raise NotImplementedError
 
# outliers
