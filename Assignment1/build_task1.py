"""
Builds DQR tables and histogram/bar-plot PDFs.
"""

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

# config
INPUT_FILE = "badDataSet.csv"
SEP = ","  # comma delimited
NA_TOKEN = "?"  # missing vals

CAT_CARD_THRESH = 10  # Continuous vs Categorical decision

FORCE_CONT = []
FORCE_CAT = []

USE_TOTAL_FOR_PCT = True


# Stage 1: Load file and label columns

# pd.read_csv reads the file into a DataFrame (a table).
#   sep=SEP              -> split each line on tabs
#   header=None          -> the file has NO header row, so treat row 0 as data
#   na_values=[NA_TOKEN] -> turn every '?' into NaN (a proper missing value)
#   keep_default_na=True -> also treat blanks/"NaN" text as missing, as usual

raw = pd.read_csv(INPUT_FILE, sep=SEP, header=None, na_values=[
                  NA_TOKEN], keep_default_na=True, skipinitialspace=False,)

# raw.shape -> (rows, cols). Grab col count to build names
n_cols = raw.shape[1]

# label columns 1..(second last) as A1..A(n-1), and the LAST as T.
# range(1, n_cols) yields 1,2,...,n_cols-1, so we get A1..A46 for 47 columns,
# then we append "T" for the final (target) column.
col_names = [f"A{i}" for i in range(1, n_cols)] + ["T"]
raw.columns = col_names

# N -> total num of instances (rows); used everywhere for percentages
N = len(raw)
print(f"Loaded {N} rows x {n_cols}  columns")
print(f"Descriptive features: A1..A{n_cols-1}   Target: T\n")

# Split the names into "descriptive features"
# and the single target column.
feature_cols = col_names[:-1]   # A1 .. A46
target_col = "T"

# Stage 2: Classify each feature as continuous or categorical


def is_numeric_col(s):
    """
    Returns True if the column s is numeric.
    """
    coerced = pd.to_numeric(s, errors="coerce")
    became_bad = coerced.isna() & s.notna()   # was a value before, is NaN now
    return became_bad.sum() == 0


def classify(col):
    s = raw[col]
    if col in FORCE_CONT:
        return "continuous"
    if col in FORCE_CAT:
        return "categorical"
    if not is_numeric_col(s):
        return "categorical"                       # has text -> categorical
    # count of distinct non-missing values
    card = s.dropna().nunique()
    return "categorical" if card <= CAT_CARD_THRESH else "continuous"


# build a dict then split feature names into 2 lists based on the decision
types = {c: classify(c) for c in feature_cols}
continuous_feats = [c for c in feature_cols if types[c] == "continuous"]
categorical_feats = [c for c in feature_cols if types[c] == "categorical"]

# Stage 2b: Print a feature summary

summary = []
for c in feature_cols:
    s = raw[c]
    non_null = s.dropna()
    summary.append({
        "Feature": c,
        "AssignedType": types[c],
        "RawDtype": str(s.dtype),
        "Numeric?": is_numeric_col(s),
        "Card": non_null.nunique(),
        "%Miss": round(100 * s.isna().sum() / N, 2),
        "Sample": ", ".join(map(str, non_null.unique()[:5])),
    })
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
print()

# Stage 3: Compute the dqr for percentages


def pct_base(s):
    """
    The denom used for percentage.
    """
    return N if USE_TOTAL_FOR_PCT else s.notna().sum()


def continuous_row(col):
    """
    Build one row of the continuous DQR for a single feature.
    """
    s = pd.to_numeric(raw[col], errors="coerce")  # ensure numeric
    v = s.dropna()  # non-missing values only
    return {
        "Feature": col,
        "Count": N,  # total rows
        "% Miss.": round(100 * s.isna().sum() / N, 2),  # percent missing
        "Card.": v.nunique(),  # distinct values
        "Min.": v.min(),
        "1st Qrt.":  v.quantile(0.25),  # 25th percentile
        "Mean": v.mean(),
        "Median": v.median(),  # 50th percentile
        "3rd Qrt.":  v.quantile(0.75),  # 75th percentile
        "Max.": v.max(),
        "Std. Dev.": v.std(),
    }


def categorical_row(col):
    """
    Build ONE row of the categorical DQR for a single feature.
    """
    s = raw[col]
    v = s.dropna()  # non-missing values only
    vc = v.value_counts()  # values sorted by frequency, most common first
    base = pct_base(s)  # denominator for the percentages

    # Default everything to NaN in case a feature has fewer than 2 distinct values.
    mode1 = mode1_f = mode2 = mode2_f = np.nan
    m1p = m2p = np.nan

    if len(vc) >= 1:  # there is at least one value
        # most common value + its count
        mode1, mode1_f = vc.index[0], int(vc.iloc[0])
        m1p = round(100 * mode1_f / base, 2)
    if len(vc) >= 2:  # there is a second distinct value
        mode2, mode2_f = vc.index[1], int(vc.iloc[1])
        m2p = round(100 * mode2_f / base, 2)

    return {
        "Feature": col,
        "Count": N,
        "% Miss.": round(100 * s.isna().sum() / N, 2),
        "Card.": v.nunique(),
        "Mode": mode1,
        "Mode Freq.": mode1_f,
        "Mode %": m1p,
        "2nd Mode": mode2,
        "2nd Mode Freq.": mode2_f,
        "2nd Mode %": m2p,
    }


# The EXACT column order required
CONT_ORDER = ["Feature", "Count", "% Miss.", "Card.", "Min.", "1st Qrt.", "Mean",
              "Median", "3rd Qrt.", "Max.", "Std. Dev."]
CAT_ORDER = ["Feature", "Count", "% Miss.", "Card.", "Mode", "Mode Freq.", "Mode %",
             "2nd Mode", "2nd Mode Freq.", "2nd Mode %"]

# Build each table by making one row per feature
cont_dqr = pd.DataFrame([continuous_row(c)
                        for c in continuous_feats])[CONT_ORDER]
cat_dqr = pd.DataFrame([categorical_row(c)
                       for c in categorical_feats])[CAT_ORDER]

# The target is a classification label
target_dqr = pd.DataFrame([categorical_row(target_col)])[CAT_ORDER]

# Stage 4: Save the tables
cont_dqr.to_csv("continuous_dqr.csv", index=False, header=False)
cat_dqr.to_csv("categorical_dqr.csv", index=False, header=False)
target_dqr.to_csv("target_dqr.csv", index=False, header=False)

# Stage 4: Draw the plots


def grid(n):
    """
    Work out a near-square grid (rows, cols) that can hold n plots.
    """
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def save_hist(feats, path):
    """Draw a histogram for every continuous feature into one grid, save as PDF."""
    if not feats:
        return
    nr, nc = grid(len(feats))
    # Create a figure with nr x nc small sub-plots. figsize is in inches;
    # we scale it by the grid size so each little plot stays legible.
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.6, nr * 2.2))
    # .reshape(-1) flattens the 2D grid of axes into a simple 1D list so we can
    # loop through it easily alongside the feature names.
    axes = np.array(axes).reshape(-1)
    for ax, c in zip(axes, feats):
        ax.hist(pd.to_numeric(raw[c], errors="coerce").dropna(), bins=20)
        ax.set_title(c, fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("value", fontsize=5)
        ax.set_ylabel("freq", fontsize=5)
    # Any leftover empty cells in the grid get switched off so they are blank.
    for ax in axes[len(feats):]:
        ax.axis("off")
    fig.tight_layout()  # stop labels overlapping
    with PdfPages(path) as pdf:  # write the whole figure into one PDF
        pdf.savefig(fig)
    plt.close(fig)  # free the memory


def save_bars(feats, path, max_levels=25):
    """
    Draw a bar plot (value counts) for every categorical feature into one grid.
    """
    if not feats:
        return
    nr, nc = grid(len(feats))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.8, nr * 2.4))
    axes = np.array(axes).reshape(-1)
    for ax, c in zip(axes, feats):
        # Count how often each category appears, keep the top max_levels.
        vc = raw[c].dropna().astype(str).value_counts().head(max_levels)
        ax.bar(range(len(vc)), vc.values)  # one bar per category
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=90,
                           fontsize=4)  # category labels
        ax.set_title(c, fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_ylabel("freq", fontsize=5)
    for ax in axes[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


# Histograms for continuous features, bar plots for categorical features.
save_hist(continuous_feats,  "histograms.pdf")
save_bars(categorical_feats, "barplots.pdf")

# The target figure. T is a classification label (categorical), so we draw a
# single bar plot of how many instances fall into each class.
fig, ax = plt.subplots(figsize=(4, 3))
vc = raw[target_col].dropna().astype(str).value_counts()
ax.bar(range(len(vc)), vc.values)
ax.set_xticks(range(len(vc)))
ax.set_xticklabels(vc.index, rotation=90, fontsize=6)
ax.set_title("T (target)", fontsize=9)
ax.set_xlabel("class")
ax.set_ylabel("freq")
fig.tight_layout()
with PdfPages("target_figure.pdf") as pdf:
    pdf.savefig(fig)
plt.close(fig)


print("Done.")
print(f"  continuous features ({len(continuous_feats)}): {continuous_feats}")
print(
    f"  categorical features ({len(categorical_feats)}): {categorical_feats}")
print("  wrote: continuous_dqr.csv, categorical_dqr.csv, target_dqr.csv,")
print("         histograms.pdf, barplots.pdf, target_figure.pdf")
