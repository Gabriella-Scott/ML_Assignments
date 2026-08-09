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
NUMERIC_FRAC_THRESH = 0.9  # Continuous vs Categorical decision

FORCE_CONT = []
FORCE_CAT = []

USE_TOTAL_FOR_PCT = False  # Mode % / 2nd Mode % use the PRESENT (non-missing)
# count as denominator, matching the lecturer's DQR example (MARITAL STATUS:
# mode freq 99 with 61.2% missing shown as 51.0% = 99/194, i.e. present-based).

PLOTS_PER_ROW = 2
ROWS_PER_PAGE = 3   # 2 x 3 = 6 plots per page
BAR_MAX_LEVELS = 25  # max number of bars in a bar plot


# Stage 1: Load file and label columns

# pd.read_csv reads the file into a DataFrame (a table).
#   sep=SEP              -> split each line on commas
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


def numeric_views(s):
    """
    Try to read the column as numbers
    """
    coerced = pd.to_numeric(
        s, errors="coerce")  # convert to numeric, invalids become NaN
    present = s.notna()
    n_present = int(present.sum())
    n_numeric = int((coerced.notna() & present).sum())
    invalid = n_present - n_numeric
    frac = (n_numeric / n_present) if n_present else 0.0
    return coerced, frac, invalid


def classify(col):
    s = raw[col]
    if col in FORCE_CONT:
        return "continuous"
    if col in FORCE_CAT:
        return "categorical"
    coerced, frac, _ = numeric_views(s)
    if frac < NUMERIC_FRAC_THRESH:
        return "categorical"  # mostly text -> categorical
    card = coerced.dropna().nunique()
    return "categorical" if card <= CAT_CARD_THRESH else "continuous"


# build a dict then split feature names into 2 lists based on the decision
types = {c: classify(c) for c in feature_cols}
continuous_feats = [c for c in feature_cols if types[c] == "continuous"]
categorical_feats = [c for c in feature_cols if types[c] == "categorical"]

# Stage 2b: Print a feature summary

summary = []
for c in feature_cols:
    s = raw[c]
    _, _, invalid = numeric_views(s)
    non_null = s.dropna()
    summary.append({
        "Feature": c,
        "AssignedType": types[c],
        "RawDtype": str(s.dtype),
        "Card": non_null.nunique(),
        "%Miss": round(100 * s.isna().sum() / N, 2),
        "Invalid": invalid,
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
        # % Miss. counts only the '?' tokens (the spec's definition of missing),
        # exactly like the categorical table. Non-numeric INVALID tokens (e.g.
        # A10's 'a') are NOT missing, so they are excluded here and captured as
        # an Invalid Values issue in Task 2 instead.
        "% Miss.": round(100 * raw[col].isna().sum() / N, 2),
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

    if len(vc) >= 1:
        mode1, mode1_f = vc.index[0], int(vc.iloc[0])
        m1p = round(100 * mode1_f / base, 2)
    if len(vc) >= 2:
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

# Stage 5: Draw the plots


def save_grid(feats, path, kind):
    """
    Draw one small plot per feature, arranged as a matrix, PAGINATED across
    multiple pages of a single PDF so nothing is cramped.
      kind="hist" -> histogram (continuous)
      kind="bar"  -> bar plot of value counts (categorical)
    """
    if not feats:
        return
    per_page = PLOTS_PER_ROW * ROWS_PER_PAGE
    with PdfPages(path) as pdf:
        # step through the features in chunks of per_page
        for start in range(0, len(feats), per_page):
            chunk = feats[start:start + per_page]
            fig, axes = plt.subplots(ROWS_PER_PAGE, PLOTS_PER_ROW,
                                     figsize=(PLOTS_PER_ROW * 3.6,
                                              ROWS_PER_PAGE * 3.0))
            axes = np.array(axes).reshape(-1)      # flatten grid to a 1-D list
            for ax, c in zip(axes, chunk):
                if kind == "hist":
                    ax.hist(pd.to_numeric(raw[c], errors="coerce").dropna(),
                            bins=30)
                    ax.set_xlabel("value", fontsize=9)
                else:  # bar
                    vc = (raw[c].dropna().astype(str)
                          .value_counts().head(BAR_MAX_LEVELS))
                    ax.bar(range(len(vc)), vc.values)
                    ax.set_xticks(range(len(vc)))
                    ax.set_xticklabels(vc.index, rotation=90, fontsize=7)
                    ax.set_xlabel("category", fontsize=9)
                ax.set_title(c, fontsize=12)
                ax.set_ylabel("frequency", fontsize=9)
                ax.tick_params(labelsize=8)
            # blank out any unused cells on the last page
            for ax in axes[len(chunk):]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


save_grid(continuous_feats, "histograms.pdf", kind="hist")
save_grid(categorical_feats, "barplots.pdf", kind="bar")

# Target figure: T is a class label (categorical) -> single bar plot.
fig, ax = plt.subplots(figsize=(6, 4))
vc = raw[target_col].dropna().astype(str).value_counts()
ax.bar(range(len(vc)), vc.values)
ax.set_xticks(range(len(vc)))
ax.set_xticklabels(vc.index, rotation=90, fontsize=9)
ax.set_title("T (target)", fontsize=12)
ax.set_xlabel("class", fontsize=10)
ax.set_ylabel("frequency", fontsize=10)
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
