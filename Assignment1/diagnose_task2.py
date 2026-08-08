"""Evidence gathering"""

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
CORR_FLAG = 0.95  # |r| at or above this = "Strong correlated"
INDEX_LIKE = ["A40", "A7"]  # unique-per-row columns to drop for dup check

# Stage 1: load and label
raw = pd.read_csv(INPUT_FILE, sep=SEP, header=None,
                  na_values=[NA_TOKEN], keep_default_na=True)
n_cols = raw.shape[1]
raw.columns = [f"A{i}" for i in range(1, n_cols)] + ["T"]
N = len(raw)
feature_cols = list(raw.columns[:-1])
target_col = "T"
print(f"Loaded {N} rows x {n_cols} columns\n")

# Stage 2: re-derive continuous / categorical split so the diagnostics
# operate on the right feature sets


def numeric_frac(s):
    """Fraction of Present values that parse a numbers (0..1)"""
    coerced = pd.to_numeric(s, errors="coerce")
    present = s.notna()
    n_present = int(present.sum())
    n_numeric = int((coerced.notna() & present).sum())
    return (n_numeric / n_present) if n_present else 0.0, n_present - n_numeric


def classify(col):
    s = raw[col]
    frac, _ = numeric_frac(s)
    if frac < NUMERIC_FRAC_THRESH:
        return "categorical"
    card = pd.to_numeric(s, errors="coerce").dropna().nunique()
    return "categorical" if card <= CAT_CARD_THRESH else "continuous"


types = {c: classify(c) for c in feature_cols}
continuous_feats = [c for c in feature_cols if types[c] == "continuous"]
categorical_feats = [c for c in feature_cols if types[c] == "categorical"]
print(f"continuous ({len(continuous_feats)}): {continuous_feats}")
print(f"categorical ({len(categorical_feats)}): {categorical_feats}\n")

# Stage 3: Invalid (non-numeric) tokens per feature
print("INVALID VALUES  (non-numeric tokens inside mostly-numeric features)")
print("-" * 70)

for c in feature_cols:
    s = raw[c]
    frac, n_invalid = numeric_frac(s)
    if frac >= NUMERIC_FRAC_THRESH and n_invalid > 0:
        # which present values fail to parse as numbers?
        present = s.dropna()
        bad = present[pd.to_numeric(present, errors="coerce").isna()]
        sample = ", ".join(map(str, pd.Series(bad.unique())[:5]))
        print(f"  {c:>4}: {n_invalid:>6} invalid "
              f"({100*n_invalid/N:.3f}% of rows)  e.g. [{sample}]")
print()

# Stage 4: Duplicate rows
print("-" * 70)
print("INSTANCE DUPLICATION")
print("=" * 70)
present_index_like = [c for c in INDEX_LIKE if c in raw.columns]
subset = raw.drop(columns=present_index_like)
n_dup_full = int(raw.duplicated().sum())
n_dup_noidx = int(subset.duplicated().sum())
print(f"  duplicate rows including {present_index_like}: {n_dup_full}")
print(f"  duplicate rows after dropping {present_index_like}: {n_dup_noidx} "
      f"({100*n_dup_noidx/N:.3f}% of rows)")
print()

# Stage 5: Correlation between continuous features
print("=" * 70)
print(f"CORRELATION  (continuous pairs with |r| >= {CORR_FLAG})")
print("=" * 70)
cont_num = raw[continuous_feats].apply(pd.to_numeric, errors="coerce")
corr = cont_num.corr()  # Pearson correlation, pairwise complete
flagged = []
cols = list(corr.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr.iloc[i, j]
        if pd.notna(r) and abs(r) >= CORR_FLAG:
            flagged.append((cols[i], cols[j], r))
flagged.sort(key=lambda t: -abs(t[2]))
if flagged:
    for a, b, r in flagged:
        print(f"  {a:>4} <-> {b:<4}  r = {r:+.4f}")
else:
    print("  (no pairs above the threshold)")
print()

# heatmap PDF (a citable figure)
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=6)
ax.set_yticks(range(len(cols)))
ax.set_yticklabels(cols, fontsize=6)
ax.set_title("Correlation matrix (continuous features)", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout()
with PdfPages("correlation_heatmap.pdf") as pdf:
    pdf.savefig(fig)
plt.close(fig)
print("  wrote correlation_heatmap.pdf\n")

# Stage 6: Integer / float twin check
# If two columns hold the same values but one is int-coded and one is float,
# that is both a data inconsistency and a redundancy. Measure how often they
# are numerically equal on rows where both are present.
print("=" * 70)
print("SUSPECTED TWIN COLUMNS")
print("=" * 70)


def twin_match(a, b):
    if a not in raw.columns or b not in raw.columns:
        return
    x = pd.to_numeric(raw[a], errors="coerce")
    yb = pd.to_numeric(raw[b], errors="coerce")
    both = x.notna() & yb.notna()
    if both.sum() == 0:
        print(f"  {a} vs {b}: no overlapping present values")
        return
    match = (x[both] == yb[both]).mean()
    print(f"  {a} vs {b}: equal on {100*match:.2f}% of shared-present rows "
          f"(n={int(both.sum())})")


for a, b in [("A41", "A42"), ("A24", "A25"), ("A23", "A26"), ("A28", "A29")]:
    twin_match(a, b)
print()

# Stage 7: Feature to target association (spot independent features)
print("=" * 70)
print("ASSOCIATION WITH TARGET T  (low = candidate 'Independent' feature)")
print("=" * 70)

y = raw[target_col]
y_present = y.notna()


def eta(cont_col):
    """Correlation ratio between a continuous feature and the class label."""
    x = pd.to_numeric(raw[cont_col], errors="coerce")
    m = x.notna() & y_present
    xs, ys = x[m], y[m].astype(str)
    grand = xs.mean()
    ss_total = ((xs - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, grp in xs.groupby(ys):
        ss_between += len(grp) * (grp.mean() - grand) ** 2
    return np.sqrt(ss_between / ss_total)


def cramers_v(cat_col):
    """Cramer's V between a categorical feature and the class label."""
    t = pd.crosstab(raw[cat_col], y)
    if t.size == 0:
        return 0.0
    chi2 = 0.0
    rt, ct, tot = t.sum(axis=1).values, t.sum(axis=0).values, t.values.sum()
    exp = np.outer(rt, ct) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((t.values - exp) ** 2 / exp)
    k = min(t.shape) - 1
    return np.sqrt(chi2 / (tot * k)) if k > 0 and tot > 0 else 0.0


assoc = []
for c in continuous_feats:
    assoc.append((c, "eta", eta(c)))
for c in categorical_feats:
    assoc.append((c, "CramersV", cramers_v(c)))
assoc.sort(key=lambda t: t[2])  # weakest association first
print("  weakest 15 associations (most likely independent / irrelevant):")
for c, kind, v in assoc[:15]:
    print(f"    {c:>4}  {kind:>9} = {v:.4f}")
print()

print("Done. Sections above are your Task 2 evidence for:")
print("  invalid values, duplicates, correlation, twins, and independence.")
