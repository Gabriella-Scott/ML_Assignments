"""Completeness probe for Task 2: objective skewness + looser-threshold correlation.
Run on the full badDataSet.csv, then paste the printed output."""
import pandas as pd
import numpy as np

INPUT_FILE = "badDataSet.csv"
SEP = ","
NA_TOKEN = "?"
CAT_CARD_THRESH = 10
NUMERIC_FRAC_THRESH = 0.9

raw = pd.read_csv(INPUT_FILE, sep=SEP, header=None,
                  na_values=[NA_TOKEN], keep_default_na=True)
raw.columns = [f"A{i}" for i in range(1, raw.shape[1])]+["T"]
feature_cols = list(raw.columns[:-1])


def numeric_frac(s):
    c = pd.to_numeric(s, errors="coerce")
    p = s.notna()
    npz = int(p.sum())
    return (int((c.notna() & p).sum())/npz if npz else 0.0)


def classify(col):
    s = raw[col]
    if numeric_frac(s) < NUMERIC_FRAC_THRESH:
        return "categorical"
    return "categorical" if pd.to_numeric(s, errors="coerce").dropna().nunique() <= CAT_CARD_THRESH else "continuous"


cont = [c for c in feature_cols if classify(c) == "continuous"]

print("="*70)
print("OBJECTIVE SKEWNESS per continuous feature (Fisher-Pearson, via pandas)")
print("  |skew| > 2 = highly skewed;  1-2 = moderately skewed")
print("="*70)
rows = []
for c in cont:
    v = pd.to_numeric(raw[c], errors="coerce").dropna()
    rows.append((c, float(v.skew())))
for c, sk in sorted(rows, key=lambda t: -abs(t[1])):
    tag = "HIGH" if abs(sk) > 2 else ("mod" if abs(sk) > 1 else "")
    print(f"  {c:>4}  skew = {sk:+12.3f}   {tag}")

print()
print("="*70)
print("CORRELATION at looser 'very strong' threshold  |r| >= 0.90")
print("="*70)
cn = raw[cont].apply(pd.to_numeric, errors="coerce")
corr = cn.corr()
cols = list(corr.columns)
flagged = []
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        r = corr.iloc[i, j]
        if pd.notna(r) and abs(r) >= 0.90:
            flagged.append((cols[i], cols[j], r))
for a, b, r in sorted(flagged, key=lambda t: -abs(t[2])):
    star = "" if abs(
        r) >= 0.95 else "   <-- NEW at 0.90 (was below 0.95 cutoff)"
    print(f"  {a:>4} <-> {b:<4} r={r:+.4f}{star}")
