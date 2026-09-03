"""Check for duplicate rows with different labels."""
import pandas as pd
from sklearn.metrics import f1_score

import preprocess as pp

NAMES = {0: "Normal", 1: "Reconnaissance", 2: "Backdoor", 3: "DoS",
         4: "Exploits", 5: "Analysis", 6: "Fuzzers", 7: "Worms",
         8: "Shellcode", 9: "Generic"}


def main():
    df, _ = pp.clean_common(pp.load_raw(), verbose=False) # load and clean data
    X, y = pp.split_X_y(df) # split into feats and lbls
    n = len(X)
    # Treat missing values as a category.
    key = X.fillna("NA").astype(str).agg("|".join, axis=1)

    pairs = pd.DataFrame({"key": key, "y": y}).value_counts().reset_index(name="n")
    size = pairs.groupby("key")["n"].sum()
    distinct = pairs.groupby("key").size()

    # Use the most common label per group.
    majority = pairs.sort_values("n", ascending=False).drop_duplicates("key")
    oracle = key.map(majority.set_index("key")["y"])

    # Report on dups
    conflict = distinct > 1
    print(f"instances:                 {n:,}")
    print(f"distinct feature vectors:  {len(size):,}")
    print(f"in a conflicting group:    {size[conflict].sum():,} "
          f"({100 * size[conflict].sum() / n:.1f}%)")

    print("\nceiling for any classifier on these features")
    print(f"  accuracy:   {(oracle == y).mean():.4f}")
    print(f"  macro F1:   {f1_score(y, oracle, average='macro', zero_division=0):.4f}")
    print(f"  per class:  ")
    per = f1_score(y, oracle, average=None, labels=sorted(NAMES), zero_division=0)
    for c, s in enumerate(per):
        print(f"    {NAMES[c]:<15} {s:.4f}")

    # Show labels for ambiguous rows.
    bad = y != oracle
    print(f"\nunreachable instances: {bad.sum():,}")
    tab = pd.crosstab(y[bad].map(NAMES), oracle[bad].map(NAMES))
    print("rows are true class, columns are the label they collapse into")
    print(tab.to_string())


if __name__ == "__main__":
    main()