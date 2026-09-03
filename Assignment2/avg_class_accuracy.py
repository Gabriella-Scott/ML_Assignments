"""
Average class accuracy, arithmetic and harmonic.
"""

from pathlib import Path

import pandas as pd

TAB_DIR = Path("output/phase4/tables")
MODELS = ["tree", "knn"]


def class_recall(cm):
    """Per-class recall from a confusion matrix. Rows are true classes."""
    return cm.values.diagonal() / cm.values.sum(axis=1)


def main():
    rows = []
    for name in MODELS:
        path = TAB_DIR / f"confusion_{name}.csv"
        cm = pd.read_csv(path, index_col=0)
        rec = class_recall(cm)

        # harmonic mean is undefined if any class is never recalled
        if (rec == 0).any():
            dead = [c for c, r in zip(cm.index, rec) if r == 0]
            hm = float("nan")
            print(f"{name}: zero recall on {dead}, harmonic mean undefined")
        else:
            hm = len(rec) / (1 / rec).sum()

        rows.append({"model": name,
                     "avg_class_acc_AM": rec.mean(),  # equals balanced accuracy
                     "avg_class_acc_HM": hm})

        print(f"\n{name} per-class recall")
        print(pd.Series(rec, index=cm.index).round(4).to_string())

    out = pd.DataFrame(rows).round(4)
    print()
    print(out.to_string(index=False))
    out.to_csv(TAB_DIR / "avg_class_accuracy.csv", index=False)
    print(f"\n-> {TAB_DIR / 'avg_class_accuracy.csv'}")
    print("check: the AM column should match balanced accuracy in summary.csv")


if __name__ == "__main__":
    main()