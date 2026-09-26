import config
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


RESULTS = "results/results.csv"
HISTORY = "results/history.csv"
FIG_DIR = "figures"
TABLE_DIR = "results/tables"

ALPHA = 0.05
ALGOS = ["sgd", "pso", "qso"]
NAMES = {"sgd": "SGD", "pso": "PSO", "qso": "QSO"}
PAIRS = [("qso", "pso"), ("qso", "sgd"), ("pso", "sgd")]
BATCHES = [str(b) for b in config.BATCH_SIZES]  # csv stores batch as str
LOWER_BETTER = {"ce": True, "acc": False, "f1": False}


def holm(pvals):
    # Holm-Bonferroni adjusted p-values, same order as input
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0
    for rank, i in enumerate(order):
        adj[i] = (m - rank) * pvals[i]
        running_max = max(running_max, adj[i])
        adj[i] = min(running_max, 1.0)
    return adj


def paired(df, dataset, batch, measure):
    # {algo: values sorted by run} -> same run order for every algo
    sub = df[(df.dataset == dataset) & (df.batch == batch)]
    return {a: sub[sub.algo == a].sort_values("run")[measure].to_numpy() for a in ALGOS}


def wilcoxon_p(x, y):
    # paired test, all diffs zero -> identical results, p = 1
    if np.all(x == y):
        return 1.0
    return stats.wilcoxon(x, y).pvalue


def compare(df, measure):
    # one row per (dataset, batch): mean, std, friedman p, holm p per pair
    rows = []
    for d in config.DATASETS:
        for b in BATCHES:
            vals = paired(df, d, b, measure)
            row = {"dataset": d, "batch": b}
            for a in ALGOS:
                row[f"mean_{a}"] = np.mean(vals[a])
                row[f"std_{a}"] = np.std(
                    vals[a], ddof=1)

            if all(np.all(vals[a] == vals[ALGOS[0]]) for a in ALGOS):
                row["friedman_p"] = 1.0
            else:
                row["friedman_p"] = stats.friedmanchisquare(
                    *vals.values()).pvalue

            raw = [wilcoxon_p(vals[a], vals[c]) for a, c in PAIRS]
            adj = holm(raw)
            for (a, c), p in zip(PAIRS, adj):
                row[f"p_{a}_{c}"] = p
            # mark significance for easy access
            for a, c in PAIRS:
                row[f"sig_{a}_{c}"] = significant(row, a, c)

            rows.append(row)
    return pd.DataFrame(rows)


def significant(row, a, c):
    # post-hoc only counts if the friedman test was significant too
    return row["friedman_p"] < ALPHA and row[f"p_{a}_{c}"] < ALPHA


def verdict(cmp, measure, a, c):
    # win/draw/loss of a vs c over all cases
    # win = significant + a better, loss = significant + a worse, else draw
    wins = draws = losses = 0
    for _, row in cmp.iterrows():
        a_better = row[f"mean_{a}"] < row[f"mean_{c}"] if LOWER_BETTER[measure] else row[f"mean_{a}"] > row[f"mean_{c}"]
        sig = significant(row, a, c)
        if sig and a_better:
            wins += 1
        elif sig and not a_better:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses
    pass


def latex_table(cmp, measure, path):
    # mean +- std per algo, best mean in bold, * = significantly different from QSO
    lines = []
    for _, row in cmp.iterrows():
        best = min((row[f"mean_{a}"], a) for a in ALGOS) if LOWER_BETTER[measure] else max(
            (row[f"mean_{a}"], a) for a in ALGOS)
        best_algo = best[1]
        cells = []
        for a in ALGOS:
            cell = f"{row[f'mean_{a}']:.3f} $\\pm$ {row[f'std_{a}']:.3f}"
            if a == best_algo:
                cell = f"\\textbf{{{cell}}}"
            if a != "qso" and significant(row, "qso", a):
                cell += "$^{*}$"
            cells.append(cell)
        lines.append(f"{row['dataset']} & {row['batch']} & " +
                     " & ".join(cells) + " \\\\")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_vs_batch(df, measure, ylabel, path):
    # 1 subplot per dataset, x = batch size, 1 line per algo, mean +- std
    fig, axes = plt.subplots(1, len(config.DATASETS), figsize=(7, 2.4))
    x = np.arange(len(BATCHES))
    for ax, d in zip(axes, config.DATASETS):
        for a in ALGOS:
            sub = df[(df["dataset"] == d) & (df["algo"] == a)]
            means = sub.groupby("batch")[measure].mean().reindex(BATCHES)
            stds = sub.groupby("batch")[measure].std().reindex(BATCHES)
            ax.errorbar(x, means, yerr=stds,
                        label=NAMES[a], capsize=2, marker="o")
        ax.set_xticks(x, BATCHES)
        ax.set_title(d)
        ax.set_xlabel("Batch size")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(hist, batches, path):
    # median val CE vs % of budget, rows = datasets, cols = chosen batch sizes
    # log point index -> same x for every run (0..LOG_POINTS)
    hist["point"] = hist.groupby(
        ["dataset", "batch", "algo", "run"]).cumcount()
    fig, axes = plt.subplots(len(config.DATASETS), len(batches),
                             figsize=(7, 6), squeeze=False)
    for i, d in enumerate(config.DATASETS):
        for j, b in enumerate(batches):
            ax = axes[i][j]
            for a in ALGOS:
                sub = hist[(hist["dataset"] == d) & (
                    hist["batch"] == str(b)) & (hist["algo"] == a)]
                med = sub.groupby("point")["val_ce"].median()
                ax.plot(med.index, med.values, label=NAMES[a])
            ax.set_title(f"{d}, batch {b}", fontsize=8)
            ax.set_yscale("log")  # CE spans orders of magnitude
            if i == len(config.DATASETS) - 1:
                ax.set_xlabel("% of budget")
            if j == 0:
                ax.set_ylabel("val CE")
    axes[0][0].legend(fontsize=7)
    # fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS, dtype={"batch": str})
    hist = pd.read_csv(HISTORY, dtype={"batch": str})

    for m in ("ce", "acc", "f1"):
        cmp = compare(df, m)
        cmp.to_csv(f"{TABLE_DIR}/compare_{m}.csv", index=False)
        latex_table(cmp, m, f"{TABLE_DIR}/table_{m}.tex")
        for a, c in [("qso", "pso"), ("qso", "sgd")]:
            print(
                f"{m}: {NAMES[a]} vs {NAMES[c]} (W/D/L): {verdict(cmp, m, a, c)}")

    plot_vs_batch(df, "ce", "Test CE", f"{FIG_DIR}/ce_vs_batch.pdf")
    plot_vs_batch(df, "acc", "Test accuracy", f"{FIG_DIR}/acc_vs_batch.pdf")
    plot_convergence(hist, ["8", "full"], f"{FIG_DIR}/convergence.pdf")


if __name__ == "__main__":
    main()
