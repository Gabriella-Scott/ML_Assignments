from nnpso.swarm import train_swarm
from nnpso.sgd import train_sgd
from nnpso.network import Network
from nnpso.data import load_data
import config
from multiprocessing import Pool
import time
import itertools
import csv
import argparse
import os

RESULTS = "results/results.csv"
HISTORY = "results/history.csv"


def get_params(algo, dataset):
    # control params for one algo on one dataset
    if algo == "sgd":
        return config.SGDParams()
    if algo == "pso":
        return config.PSOParams()
    return config.QSOParams(r_cloud=config.R_CLOUD[dataset])


def run_one(dataset, batch_size, algo, run, lam, max_patterns, params, split="test"):
    # one independent run -> (result row, history)
    # same seed for every algo/batch in a run -> same split (paired design)
    seed = config.BASE_SEED + run
    data = load_data(dataset, seed, config.SPLIT)
    n_in, n_out = data["X_train"].shape[1], data["T_train"].shape[1]
    net = Network(n_in, config.HIDDEN_UNITS[dataset], n_out)

    t0 = time.perf_counter()
    if algo == "sgd":
        w, history = train_sgd(net, data, batch_size,
                               lam, params, max_patterns, seed)
    else:  # pso or qso
        w, history = train_swarm(
            net, data, batch_size, lam, params, max_patterns, seed, algo=algo)
    seconds = time.perf_counter() - t0

    row = {"dataset": dataset, "batch": str(batch_size), "algo": algo, "run": run,
           "lam": lam, "seconds": round(seconds, 2)}
    row.update(net.evaluate(w, data, split))
    return row, history


def append_rows(path, rows):
    # append dicts to csv, header only when file is new
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def done_runs():
    # (dataset, batch, algo, run) already in results.csv -> skipped on restart
    if not os.path.exists(RESULTS):
        return set()

    with open(RESULTS, "r", newline="") as f:
        reader = csv.DictReader(f)
        done = set()  # set of (dataset, batch, algo, run) already completed
        for row in reader:  # add each completed run to the set
            done.add((row["dataset"], row["batch"],
                     row["algo"], int(row["run"])))
    return done


def job(args):
    # worker fn for Pool, one grid cell
    dataset, batch_size, algo, run = args
    return run_one(dataset, batch_size, algo, run,
                   lam=config.WEIGHT_DECAY[dataset],
                   max_patterns=config.BUDGET_PATTERNS[dataset],
                   params=get_params(algo, dataset))


def timing(max_patterns=100_000):
    # secs per 1M patterns, smallest + full batch -> used to pick BUDGET_PATTERNS
    for dataset in config.DATASETS:
        for algo in config.ALGORITHMS:
            for b in (8, "full"):
                row, _ = run_one(dataset, b, algo, 0, lam=1e-4,
                                 max_patterns=max_patterns, params=get_params(algo, dataset))
                print(f"{dataset}, {algo}, batch {b}: "
                      f"{row['seconds'] / max_patterns * 1e6:.2f} s per 1M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--runs", type=int, default=config.N_RUNS)
    args = ap.parse_args()

    if args.timing:
        timing()
        return

    os.makedirs("results", exist_ok=True)

    done = done_runs()
    jobs = [
        (dataset, batch, algo, run)
        for dataset, batch, algo, run in itertools.product(
            config.DATASETS, config.BATCH_SIZES, config.ALGORITHMS, range(
                args.runs)
        )
        if (dataset, str(batch), algo, run) not in done
    ]  # list of jobs to run, each a tuple (dataset, batch, algo, run) not already done
    print(f"{len(jobs)} runs to do")

    # run the jobs in parallel using a pool of workers
    with Pool(args.workers) as pool:
        for i, (row, history) in enumerate(pool.imap_unordered(job, jobs), 1):
            append_rows(RESULTS, [row])
            append_rows(HISTORY, [{**h, "dataset": row["dataset"], "batch": row["batch"],
                        "algo": row["algo"], "run": row["run"]} for h in history])
            print(
                f"{i}/{len(jobs)}: {row['dataset']}, {row['batch']}, {row['algo']}, {row['run']}, acc={row['acc']}, seconds={row['seconds']}")


if __name__ == "__main__":
    main()
