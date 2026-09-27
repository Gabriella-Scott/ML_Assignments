import argparse
import itertools
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import config
from experiments.run import append_rows, run_one

N_PILOT = 10
PILOT_BATCH = 32
PILOT_RUN_OFFSET = 1000  # pilot seeds != final run seeds
OUT_DIR = "results/pilot"


def pilot_job(args):
    # one pilot run on the val split, value = lr (sgd) or r_cloud (qso)
    dataset, algo, lam, value, run = args

    params = config.SGDParams(
        learning_rate=value) if algo == "sgd" else config.QSOParams(r_cloud=value)
    row, _ = run_one(dataset, PILOT_BATCH, algo, PILOT_RUN_OFFSET + run, lam,
                     config.BUDGET_PATTERNS[dataset], params, split="val")
    row["value"] = value  # needed for grouping

    return row


def run_pilot(jobs, out_path, workers):
    # parallel runs, fresh csv, returns all rows
    if os.path.exists(out_path):
        os.remove(out_path)  # rerun -> no duplicates
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(pilot_job, jobs), 1):
            rows.append(row)
            append_rows(out_path, [row])
            if i % 20 == 0 or i == len(jobs):
                print(f"Progress: {i}/{len(jobs)}")
    return rows


def best_setting(rows, dataset, keys):
    # lowest mean val CE for one dataset, keys e.g. ("lam", "value")
    rows = [row for row in rows if row["dataset"] == dataset]
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row["ce"])

    for key, vals in sorted(grouped.items(), key=lambda x: np.mean(x[1])):
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"Setting {key}: {mean:.3f} +- {std:.3f}")

    best_key = min(grouped, key=lambda k: np.mean(grouped[k]))
    return best_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sgd", "qso"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.stage == "sgd":
        # lr x lam grid -> lam + lr per problem
        jobs = [(d, "sgd", lam, lr, r) for d in config.DATASETS
                for lam in config.WEIGHT_DECAY_GRID
                for lr in config.LR_GRID
                for r in range(N_PILOT)]
        rows = run_pilot(jobs, f"{OUT_DIR}/sgd.csv", args.workers)
        for d in config.DATASETS:
            print(f"\n=== {d}: (lam, lr) -> mean val CE +- std ===")
            print(
                f"Best setting for {d}: {best_setting(rows, d, ('lam', 'value'))}")

    else:
        # r_cloud search, lam from stage 1
        jobs = [(d, "qso", config.WEIGHT_DECAY[d], rc, r) for d in config.DATASETS
                for rc in config.RCLOUD_GRID
                for r in range(N_PILOT)]
        rows = run_pilot(jobs, f"{OUT_DIR}/qso.csv", args.workers)
        for d in config.DATASETS:
            print(f"\n=== {d}: (r_cloud,) -> mean val CE +- std ===")
            print(f"Best setting for {d}: {best_setting(rows, d, ('value',))}")


if __name__ == "__main__":
    main()
