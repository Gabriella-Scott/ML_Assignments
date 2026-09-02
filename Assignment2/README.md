# ML 741 Assignment 2

Experimental pipeline for comparing k-nearest neighbours and decision trees
on the provided UNSW-NB15 network-traffic data.

## Requirements

Python 3 with `numpy`, `pandas`, `scikit-learn`, `scipy`, and `matplotlib`.
The supplied data files must remain in `data/`.

## Run

Run the stages from the repository root:

```bash
python3 explore.py
python3 preprocess.py
python3 tune.py --jobs 8
python3 evaluate.py --jobs 8
```

`explore.py` produces the data-quality analysis in `output/phase1/`.
`preprocess.py` records model-specific preprocessing decisions in
`output/phase2/`. `tune.py` uses 5-fold cross-validation on a tuning pool and
writes the selected parameters to `output/phase3/best_params.json`.
`evaluate.py` reads that file and reports a 10-fold paired comparison on the
disjoint evaluation pool in `output/phase4/`.

Use `--help` on `tune.py` or `evaluate.py` to view data, sample-size, fold,
and parallelism options. Use fewer `--jobs` where system resources are
limited.

## Outputs

Each phase writes report-ready CSV tables and figures beneath `output/phaseN/`.
The primary final results are in `output/phase4/tables/`, including fold-level
metrics, per-class scores, confusion matrices, runtime, and the paired model
comparison.

## Utilities

```bash
python3 duplicates.py
python3 make_devdata.py
```

`duplicates.py` estimates the performance ceiling imposed by identical feature
vectors with conflicting labels. `make_devdata.py` creates `data/devTraffic.csv`,
a small fixture for development only; it is not part of the final experiment.
