# ML 741 Assignment 3: Mini-batch training with dynamic meta-heuristics

Tests the hypothesis that a PSO developed for dynamic environments (quantum PSO)
trains a feedforward neural network better under mini-batching than a static
PSO (gbest inertia PSO, the control) and than mini-batch SGD (the baseline).

## Layout

| Path | Purpose |
|---|---|
| `config.py` | Every control parameter in one place, with the reason and reference for each |
| `nnpso/data.py` | Loads/splits/normalises the sklearn datasets (iris, wine, cancer) |
| `nnpso/network.py` | Feedforward network (sigmoid hidden, softmax output, cross-entropy loss) |
| `nnpso/sgd.py` | Mini-batch SGD training loop (baseline) |
| `nnpso/swarm.py` | Mini-batch PSO / QSO training loop (control / treatment) |
| `experiments/tune.py` | Pilot runs on the val split to pick SGD learning rate/weight decay and QSO `r_cloud` |
| `experiments/run.py` | Runs the full `dataset x batch size x algorithm x run` grid, appends to `results/` |
| `experiments/analyse.py` | Statistical comparison (Friedman + Holm-corrected Wilcoxon), LaTeX tables, figures |
| `results/` | Raw run results (`results.csv`), training curves (`history.csv`), pilot runs, tables |
| `figures/` | Plots produced by `experiments/analyse.py` |

## Running (Linux, Python 3.10+)

```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1        # stop numpy multithreading tiny matmuls
python -m experiments.tune --stage sgd      # pilot: SGD learning rate + weight decay per dataset
python -m experiments.tune --stage qso      # pilot: QSO r_cloud per dataset
python -m experiments.run --workers 6       # full grid: dataset x batch size x {sgd,pso,qso} x run
python -m experiments.analyse               # stats, LaTeX tables, figures
```