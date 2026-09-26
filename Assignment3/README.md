# ML 741 Assignment 3: Mini-batch training with dynamic meta-heuristics

Tests the hypothesis that a PSO developed for dynamic environments (quantum PSO)
trains a feedforward neural network better under mini-batching than a static
PSO (gbest inertia PSO, the control) and than mini-batch SGD (the baseline).

## Layout

| Path | Purpose |
|---|---|
| `config.py` | Every control parameter in one place, with the reason and reference for each |

## Running (Linux, Python 3.10+)

```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1        # stop numpy multithreading tiny matmuls
python -m experiments.tune_weight_decay
python -m experiments.tune_params
python -m experiments.run_experiments --workers 6
python -m experiments.analyse
```