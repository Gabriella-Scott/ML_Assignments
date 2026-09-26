from dataclasses import dataclass, field

N_RUNS = 30
BASE_SEED = 2026
SPLIT = (0.6, 0.2, 0.2)
DATASETS = ["iris", "wine", "cancer"]
BATCH_SIZES = [8, 16, 32, 64, "full"]

# Network architecture
HIDDEN_UNITS = {
    "iris": 10,    # PILOT/VERIFY: deliberate overestimate
    "wine": 20,   
    "cancer": 20, 
}

HIDDEN_ACTIVATION = "sigmoid"   
OUTPUT_ACTIVATION = "softmax"
LOSS = "cross_entropy"          

INIT_RANGE = (-1.0, 1.0)

# Weight decay (L2 penalty)
WEIGHT_DECAY_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
WEIGHT_DECAY = {"iris": None, "wine": None, "cancer": None}  # PILOT

PENALISE_BIASES = False

# Training budget
BUDGET_PATTERNS = {"iris": None, "wine": None, "cancer": None}  # PILOT
GRAD_COST_FACTOR = 2
LOG_EVERY = 50

# Algo control params
@dataclass
class SGDParams:
    learning_rate: float = 0.05 
    momentum: float = 0.0
    # Plain SGD without momentum keeps the baseline simple

LR_GRID = [0.001, 0.01, 0.05, 0.1, 0.5]


@dataclass
class PSOParams:
    swarm_size: int = 30
    # 30 particles is the common default
    w: float = 0.729844
    c1: float = 1.49618
    c2: float = 1.49618
    # Satisfy the convergence condition c1 + c2 < 24(1 - w^2)/(7 - 5w)
    vmax_fraction: float = 0.5
    # Velocity clamped to vmax_fraction * width of INIT_RANGE per dimension
    iters_per_batch: int = 1


@dataclass
class QSOParams(PSOParams):
    quantum_fraction: float = 0.5
    r_cloud: float = 0.5               # PILOT, grid in RCLOUD_GRID
    reevaluate_on_change: bool = True

RCLOUD_GRID = [0.1, 0.5, 1.0, 2.0]

ALGORITHMS = ["sgd", "pso", "qso"]