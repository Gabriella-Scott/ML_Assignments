import numpy as np

import config
from nnpso.data import batches
from nnpso.network import log_point


class PSO:
    """gbest inertia PSO, static control: pbests never re-evaluated."""

    def __init__(self, net, params, lam, rng):
        self.net, self.p, self.lam, self.rng = net, params, lam, rng
        lo, hi = config.INIT_RANGE
        self.vmax = params.vmax_fraction * (hi - lo)

        self.n_s, self.D = params.swarm_size, net.D
        self.X = rng.uniform(lo, hi, size=(self.n_s, self.D))  # positions
        self.V = np.zeros_like(self.X)  # velocities
        self.Y = self.X.copy()  # pbest positions
        self.Yf = np.full(self.n_s, np.inf)  # pbest losses
        self.gbest = self.Y[0].copy()

    def evaluate(self, Xb, Tb):
        # loss for all particles, update pbests + gbest
        f = self.net.swarm_loss(self.X, Xb, Tb, self.lam)
        better = f < self.Yf
        self.Y[better] = self.X[better]
        self.Yf[better] = f[better]
        self.gbest = self.Y[np.argmin(self.Yf)].copy()
        return self.n_s * len(Xb)  # patterns used

    def move(self):
        r1 = self.rng.uniform(size=(self.n_s, self.D))
        r2 = self.rng.uniform(size=(self.n_s, self.D))
        self.V = (self.p.w * self.V
                  + self.p.c1 * r1 * (self.Y - self.X)
                  + self.p.c2 * r2 * (self.gbest - self.X))
        self.V = np.clip(self.V, -self.vmax, self.vmax)
        self.X += self.V

    def on_new_batch(self, Xb, Tb):
        # static PSO ignores the change -> stale memory
        return 0


class QSO(PSO):
    """Quantum PSO: quantum particles around gbest + pbest re-eval on change."""

    def __init__(self, net, params, lam, rng):
        super().__init__(net, params, lam, rng)
        self.n_q = int(params.quantum_fraction * params.swarm_size)
        self.quantum = np.arange(self.n_s) < self.n_q  # first n_q are quantum
        self.r_cloud = params.r_cloud

    def on_new_batch(self, Xb, Tb):
        # re-evaluate pbests on the new batch, recompute gbest
        self.Yf = self.net.swarm_loss(self.Y, Xb, Tb, self.lam)
        self.gbest = self.Y[np.argmin(self.Yf)].copy()
        return len(self.Y) * len(Xb)  # patterns used

    def move(self):
        super().move()  # normal update for everyone
        # quantum particles: resample uniformly in ball around gbest
        self.X[self.quantum] = sample_in_ball(
            self.gbest, self.r_cloud, self.n_q, self.rng)


def sample_in_ball(centre, radius, n, rng):
    # n points uniform inside a D-dim ball around centre
    # direction = normalised gaussian vector, distance = radius * U^(1/D)
    D = centre.shape[0]
    directions = rng.normal(size=(n, D))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = radius * rng.uniform(size=(n, 1)) ** (1 / D)
    return centre + directions * distances


def train_swarm(net, data, batch_size, lam, params, max_patterns, seed, algo="pso"):
    # shared loop for PSO + QSO, returns gbest + history
    rng = np.random.default_rng([seed, 0])
    batch_rng = np.random.default_rng([seed, 1])  # same batch order as SGD
    gen = batches(data["X_train"], data["T_train"], batch_size, batch_rng)

    cls = QSO if algo == "qso" else PSO
    swarm = cls(net, params, lam, rng)

    Xb, Tb = next(gen)
    used = swarm.evaluate(Xb, Tb)  # initial pbests
    it = 0
    history = [log_point(net, swarm.gbest, data, lam, used)]  # starting point

    while used < max_patterns:
        it += 1
        # new batch every iters_per_batch its, full batch never changes
        if batch_size != "full" and it % params.iters_per_batch == 0:
            Xb, Tb = next(gen)
            used += swarm.on_new_batch(Xb, Tb)
        swarm.move()
        used += swarm.evaluate(Xb, Tb)
        if it % config.LOG_EVERY == 0:
            history.append(log_point(net, swarm.gbest, data, lam, used))

    return swarm.gbest.copy(), history


if __name__ == "__main__":
    # quick check: python3 -m nnpso.swarm (from Assignment3/)
    from nnpso.data import load_data
    from nnpso.network import Network

    rng = np.random.default_rng(0)
    pts = sample_in_ball(np.zeros(83), 0.5, 1000, rng)
    dist = np.linalg.norm(pts, axis=1)
    print("ball: max dist", round(dist.max(), 3),
          "mean dist", round(dist.mean(), 3))

    d = load_data("iris", seed=0)
    net = Network(d["X_train"].shape[1], 10, d["T_train"].shape[1])
    for b in (16, "full"):
        for algo, params in (("pso", config.PSOParams()), ("qso", config.QSOParams())):
            w, hist = train_swarm(net, d, batch_size=b, lam=1e-4, params=params,
                                  max_patterns=500_000, seed=0, algo=algo)
            print(f"{algo} batch {b}: {len(hist)} log points")
            print("  train loss first -> last:",
                  round(hist[0]["train_loss"], 3), "->", round(hist[-1]["train_loss"], 3))
            print("  val acc:", round(
                net.evaluate(w, d, split="val")["acc"], 3))
