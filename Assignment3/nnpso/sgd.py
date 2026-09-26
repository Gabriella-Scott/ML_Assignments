import numpy as np

import config
from nnpso.data import batches
from nnpso.network import log_point


def train_sgd(net, data, batch_size, lam, params, max_patterns, seed):
    # returns final weights + history
    rng = np.random.default_rng([seed, 0])  # weight init
    # batch order, same for every algo
    batch_rng = np.random.default_rng([seed, 1])
    gen = batches(data["X_train"], data["T_train"], batch_size, batch_rng)

    w = net.init_weights(rng, *config.INIT_RANGE)
    v = np.zeros(net.D)  # momentum buffer (no effect if momentum = 0)
    used, step = 0, 0
    history = [log_point(net, w, data, lam, used)]  # starting point for plots

    while used < max_patterns:
        Xb, Tb = next(gen)
        loss, g = net.gradient(w, Xb, Tb, lam)
        v = params.momentum * v - params.learning_rate * g  # update momentum buffer
        w += v
        used += len(Xb) * config.GRAD_COST_FACTOR
        step += 1
        if step % config.LOG_EVERY == 0:
            history.append(log_point(net, w, data, lam, used))

    return w, history


if __name__ == "__main__":
    from nnpso.data import load_data
    from nnpso.network import Network

    d = load_data("iris", seed=0)
    net = Network(d["X_train"].shape[1], 10, d["T_train"].shape[1])
    params = config.SGDParams(learning_rate=0.1)

    for b in (16, "full"):
        w, hist = train_sgd(net, d, batch_size=b, lam=1e-4, params=params,
                            max_patterns=50_000, seed=0)
        print(f"batch {b}: {len(hist)} log points")
        print("  train loss first -> last:",
              round(hist[0]["train_loss"], 3), "->", round(hist[-1]["train_loss"], 3))
        print("  val:", {k: round(v, 3)
              for k, v in net.evaluate(w, d, split="val").items()})
