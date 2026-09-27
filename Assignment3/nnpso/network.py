import numpy as np
from sklearn.metrics import accuracy_score, f1_score

EPS = 1e-12  # avoid log(0)


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # subtract row max for stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def add_bias(A):
    # append col of ones -> bias = last col of each weight matrix
    return np.hstack([A, np.ones((A.shape[0], 1))])


def cross_entropy(O, T):
    # mean CE over patterns, no penalty
    return -np.sum(T * np.log(O + EPS)) / O.shape[0]


class Network:
    """1 hidden layer: I -> H sigmoid -> K softmax, all weights in one flat vector."""

    def __init__(self, n_in, n_hidden, n_out):
        self.I, self.H, self.K = n_in, n_hidden, n_out
        # size of W1 incl. biases
        self.n1 = n_hidden * (n_in + 1)
        self.D = self.n1 + n_out * (n_hidden + 1)  # total no. of weights

        # 1 = weight (decayed), 0 = bias (not penalised)
        m1 = np.ones((self.H, self.I + 1), dtype=int)
        m1[:, -1] = 0
        m2 = np.ones((self.K, self.H + 1), dtype=int)
        m2[:, -1] = 0
        self.mask = np.concatenate([m1.ravel(), m2.ravel()])

    def unpack(self, w):
        # flat (D,) -> W1 (H, I+1), W2 (K, H+1)
        W1 = w[:self.n1].reshape(self.H, self.I + 1)
        W2 = w[self.n1:].reshape(self.K, self.H + 1)
        return W1, W2

    def init_weights(self, rng, low=-1.0, high=1.0):
        return rng.uniform(low, high, self.D)

    def forward(self, w, X):
        # returns hidden acts A (N, H), outputs O (N, K)
        W1, W2 = self.unpack(w)
        A = sigmoid(add_bias(X) @ W1.T)
        O = softmax(add_bias(A) @ W2.T)
        return A, O

    def loss(self, w, X, T, lam):
        # mean CE over batch + (lam/2) * ||w * mask||^2
        A, O = self.forward(w, X)
        N = X.shape[0]
        ce = cross_entropy(O, T)
        reg = 0.5 * lam * np.sum((w * self.mask) ** 2)
        return ce + reg

    def gradient(self, w, X, T, lam):
        W1, W2 = self.unpack(w)
        A, O = self.forward(w, X)
        N = X.shape[0]
        d_out = (O - T) / N  # output delta
        g2 = d_out.T @ add_bias(A)  # (K, H+1)
        d_hid = (d_out @ W2[:, :-1]) * A * \
            (1 - A)       # drop bias col, (N, H)
        g1 = d_hid.T @ add_bias(X)                       # (H, I+1)
        grad = np.concatenate([g1.ravel(), g2.ravel()]) + lam * w * self.mask
        loss = cross_entropy(O, T) + 0.5 * lam * np.sum((w * self.mask) ** 2)
        return loss, grad

    def swarm_loss(self, P, X, T, lam):
        # loss of every particle on the same batch
        return np.array([self.loss(p, X, T, lam) for p in P])

    def evaluate(self, w, data, split="test"):
        # final measures, split = "test" (final runs) or "val" (tuning)
        # CE here without penalty
        _, O_train = self.forward(w, data["X_train"])
        n_train = data["X_train"].shape[0]
        train_ce = cross_entropy(O_train, data["T_train"])
        train_pred = O_train.argmax(axis=1)
        train_acc = accuracy_score(data["y_train"], train_pred)

        X_sp, T_sp, y_sp = data[f"X_{split}"], data[f"T_{split}"], data[f"y_{split}"]
        _, O_sp = self.forward(w, X_sp)  # forward pass on chosen split
        # cross-entropy for the chosen split
        ce = cross_entropy(O_sp, T_sp)
        pred = O_sp.argmax(axis=1)
        acc = accuracy_score(y_sp, pred)
        f1 = f1_score(y_sp, pred, average="macro")

        return {
            "train_ce": train_ce, "ce": ce,
            "train_acc": train_acc, "acc": acc,
            "f1": f1, "gen_factor": ce / max(train_ce, EPS),
        }


def log_point(net, w, data, lam, used):
    # one point for the convergence plots, not charged to budget
    train_loss = net.loss(w, data["X_train"], data["T_train"], lam)
    val_ce = cross_entropy(net.forward(w, data["X_val"])[1], data["T_val"])
    return {"patterns": used, "train_loss": train_loss, "val_ce": val_ce}


def check_gradient(net, X, T, lam, rng, h=1e-6):
    # backprop vs central differences on 10 random weights, want < 1e-6
    w = net.init_weights(rng)
    _, g = net.gradient(w, X, T, lam)
    worst = 0.0
    for j in rng.choice(net.D, 10, replace=False):
        e = np.zeros(net.D)
        e[j] = h
        num = (net.loss(w + e, X, T, lam) -
               net.loss(w - e, X, T, lam)) / (2 * h)  # central difference approximation
        # relative error for this weight
        worst = max(worst, abs(num - g[j]) / (abs(num) + abs(g[j]) + 1e-12))
    return worst


if __name__ == "__main__":
    from nnpso.data import load_data

    rng = np.random.default_rng(0)
    d = load_data("wine", seed=0)
    net = Network(d["X_train"].shape[1], 20, d["T_train"].shape[1])
    Xb, Tb = d["X_train"][:32], d["T_train"][:32]

    w = net.init_weights(rng)
    A, O = net.forward(w, Xb)
    print("shapes A, O:", A.shape, O.shape)  # (32, 20), (32, 3)
    print("softmax rows sum to 1:", np.allclose(O.sum(axis=1), 1))
    print("mask: weights, biases:", int(net.mask.sum()),
          int(net.D - net.mask.sum()))  # 320, 23
    # roughly 1 to 2 for random weights
    print("loss:", net.loss(w, Xb, Tb, lam=1e-3))
    print("grad check:", check_gradient(
        net, Xb, Tb, lam=1e-3, rng=rng))  # want < 1e-6
