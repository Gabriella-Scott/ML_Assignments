import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

LOADERS = {"iris": load_iris, "wine": load_wine, "cancer": load_breast_cancer}


def one_hot(y, n_classes):
    # int labels (N,) -> 0/1 matrix (N, K)
    y_one_hot = np.zeros((y.shape[0], n_classes), dtype=int)
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot


def load_data(name, seed, split=(0.6, 0.2, 0.2)):
    # returns dict: X_/T_/y_ for train, val, test
    X, y = LOADERS[name](return_X_y=True)
    n_classes = y.max() + 1

    # train vs rest, rest -> val/test
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=split[0], stratify=y, random_state=seed
    )
    val_size = split[1] / (split[1] + split[2])
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=val_size, stratify=y_rest, random_state=seed
    )  # split rest into val/test

    # z-score with train mean/std only (no leakage into val/test)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # one-hot targets for softmax/CE
    T_train = one_hot(y_train, n_classes)
    T_val = one_hot(y_val, n_classes)
    T_test = one_hot(y_test, n_classes)

    return {
        "X_train": X_train, "T_train": T_train, "y_train": y_train,
        "X_val": X_val, "T_val": T_val, "y_val": y_val,
        "X_test": X_test, "T_test": T_test, "y_test": y_test,
    }


def batches(X, T, batch_size, rng):
    # endless generator of (X_batch, T_batch)
    n_samples = X.shape[0]
    if batch_size == "full":
        batch_size = n_samples  # whole train set evry time, static problem

    indices = np.arange(n_samples)
    while True:
        rng.shuffle(indices)  # shuffle indices for this epoch
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield X[batch_idx], T[batch_idx]


if __name__ == "__main__":
    # quick check: python -m nnpso.data
    rng = np.random.default_rng(0)
    for name in LOADERS:
        d = load_data(name, seed=0)
        print(f"Dataset: {name}")
        print(f"  Train size: {d['X_train'].shape[0]}")
        print(f"  Val size: {d['X_val'].shape[0]}")
        print(f"  Test size: {d['X_test'].shape[0]}")
        print(
            f"  Train mean (max abs): {abs(d['X_train'].mean(axis=0)).max():.4f}")
        print(
            f"  Train std (min, max): {d['X_train'].std(axis=0).min():.2f}, {d['X_train'].std(axis=0).max():.2f}")
        X_batch, T_batch = next(
            batches(d['X_train'], d['T_train'], batch_size=4, rng=rng))
        print(
            f"  One batch X shape: {X_batch.shape}, T shape: {T_batch.shape}")
        print()
