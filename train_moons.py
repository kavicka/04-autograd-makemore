"""Train an MLP built only from our Value class to separate two interleaving half-moons."""

import json
import random

import numpy as np

from micrograd.engine import Value
from micrograd.nn import MLP


def make_moons(n: int = 100, noise: float = 0.1, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n)
    half = n // 2
    upper = np.c_[np.cos(t[:half]), np.sin(t[:half])]
    lower = np.c_[1 - np.cos(t[half:]), 0.5 - np.sin(t[half:])]
    X = np.r_[upper, lower] + rng.normal(0, noise, (n, 2))
    y = np.r_[np.ones(half), -np.ones(n - half)]  # labels +1 / -1
    return X, y


def loss_fn(model: MLP, X: np.ndarray, y: np.ndarray, alpha: float = 1e-4) -> tuple[Value, float]:
    scores = [model([Value(a), Value(b)]) for a, b in X]
    # hinge loss: zero once a point is on the right side with margin ≥ 1
    data_loss = sum((1 + -yi * s).relu() for yi, s in zip(y, scores)) / len(y)
    reg_loss = alpha * sum(p * p for p in model.parameters())  # L2 keeps weights small
    acc = float(np.mean([(s.data > 0) == (yi > 0) for yi, s in zip(y, scores)]))
    return data_loss + reg_loss, acc


def main() -> None:
    random.seed(1337)
    X, y = make_moons()
    model = MLP(2, [16, 16, 1])
    print(f"MLP(2 → 16 → 16 → 1) with {len(model.parameters())} parameters, {len(X)} points")

    history = []
    steps = 100
    for k in range(steps):
        loss, acc = loss_fn(model, X, y)
        model.zero_grad()
        loss.backward()
        lr = 1.0 - 0.9 * k / steps  # linear learning-rate decay
        for p in model.parameters():
            p.data -= lr * p.grad  # gradient descent: step against the gradient
        history.append({"step": k, "loss": loss.data, "acc": acc})
        if k % 10 == 0 or k == steps - 1:
            print(f"step {k:3d}  loss {loss.data:.4f}  accuracy {acc:.0%}")

    # decision boundary on a grid, evaluated with our own engine
    xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 60)
    ys = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 60)
    grid = [[model([Value(a), Value(b)]).data for a in xs] for b in ys]

    with open("results_moons.json", "w") as f:
        json.dump({"X": X.tolist(), "y": y.tolist(), "history": history, "grid_x": xs.tolist(), "grid_y": ys.tolist(), "grid": grid,
                   "n_params": len(model.parameters())}, f)


if __name__ == "__main__":
    main()
