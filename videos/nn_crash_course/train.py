"""Data + real training for the crash-course video (Google MLCC, Neural networks module).
Circle data (blue inside, orange ring) -> 2 inputs -> 3 ReLU hidden -> 1 sigmoid output.
Writes results.json: points, weights, boundary snapshots during training, and one worked inference."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
rng = np.random.default_rng(7)


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def circle_data(n=90):
    inner_r = rng.uniform(0, 1.3, n // 2)
    outer_r = rng.uniform(2.1, 2.9, n // 2)
    r = np.concatenate([inner_r, outer_r])
    a = rng.uniform(0, 2 * np.pi, n)
    X = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])   # 1 = blue (inside)
    return X, y


def quadrant_data(n=60):
    X = rng.uniform(-2.8, 2.8, (n, 2))
    X = X[np.abs(X).min(1) > 0.35][:48]
    y = (X[:, 0] * X[:, 1] > 0).astype(float)                 # 1 = blue (top-right / bottom-left)
    return X, y


def forward(P, X):
    z1 = X @ P["W1"] + P["b1"]
    h = np.maximum(z1, 0)
    return z1, h, sig(h @ P["W2"] + P["b2"])


def train(X, y, seed, epochs=1500, lr=0.05, snap_at=()):
    r = np.random.default_rng(seed)
    P = {"W1": r.normal(0, 0.8, (2, 3)), "b1": r.normal(0, 0.3, 3),
         "W2": r.normal(0, 0.8, 3), "b2": np.zeros(())}
    snaps, losses = [], []
    for ep in range(epochs + 1):
        z1, h, p = forward(P, X)
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        losses.append(loss)
        if ep in snap_at:
            snaps.append((ep, {k: v.copy() for k, v in P.items()}, loss))
        # backprop: error at the output, pushed back through the weights
        d_out = (p - y) / len(y)
        gW2, gb2 = h.T @ d_out, d_out.sum()
        d_h = np.outer(d_out, P["W2"]) * (z1 > 0)
        gW1, gb1 = X.T @ d_h, d_h.sum(0)
        P["W1"] -= lr * 6 * gW1; P["b1"] -= lr * 6 * gb1
        P["W2"] -= lr * 6 * gW2; P["b2"] -= lr * 6 * gb2
    return P, losses, snaps


def grid(P, n=72, lim=3.2):
    xs = np.linspace(-lim, lim, n)
    gx, gy = np.meshgrid(xs, xs[::-1])            # row 0 = top
    return forward(P, np.stack([gx.ravel(), gy.ravel()], 1))[2].reshape(n, n)


def hidden_grid(P, j, n=72, lim=3.2):
    xs = np.linspace(-lim, lim, n)
    gx, gy = np.meshgrid(xs, xs[::-1])
    return forward(P, np.stack([gx.ravel(), gy.ravel()], 1))[1][:, j].reshape(n, n)


def loss_of(P, X, y):
    p = forward(P, X)[2]
    return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))


def acc(P, X, y):
    return float(np.mean((forward(P, X)[2] > 0.5) == y))


X, y = circle_data()
Xt, yt = circle_data(40)                 # rooms kept aside: never trained on
SNAP = (0, 60, 200, 600, 1500)
best = max((train(X, y, s, snap_at=SNAP) for s in range(40)), key=lambda t: (acc(t[0], X, y), -t[1][-1]))
P, losses, snaps = best

# quadrant data for the feature-cross scene
Xq, yq = quadrant_data()

# one worked inference for a point inside the circle
pt = np.array([1.0, 0.5])
z1, h, out = forward(P, pt[None])

R = {
    "circle": {"X": X.round(3).tolist(), "y": y.tolist()},
    "quad": {"X": Xq.round(3).tolist(), "y": yq.tolist()},
    "W1": P["W1"].round(2).tolist(), "b1": P["b1"].round(2).tolist(),
    "W2": P["W2"].round(2).tolist(), "b2": round(float(P["b2"]), 2),
    "accuracy": acc(P, X, y),
    "test": {"X": Xt.round(3).tolist(), "y": yt.tolist(), "loss": round(loss_of(P, Xt, yt), 3),
             "acc": acc(P, Xt, yt)},
    "hidden_grids": [hidden_grid(P, j).round(2).tolist() for j in range(3)],
    "final_grid": grid(P).round(3).tolist(),
    "snaps": [{"epoch": e, "loss": round(float(l), 3), "acc": acc(Q, X, y), "grid": grid(Q).round(3).tolist()}
              for e, Q, l in snaps],
    "curve": [round(float(l), 4) for l in losses[::10]],
    "infer": {"x": pt.tolist(), "z1": z1[0].round(2).tolist(), "h": h[0].round(2).tolist(),
              "out": round(float(out[0]), 2)},
}
(HERE / "results.json").write_text(json.dumps(R))
print("test", R["test"]["loss"], R["test"]["acc"], "accuracy", R["accuracy"], "loss", R["snaps"][-1]["loss"], "infer", R["infer"])
print("snaps", [(s["epoch"], s["loss"], s["acc"]) for s in R["snaps"]])
print("W1", R["W1"], "b1", R["b1"], "W2", R["W2"], "b2", R["b2"])
