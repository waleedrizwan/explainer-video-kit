"""Data + real training for the first-principles video.
Network: 2 inputs (hours studied, hours slept) -> 2 hidden -> 1 output, sigmoid everywhere.
Inputs are used in raw hours so line equations read directly on the chart."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

NAMES = ["Ana", "Ben", "Cam", "Dev", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jo"]
DATA = np.array([
    [8, 7, 1],
    [2, 8, 0],
    [6, 5, 1],
    [9, 3, 0],
    [3, 3, 0],
    [7, 7, 1],
    [5, 6, 1],
    [8, 2, 0],
    [9, 8, 1],
    [3, 9, 0],
], dtype=float)
X, Y = DATA[:, :2], DATA[:, 2]
NEW = np.array([7.0, 4.0])


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def forward(P, x):
    h = sig(x @ P["W1"] + P["b1"])
    return h, sig(h @ P["W2"] + P["b2"])


def best_single_line_errors():
    """Brute-force the fewest mistakes any straight line can make."""
    best = 99
    for a in np.linspace(0, np.pi, 721):
        n = np.array([np.cos(a), np.sin(a)])
        proj = X @ n
        for c in np.concatenate([proj - 1e-6, proj + 1e-6]):
            for sgn in (1, -1):
                pred = (sgn * (proj - c) > 0).astype(int)
                best = min(best, int((pred != Y).sum()))
    return best


def train(seed, lr=0.05, epochs=20000, snap_every=50):
    rng = np.random.default_rng(seed)
    P = {"W1": rng.normal(0, 0.5, (2, 2)), "b1": rng.normal(0, 1, 2),
         "W2": rng.normal(0, 1.0, 2), "b2": np.array([rng.normal(0, 0.5)])}
    snaps, losses = [], []
    for e in range(epochs + 1):
        h, out = forward(P, X)
        loss = float(((out - Y) ** 2).sum())
        if e % snap_every == 0:
            snaps.append({k: v.tolist() for k, v in P.items()})
            losses.append(loss)
        if e == epochs:
            break
        d_out = 2 * (out - Y) * out * (1 - out)
        gW2 = h.T @ d_out; gb2 = d_out.sum()
        d_h = np.outer(d_out, P["W2"]) * h * (1 - h)
        gW1 = X.T @ d_h; gb1 = d_h.sum(0)
        P["W2"] -= lr * gW2; P["b2"] -= lr * gb2
        P["W1"] -= lr * gW1; P["b1"] -= lr * gb1
    return P, snaps, losses


if __name__ == "__main__":
    print("fewest mistakes by one straight line:", best_single_line_errors())
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    for s in seeds:
        P, snaps, losses = train(s)
        _, out = forward(P, X)
        _, nout = forward(P, NEW[None, :])
        W1 = P["W1"]
        print(f"seed {s:2d} loss {losses[0]:.2f}->{losses[-1]:.4f} correct {int(((out > 0.5) == Y).sum())}/10 "
              f"h1 w=({W1[0,0]:+.2f},{W1[1,0]:+.2f}) b={P['b1'][0]:+.2f} | h2 w=({W1[0,1]:+.2f},{W1[1,1]:+.2f}) b={P['b1'][1]:+.2f} "
              f"| out w={np.round(P['W2'],2)} b={P['b2'][0]:+.2f} | new {nout[0]:.2f}")


def save(seed=11):
    P, snaps, losses = train(seed)
    _, out = forward(P, X)
    _, nout = forward(P, NEW[None, :])
    h_new, _ = forward(P, NEW[None, :])
    res = {"names": NAMES, "data": DATA.tolist(), "new": NEW.tolist(), "snap_every": 50,
           "snaps": snaps, "losses": losses, "final_out": out.tolist(), "new_out": float(nout[0]),
           "new_hidden": h_new[0].tolist(), "best_line_errors": best_single_line_errors()}
    (HERE / "results.json").write_text(json.dumps(res))
    return res
