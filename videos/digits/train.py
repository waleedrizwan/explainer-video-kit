"""Real training for the digits video: one decision tree vs a small neural network on
scikit-learn's 8x8 handwritten digits. Everything the video shows comes from results.json.

    python train.py          -> results.json

Pixels are numbered 0..63, row by row from the top left, with brightness 0..16.
Network: 64 inputs -> 16 hidden (sigmoid) -> 10 outputs (softmax), plain numpy, mini-batch SGD."""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
SEED = 0
HIDDEN = 16
EPOCHS = 60
SNAP_EPOCHS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60]

digits = load_digits()
X_all, y_all = digits.data, digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, random_state=SEED, stratify=y_all)


# ---------------------------------------------------------------- trees
small = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_tr, y_tr)
full = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)
small_acc = float((small.predict(X_te) == y_te).mean())
full_acc = float((full.predict(X_te) == y_te).mean())

t = small.tree_


def node(i):
    if t.children_left[i] == -1:
        counts = t.value[i][0]
        return {"leaf": True, "pred": int(np.argmax(counts)),
                "purity": float(counts.max() / counts.sum()), "n": int(t.n_node_samples[i])}
    return {"leaf": False, "pixel": int(t.feature[i]), "thr": float(t.threshold[i]),
            "n": int(t.n_node_samples[i]),
            # sklearn goes left when pixel <= thr, so "brighter than thr?" yes -> right
            "no": node(t.children_left[i]), "yes": node(t.children_right[i])}


def rules(i=0, depth=0, out=None):
    out = [] if out is None else out
    if t.children_left[i] != -1 and depth < 2:
        out.append({"pixel": int(t.feature[i]), "thr": float(t.threshold[i]), "depth": depth})
        rules(t.children_left[i], depth + 1, out)
        rules(t.children_right[i], depth + 1, out)
    return out


def tree_path(x):
    """Pixels asked, answers given and the leaf's guess for one digit, using the small tree."""
    i, steps = 0, []
    while t.children_left[i] != -1:
        p, thr = int(t.feature[i]), float(t.threshold[i])
        yes = bool(x[p] > thr)
        steps.append({"pixel": p, "thr": thr, "value": int(x[p]), "yes": yes})
        i = t.children_right[i] if yes else t.children_left[i]
    return steps, int(np.argmax(t.value[i][0]))


# ---------------------------------------------------------------- network
def sig(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def forward(P, X):
    h = sig((X / 16.0) @ P["W1"] + P["b1"])
    return h, softmax(h @ P["W2"] + P["b2"])


def acc(P, X, y):
    return float((forward(P, X)[1].argmax(1) == y).mean())


rng = np.random.default_rng(SEED)
P = {"W1": rng.normal(0, 0.1, (64, HIDDEN)), "b1": np.zeros(HIDDEN),
     "W2": rng.normal(0, 0.3, (HIDDEN, 10)), "b2": np.zeros(10)}
LR, BATCH, L2 = 0.5, 32, 3e-3   # weight decay keeps the 8x8 weight maps smooth enough to read
Y_tr = np.eye(10)[y_tr]
curve, snaps = [], []
for ep in range(EPOCHS + 1):
    if ep > 0:
        order = rng.permutation(len(X_tr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            xb = X_tr[b] / 16.0
            h, p = forward(P, X_tr[b])
            d2 = (p - Y_tr[b]) / len(b)
            d1 = (d2 @ P["W2"].T) * h * (1 - h)
            P["W2"] -= LR * (h.T @ d2 + L2 * P["W2"]); P["b2"] -= LR * d2.sum(0)
            P["W1"] -= LR * (xb.T @ d1 + L2 * P["W1"]); P["b1"] -= LR * d1.sum(0)
    curve.append({"epoch": ep, "train": acc(P, X_tr, y_tr), "test": acc(P, X_te, y_te)})
    if ep in SNAP_EPOCHS:
        snaps.append({"epoch": ep, "W1": P["W1"].T.round(4).tolist(), "test": curve[-1]["test"]})
net_acc = curve[-1]["test"]
h_te, p_te = forward(P, X_te)
net_pred = p_te.argmax(1)
tree_pred = full.predict(X_te)


# ---------------------------------------------------------------- examples
def digit_record(i):
    x = X_te[i]
    steps, small_guess = tree_path(x)
    return {"test_index": int(i), "label": int(y_te[i]), "pixels": x.astype(int).tolist(),
            "tree_small_path": steps, "tree_small_guess": small_guess, "tree_full_guess": int(tree_pred[i]),
            "net_guess": int(net_pred[i]), "net_probs": p_te[i].round(4).tolist(),
            "hidden": h_te[i].round(4).tolist()}


# 3 digits to walk down the small tree / through the network: confidently-right, distinct labels
right_both = [i for i in range(len(y_te)) if net_pred[i] == y_te[i] and tree_path(X_te[i])[1] == y_te[i]]
walk = []
for i in sorted(right_both, key=lambda i: -p_te[i].max()):
    if y_te[i] not in [y_te[j] for j in walk]:
        walk.append(i)
    if len(walk) == 3:
        break
# one the small tree gets wrong, to walk down too
small_wrong = next(i for i in range(len(y_te)) if tree_path(X_te[i])[1] != y_te[i] and net_pred[i] == y_te[i])
tree_misses_net_gets = [i for i in range(len(y_te)) if tree_pred[i] != y_te[i] and net_pred[i] == y_te[i]]
net_misses_tree_gets = [i for i in range(len(y_te)) if net_pred[i] != y_te[i] and tree_pred[i] == y_te[i]]
both_miss = [i for i in range(len(y_te)) if net_pred[i] != y_te[i] and tree_pred[i] != y_te[i]]

# two 9s from training data that disagree about the root pixel: same digit, opposite answers
root_p, root_thr = int(t.feature[0]), float(t.threshold[0])
nines = [i for i in range(len(y_tr)) if y_tr[i] == 9]
nine_dark = next(i for i in nines if X_tr[i][root_p] <= root_thr)
nine_lit = next(i for i in nines if X_tr[i][root_p] >= 12)
root_by_digit = {int(d): float((X_tr[y_tr == d][:, root_p] > root_thr).mean()) for d in range(10)}

# hidden neurons ranked by how much the output layer relies on them
importance = np.abs(P["W2"]).sum(1)

res = {
    "seed": SEED, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "tree_depth3_acc": small_acc, "tree_full_acc": full_acc, "tree_full_depth": int(full.get_depth()),
    "tree_full_leaves": int(full.get_n_leaves()), "net_acc": net_acc, "hidden": HIDDEN, "epochs": EPOCHS,
    "tree_rules": rules(), "tree": node(0),
    "full_tree": {"left": full.tree_.children_left.tolist(), "right": full.tree_.children_right.tolist()},
    "root_pixel": root_p, "root_thr": root_thr, "root_yes_by_digit": root_by_digit,
    "hidden_weights": P["W1"].T.round(4).tolist(), "hidden_importance": importance.round(4).tolist(),
    "hidden_bias": P["b1"].round(4).tolist(), "out_weights": P["W2"].round(4).tolist(),
    "out_bias": P["b2"].round(4).tolist(),
    "curve": curve, "snaps": snaps,
    "walk": [digit_record(i) for i in walk], "small_tree_wrong": digit_record(small_wrong),
    "tree_misses_net_gets": [digit_record(i) for i in sorted(tree_misses_net_gets, key=lambda i: -p_te[i].max())[:6]],
    "n_tree_misses_net_gets": len(tree_misses_net_gets), "n_net_misses_tree_gets": len(net_misses_tree_gets),
    "n_both_miss": len(both_miss), "net_misses": [digit_record(i) for i in net_misses_tree_gets[:3]],
    "nine_dark": X_tr[nine_dark].astype(int).tolist(), "nine_lit": X_tr[nine_lit].astype(int).tolist(),
}
(HERE / "results.json").write_text(json.dumps(res))

print(f"train {len(y_tr)}  test {len(y_te)}")
print(f"tree depth 3: {small_acc:.3f}   full tree (depth {full.get_depth()}, {full.get_n_leaves()} leaves): {full_acc:.3f}")
print(f"network 64-{HIDDEN}-10 after {EPOCHS} epochs: {net_acc:.3f}")
print("curve:", " ".join(f"{c['epoch']}:{c['test']:.2f}" for c in curve if c["epoch"] in SNAP_EPOCHS))
print("rules:", [(r["pixel"], r["thr"]) for r in res["tree_rules"]])
print("root pixel brighter, by digit:", {d: round(v, 2) for d, v in root_by_digit.items()})
print("walk:", [(r["label"], r["tree_small_guess"], [(s["pixel"], s["value"], s["yes"]) for s in r["tree_small_path"]]) for r in res["walk"]])
print("small wrong:", res["small_tree_wrong"]["label"], "->", res["small_tree_wrong"]["tree_small_guess"])
print(f"tree misses / net gets: {len(tree_misses_net_gets)}  net misses / tree gets: {len(net_misses_tree_gets)}  both: {len(both_miss)}")
print("tree-miss examples:", [(r["label"], r["tree_full_guess"], round(max(r["net_probs"]), 2)) for r in res["tree_misses_net_gets"]])
print("importance order:", np.argsort(-importance).tolist())
