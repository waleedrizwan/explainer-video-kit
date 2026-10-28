"""The small worked examples, shared by narration.py (which states the numbers) and scene.py (which draws them).
Nothing here is from a real model: they are hand-sized numbers chosen so every step can be checked by hand."""
import numpy as np

# Counts of the word after "orange is" in a tiny made-up training text
ORANGE_NEXT = [("ripe", 6), ("cheerful", 3), ("sweet", 2), ("bright", 1)]
ORANGE_TOTAL = sum(c for _, c in ORANGE_NEXT)

# Self-attention for "it": 2-D vectors, dot-product scores, softmax weights
VEC = {"it": (1.0, 1.0), "animal": (1.4, 0.6), "street": (0.9, -0.4), "tired": (0.2, 0.8)}
OTHERS = ["animal", "street", "tired"]
SCORES = [float(np.dot(VEC["it"], VEC[w])) for w in OTHERS]           # 2.0, 0.5, 1.0
EXPS = [float(np.exp(s)) for s in SCORES]                              # 7.39, 1.65, 2.72
WEIGHTS = [e / sum(EXPS) for e in EXPS]                                # 0.63, 0.14, 0.23
BLEND = tuple(float(x) for x in sum(w * np.array(VEC[o]) for w, o in zip(WEIGHTS, OTHERS)))

# Vanishing gradient: a training signal multiplied by 0.5 at each of 10 steps back
SHRINK = [0.5 ** k for k in range(11)]
