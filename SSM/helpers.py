import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(logits, targets, eps=1e-8):
    """
    logits:  (T, 1)
    targets: (T, 1) in {0,1}
    """
    probs = sigmoid(logits)
    return -np.mean(
        targets * np.log(probs + eps) +
        (1.0 - targets) * np.log(1.0 - probs + eps)
    )
