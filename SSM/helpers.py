import numpy as np

def sigmoid(x):
    """
    Applies the sigmoid function to map values between 0 and 1.

    Used to turn model outputs (logits) into probabilities.
    """
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(logits, targets, eps=1e-8):
    """
    Calculates binary cross-entropy loss between predictions and targets.

    Converts logits to probabilities, then compares to true labels. Lower loss means better predictions.

    logits:  (T, 1)
    targets: (T, 1) in {0,1}
    """
    probs = sigmoid(logits)
    return -np.mean(
        targets * np.log(probs + eps) +
        (1.0 - targets) * np.log(1.0 - probs + eps)
    )
