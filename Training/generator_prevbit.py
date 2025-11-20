"""
prev_bit_data.py

Generate training and test data for the previous-bit copy task.

Task:
  Given a binary input sequence u_0, ..., u_{T-1},
  the target at time t is the previous input:

      target[0] = u[0]        (convention)
      target[t] = u[t-1]      for t >= 1

"""

import numpy as np


def make_prev_bit_example(T: int, rng=None):
  
    if rng is None:
        rng = np.random.default_rng()

    bits = rng.integers(0, 2, size=T)   # random 0/1 sequence of length T
    u = bits.astype(float).reshape(T, 1)

    # previous-bit copy:
    # target[0] = u[0], then shift
    target = np.vstack([u[0:1], u[:-1]])

    return u, target


def make_prev_bit_dataset(n_samples: int, T: int, rng=None):
   
    if rng is None:
        rng = np.random.default_rng()

    X = np.zeros((n_samples, T, 1), dtype=float)
    Y = np.zeros((n_samples, T, 1), dtype=float)

    for i in range(n_samples):
        u, target = make_prev_bit_example(T, rng)
        X[i] = u
        Y[i] = target

    return X, Y


def make_train_test(
    n_train: int = 10000,
    n_test: int = 1000,
    T: int = 10,
    seed: int = 0,
):
    """
    Convenience function to generate train and test sets.

    Returns:
        x_train, y_train, x_test, y_test
        where each is shaped (N, T, 1)
    """
    rng = np.random.default_rng(seed)

    x_train, y_train = make_prev_bit_dataset(n_train, T, rng)
    x_test, y_test   = make_prev_bit_dataset(n_test, T, rng)

    return x_train, y_train, x_test, y_test


