"""
prev_bit_data.py

Generate training and test data for the "Previous Bit" prediction task.

Task:
  Given a binary input sequence u_0, ..., u_{T-1}, the goal is to predict the
  previous input at each time step.

      target[0] = u[0]        (just copy the first input)
      target[t] = u[t-1]      for t >= 1
"""

import numpy as np


def make_prev_bit_example(T: int, rng=None):
    """
    Generate one random input/output example.

    Parameters:
        T : length of sequence
        rng : random number generator (optional)

    Returns:
        u : input sequence, shape (T, 1)
        target : output (previous-bit) sequence, shape (T, 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate random binary sequence of length T
    bits = rng.integers(0, 2, size=T)
    u = bits.astype(float).reshape(T, 1)    # shape (T, 1)

    # Make target by shifting right, copying first bit
    target = np.vstack([u[0:1], u[:-1]])

    return u, target


def make_prev_bit_dataset(n_samples: int, T: int, rng=None):
    """
    Generate a dataset of previous-bit examples.

    Parameters:
        n_samples : number of sequences to generate
        T : sequence length
        rng : random number generator (optional)
    
    Returns:
        X : input sequences, shape (n_samples, T, 1)
        Y : target sequences, shape (n_samples, T, 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pre-allocate arrays
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
    Generate full training and test datasets.

    Parameters:
        n_train : number of training examples
        n_test : number of test examples
        T : sequence length
        seed : random seed for reproducibility

    Returns:
        x_train, y_train, x_test, y_test : all shaped (N, T, 1)
    """
    rng = np.random.default_rng(seed)

    x_train, y_train = make_prev_bit_dataset(n_train, T, rng)
    x_test, y_test   = make_prev_bit_dataset(n_test, T, rng)

    return x_train, y_train, x_test, y_test