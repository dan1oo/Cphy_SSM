import numpy as np

def hippo_legs(N: int) -> np.ndarray:
    """
    Construct the N x N HiPPO-LegS continuous-time operator.

    This is the operator used in S4 to approximate online Legendre
    projection. Here we use the standard dense definition:

        A_{n,m} =
            -sqrt((2n+1)(2m+1))   if n > m
            -(n+1)               if n = m
            0                     if n < m
    """
    A = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            if n > m:
                A[n, m] = -np.sqrt((2 * n + 1) * (2 * m + 1))
            elif n == m:
                A[n, m] = -(n + 1)
            # else: A[n, m] = 0
    return A


def discretize_bilinear(A_ct: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Bilinear (Tustin) discretization:

        A_d = (I - dt/2 * A_ct)^(-1) (I + dt/2 * A_ct)

    This is the transform used in S4 to go from continuous-time
    to discrete-time SSM parameters.
    """
    N = A_ct.shape[0]
    I = np.eye(N)
    left = I - 0.5 * dt * A_ct
    right = I + 0.5 * dt * A_ct
    A_d = np.linalg.inv(left) @ right
    return A_d