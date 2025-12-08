import numpy as np
from SSM.hippo import hippo_legs, discretize_bilinear
from SSM.helpers import sigmoid, binary_cross_entropy

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    # numerically stable softmax
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class SimpleSSM:
    """
    Simple State Space Model.

    Discrete-time dynamics:

        x_{t+1} = A x_t + B u_t
        y_t     = C x_t

    where A is obtained from a HiPPO-LegS continuous-time operator
    and discretized with a bilinear transform.

    """

    def __init__(
        self,
        state_dim: int = 8,
        input_dim: int = 1,
        output_dim: int = 1,
        dt: float = 1.0,
        randomseed: int = 0,
        learn_A: bool = True,
    ):
        """
        
        state_dim:  size of state x_t
        input_dim:  size of input u_t (for copy task, usually 1)
        output_dim: size of output y_t (for copy task, usually 1)
        dt:        time step for discretization
        randomseed:      RNG seed
        learn_A:   if False, A is frozen (fixed HiPPO operator);
                       if True, A is updated by gradient descent.
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt = dt
        self.learn_A = learn_A

        rng = np.random.default_rng(randomseed)

        # Continuous-time HiPPO-LegS operator
        self.A_ct = hippo_legs(state_dim)

        # Discretize to get A (S4 uses a similar transform)
        self.A = discretize_bilinear(self.A_ct, dt=dt)

        # Random B and C (small scale)
        self.B = rng.normal(scale=0.1, size=(state_dim, input_dim))
        self.C = rng.normal(scale=0.1, size=(output_dim, state_dim))

    

    def forward(self, u_seq: np.ndarray):
        """
        Run the SSM forward over a sequence.

        Args:
            u_seq: (T, input_dim) array of inputs

        Returns:
            logits: (T, output_dim) array (before sigmoid)
            xs:     (T+1, state_dim) states (x_0, x_1, ..., x_T)
        """
        T, in_dim = u_seq.shape
        assert in_dim == self.input_dim

        N = self.state_dim
        x = np.zeros(N)
        xs = np.zeros((T + 1, N))
        logits = np.zeros((T, self.output_dim))

        xs[0] = x

        for t in range(T):
            # x_{t+1} = A x_t + B u_t
            u_t = u_seq[t]
            x = self.A @ x + (self.B @ u_t)
            xs[t + 1] = x

            # y_t = C x_t (here using x_{t+1} as the current state)
            logits[t] = self.C @ x

        return logits, xs


    def loss_and_grads_softmax(self, u_seq: np.ndarray, target_seq: np.ndarray):
        """
        Multi-class (0..9) cross-entropy loss and gradients via BPTT.

        Args:
            u_seq:      (T, input_dim)        input sequence
            target_seq: (T,) or (T,1) ints in {0,...,9}

        Returns:
            loss:  scalar
            grads: (dA, dB, dC)
        """
        logits, xs = self.forward(u_seq)       # logits: (T, output_dim)
        T, num_classes = logits.shape

        # ensure 1D integer labels of length T
        y = target_seq.reshape(-1).astype(int)
        assert y.shape[0] == T

        # softmax probabilities
        probs = softmax(logits, axis=1)        # (T, num_classes)

        # cross-entropy loss: -log p(correct class)
        eps = 1e-8
        correct_log_probs = -np.log(probs[np.arange(T), y] + eps)
        loss = np.mean(correct_log_probs)

        # initialize gradients
        dA = np.zeros_like(self.A)
        dB = np.zeros_like(self.B)
        dC = np.zeros_like(self.C)

        N = self.state_dim
        dx_next = np.zeros(N)

        # BPTT
        for t in reversed(range(T)):
            x_t   = xs[t + 1]   # state after seeing u_t
            x_prev = xs[t]      # previous state
            u_t   = u_seq[t]

            # dL/dlogit for softmax + CE:
            # probs[t] - one_hot(y_t)
            dL_dlogit = probs[t].copy()
            dL_dlogit[y[t]] -= 1.0   # (num_classes,)

            # gradient wrt C: y_t = C x_t
            dC += dL_dlogit.reshape(-1, 1) @ x_t.reshape(1, -1)

            # gradient wrt x_t (current state), from output and future
            dx = self.C.T @ dL_dlogit + dx_next   # (N,)

            # x_t = A x_{t-1} + B u_t
            dA += np.outer(dx, x_prev)
            dB += np.outer(dx, u_t)

            # propagate to previous state
            dx_next = self.A.T @ dx

        # average over time
        dA /= T
        dB /= T
        dC /= T

        if not self.learn_A:
            dA[:] = 0.0

        return loss, (dA, dB, dC)



    def step(self, grads, lr: float = 1e-2, clip: float | None = 1.0):
        """
        Gradient descent step.

        Args:
            grads: (dA, dB, dC) from loss_and_grads
            lr:    learning rate
            clip:  optional elementwise gradient clipping bound
        """
        dA, dB, dC = grads

        if clip is not None:
            np.clip(dA, -clip, clip, out=dA)
            np.clip(dB, -clip, clip, out=dB)
            np.clip(dC, -clip, clip, out=dC)

        # Update parameters
        if self.learn_A:
            self.A -= lr * dA
        self.B -= lr * dB
        self.C -= lr * dC
