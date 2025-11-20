import numpy as np
from SSM.hippo import hippo_legs, discretize_bilinear
from SSM.helpers import sigmoid, binary_cross_entropy




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
        A_ct = hippo_legs(state_dim)

        # Discretize to get A (S4 uses a similar transform)
        self.A = discretize_bilinear(A_ct, dt=dt)

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


    def loss_and_grads(self, u_seq: np.ndarray, target_seq: np.ndarray):
        """
        Compute loss and gradients via BPTT for a single sequence.

        Args:
            u_seq:      (T, input_dim)    input sequence
            target_seq: (T, output_dim)   target sequence in {0,1}

        Returns:
            loss:  scalar
            grads: (dA, dB, dC) where each is the same shape as the parameter
        """
        logits, xs = self.forward(u_seq)       # (T, out_dim), (T+1, N)
        T = u_seq.shape[0]

        # binary cross-entropy
        probs = sigmoid(logits)                # (T, out_dim)
        eps = 1e-8
        loss = -np.mean(
            target_seq * np.log(probs + eps) +
            (1.0 - target_seq) * np.log(1.0 - probs + eps)
        )

        # gradients
        dA = np.zeros_like(self.A)
        dB = np.zeros_like(self.B)
        dC = np.zeros_like(self.C)

        N = self.state_dim
        dx_next = np.zeros(N)  # gradient wrt x_{t} coming from future

        # BPTT
        for t in reversed(range(T)):
            x_t = xs[t + 1]       # state after seeing u_t
            x_prev = xs[t]        # previous state x_{t}
            u_t = u_seq[t]

            # dL/dlogit = p - y for BCE with sigmoid
            dL_dlogit = (probs[t] - target_seq[t])  # (output_dim,)

            # gradient wrt C: y_t = C x_t
            dC += dL_dlogit.reshape(-1, 1) @ x_t.reshape(1, -1)

            # gradient wrt x_t (current state), from output and from future
            dx = self.C.T @ dL_dlogit.flatten() + dx_next  # (N,)

            # x_t = A x_{t-1} + B u_t
            dA += np.outer(dx, x_prev)
            dB += np.outer(dx, u_t)

            # propagate to previous state
            dx_next = self.A.T @ dx

        # average over time for stability
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
