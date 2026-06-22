import numpy as np

class Optimizer:
    '''
    Base class for optimizers.
    Optimizers update network parameters based on their gradients.

    All optimizers support optional L2 regularization (weight decay): if
    ``weight_decay > 0`` the effective gradient becomes ``g + weight_decay * w``
    for the weight matrices. Biases and BatchNorm parameters are never decayed,
    matching standard practice.
    '''
    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        '''
        Initialize Optimizer.

        Args:
            lr: Learning rate (default 0.01). Controls magnitude of parameter updates.
            weight_decay: L2 penalty coefficient (default 0.0 = disabled).
        '''
        assert lr > 0, f"Learning rate must be positive, got {lr}"
        assert weight_decay >= 0, f"weight_decay must be non-negative, got {weight_decay}"
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        '''
        Perform one optimization step to update all parameters.
        Must be called after backward() to apply accumulated gradients.

        Subclasses must override this method.
        '''
        raise NotImplementedError()

    def _iter_params(self):
        '''
        Yield (index, param, effective_grad) for every learnable parameter.
        Applies weight decay to the eligible parameters (weights, not biases).
        '''
        params = self.net.params()
        grads = self.net.param_grads()
        decays = self.net.param_decays()
        for i, (param, grad, decay) in enumerate(zip(params, grads, decays)):
            if self.weight_decay > 0.0 and decay:
                grad = grad + self.weight_decay * param
            yield i, param, grad

class SGD(Optimizer):
    '''
    Stochastic Gradient Descent (SGD) optimizer.
    Simple parameter update: param = param - lr * gradient

    Good for: Basic training, often requires careful learning rate tuning.
    '''
    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        '''
        Initialize SGD.

        Args:
            lr: Learning rate (default 0.01)
            weight_decay: L2 penalty coefficient (default 0.0)
        '''
        super().__init__(lr, weight_decay)

    def step(self):
        '''
        Update all network parameters by moving them in direction opposite to gradient.
        Called after each batch.
        '''
        for _, param, grad in self._iter_params():
            param -= self.lr * grad

class SGDMomentum(Optimizer):
    '''
    SGD with Momentum optimizer.
    Maintains velocity (momentum) of parameter updates to accelerate convergence
    and reduce oscillations.

    Update rule:
        v = momentum * v + gradient
        param = param - lr * v

    Good for: Faster convergence, smoother training curves, less sensitive to learning rate.
    '''
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        '''
        Initialize SGDMomentum.

        Args:
            lr: Learning rate (default 0.01)
            momentum: Momentum coefficient (default 0.9). Higher = more "inertia".
                      Typical range: [0.8, 0.99]
            weight_decay: L2 penalty coefficient (default 0.0)
        '''
        super().__init__(lr, weight_decay)
        assert 0 <= momentum < 1, f"Momentum must be in [0, 1), got {momentum}"
        self.momentum = momentum
        self.velocities = None  # Will be initialized on first step()

    def step(self):
        '''
        Update parameters using momentum-accelerated gradients.
        Velocities are initialized on first call and then updated each step.

        The velocity accumulates gradients over time, allowing the optimizer
        to "roll downhill" faster in consistent directions.
        '''
        if self.velocities is None:
            # First step: initialize velocities with zeros (same shape as parameters)
            self.velocities = [np.zeros_like(p) for p in self.net.params()]

        for i, param, grad in self._iter_params():
            # Update velocity: accumulate gradient with momentum
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            # Update parameter: move in direction of accumulated momentum
            param -= self.lr * self.velocities[i]

class Adam(Optimizer):
    '''
    Adam optimizer (Kingma & Ba, 2015).
    Combines momentum (first moment) with per-parameter adaptive learning rates
    (second moment). Robust default for deep networks and sparse gradients.

    Update rule (per parameter):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    '''
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        '''
        Initialize Adam.

        Args:
            lr: Learning rate (default 0.001)
            beta1: Exponential decay rate for the first moment (default 0.9)
            beta2: Exponential decay rate for the second moment (default 0.999)
            eps: Term added for numerical stability (default 1e-8)
            weight_decay: L2 penalty coefficient (default 0.0)
        '''
        super().__init__(lr, weight_decay)
        assert 0 <= beta1 < 1, f"beta1 must be in [0, 1), got {beta1}"
        assert 0 <= beta2 < 1, f"beta2 must be in [0, 1), got {beta2}"
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # First moment estimates
        self.v = None  # Second moment estimates
        self.t = 0     # Timestep

    def step(self):
        '''
        Update parameters using bias-corrected first and second moment estimates.
        Moment buffers are initialized on the first call.
        '''
        if self.m is None:
            self.m = [np.zeros_like(p) for p in self.net.params()]
            self.v = [np.zeros_like(p) for p in self.net.params()]

        self.t += 1
        bias_corr1 = 1.0 - self.beta1 ** self.t
        bias_corr2 = 1.0 - self.beta2 ** self.t

        for i, param, grad in self._iter_params():
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad
            m_hat = self.m[i] / bias_corr1
            v_hat = self.v[i] / bias_corr2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
