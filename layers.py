# layers.py
import numpy as np
from numpy import ndarray
from .base import Layer, Operation
from .operations import WeightMultiply, BiasAdd
from .activation import Sigmoid
from .utils import assert_same_shape

class Dense(Layer):
    '''
    A fully connected (dense) layer which inherits from "Layer".
    Uses He initialization suitable for ReLU activations.
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()):
        super().__init__(neurons)
        self.activation = activation
        self.seed = None  # Initialized to None as default

    def _setup_layer(self, input_: ndarray) -> None:
        '''
        Initialize layer parameters (weights and biases).
        Uses He initialization: variance = 2.0 / input_dim
        This is optimal for ReLU-like activations.
        
        Args:
            input_: Input batch with shape (batch_size, input_dim)
        '''
        # Check if seed was explicitly set (including 0!)
        if getattr(self, "seed", None) is not None:
            np.random.seed(self.seed)

        # He initialization (suitable for ReLU and variants)
        in_dim = input_.shape[1]
        out_dim = self.neurons
        scale = np.sqrt(2.0 / in_dim)  # He initialization

        self.params = []
        self.params.append(np.random.standard_normal((in_dim, out_dim)) * scale)  # Weights
        self.params.append(np.zeros((1, out_dim)))  # Bias (zero initialization is fine)

        # Weight decay applies to the weight matrix but never to the bias.
        self.param_decay = [True, False]

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]
        return None


class BatchNorm(Layer):
    '''
    Batch Normalization layer with learnable scale (gamma) and shift (beta).

    During training, normalizes activations using the mean/variance of the
    current batch (population variance, divided by B, as in PyTorch with
    affine=True) and updates running statistics via an exponential moving
    average. During inference, uses the accumulated running statistics.

    Reference: Ioffe & Szegedy (2015), "Batch Normalization".
    '''
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        '''
        Initialize BatchNorm.

        Args:
            num_features: Number of input features (size of the last axis).
            momentum: EMA coefficient for the running statistics (default 0.1,
                      matching PyTorch's default).
            eps: Small constant for numerical stability (default 1e-5).
        '''
        super().__init__(num_features)
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.is_training = True

        # Learnable parameters (shape (1, num_features) for broadcasting).
        self.params = [np.ones((1, num_features)), np.zeros((1, num_features))]
        self.param_grads = [np.zeros((1, num_features)), np.zeros((1, num_features))]
        # Neither gamma nor beta are subject to weight decay.
        self.param_decay = [False, False]

        # Running statistics (not learnable).
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # Forward cache for the backward pass.
        self._x_hat = None
        self._var_eps = None
        self._x_center = None

        # No lazy setup needed; parameters are created up front.
        self.first = False

    @property
    def gamma(self) -> ndarray:
        return self.params[0]

    @property
    def beta(self) -> ndarray:
        return self.params[1]

    def forward(self, input_: ndarray) -> ndarray:
        '''Normalize the batch (training) or use running stats (inference).'''
        if input_.shape[1] != self.num_features:
            raise ValueError(
                f"BatchNorm: attese {self.num_features} features, "
                f"ricevute {input_.shape[1]}"
            )
        self.input_ = input_

        if self.is_training:
            mean = input_.mean(axis=0, keepdims=True)
            var = input_.var(axis=0, keepdims=True)  # population variance (/B)
            var_eps = var + self.eps

            x_center = input_ - mean
            x_hat = x_center / np.sqrt(var_eps)

            # Update running statistics (EMA).
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var

            self._x_hat = x_hat
            self._var_eps = var_eps
            self._x_center = x_center

            self.output = self.gamma * x_hat + self.beta
        else:
            x_hat = (input_ - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.output = self.gamma * x_hat + self.beta

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''Analytic gradient through the normalization.'''
        assert_same_shape(self.output, output_grad)

        B = output_grad.shape[0]
        x_hat = self._x_hat
        var_eps = self._var_eps
        x_center = self._x_center

        grad_gamma = np.sum(output_grad * x_hat, axis=0, keepdims=True)
        grad_beta = np.sum(output_grad, axis=0, keepdims=True)

        dx_hat = output_grad * self.gamma
        dvar = np.sum(dx_hat * x_center * -0.5 * var_eps ** (-1.5), axis=0, keepdims=True)
        dmean = (np.sum(dx_hat * (-1.0 / np.sqrt(var_eps)), axis=0, keepdims=True)
                 + dvar * (-2.0 / B) * np.sum(x_center, axis=0, keepdims=True))
        dx = (dx_hat / np.sqrt(var_eps)
              + dvar * 2.0 * x_center / B
              + dmean / B)

        self.param_grads = [grad_gamma, grad_beta]
        assert_same_shape(self.input_, dx)
        return dx
