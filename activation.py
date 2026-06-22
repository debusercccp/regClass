from .base import Operation
from numpy import ndarray
import numpy as np

class Sigmoid(Operation):
    '''
    Sigmoid activation function.
    Output range: (0, 1). Good for binary classification or gates.
    '''

    def __init__(self) -> None:
        '''Initialize Sigmoid'''
        super().__init__()

    def _output(self) -> ndarray:
        '''
        Compute output: 1 / (1 + exp(-x))
        '''
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad

class Linear(Operation):
    '''
    Identity activation function (no transformation).
    Used at output layer for regression tasks.
    '''

    def __init__(self) -> None:
        '''Initialize Linear'''        
        super().__init__()

    def _output(self) -> ndarray:
        '''Pass through unchanged'''
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Gradient passes through unchanged'''
        return output_grad

class Tanh(Operation):
    '''
    Hyperbolic tangent activation function.
    Output range: (-1, 1). Zero-centered, useful for hidden layers.
    '''

    def __init__(self) -> None:
        '''Initialize Tanh'''
        super().__init__()

    def _output(self) -> ndarray:
        '''Compute output: tanh(x)'''
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        d/dx tanh(x) = 1 - tanh(x)^2
        '''
        return output_grad * (1.0 - self.output ** 2)

class ReLU(Operation):
    '''
    Rectified Linear Unit activation.
    Output: max(0, x). Highly efficient for hidden layers.
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> ndarray:
        '''Compute ReLU: max(0, x)'''
        return np.maximum(0, self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Gradient is 1 if input > 0, else 0.
        '''
        mask = self.input_ >= 0
        return output_grad * mask

class LeakyReLU(Operation):
    '''
    Leaky ReLU activation function.
    Allows small negative gradient (alpha) when input < 0.
    Fixes the "dying ReLU" problem where neurons can become inactive.
    '''
    def __init__(self, alpha: float = 0.2) -> None:
        '''
        Initialize LeakyReLU.
        
        Args:
            alpha: Slope for negative inputs (default 0.2, i.e., 20% gradient leak).
        '''
        super().__init__()
        self.alpha = alpha

    def _output(self) -> np.ndarray:
        '''
        Compute output: max(alpha * x, x)
        '''
        return np.maximum(self.alpha * self.input_, self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Gradient is 1 if input >= 0, else alpha.
        '''
        mask = np.where(self.input_ >= 0, 1.0, self.alpha)
        return output_grad * mask

class Softmax(Operation):
    '''
    Softmax activation function.
    Converts logits to probability distribution: sum = 1, each value in [0, 1].
    Used at output layer for multi-class classification.
    
    Note: Softmax + CategoricalCrossEntropy are designed to work together.
    '''
    def __init__(self) -> None:
        '''Initialize Softmax'''
        super().__init__()

    def _output(self) -> ndarray:
        '''
        Compute softmax: exp(x - max(x)) / sum(exp(x - max(x)))
        Subtracting max for numerical stability (prevents overflow with exp).
        '''
        shifted = self.input_ - np.max(self.input_, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute Jacobian-based gradient.
        Formula: d/dx softmax(x) = p * (grad - sum(p * grad))
        where p is the softmax output.
        
        This is the general form. When paired with CategoricalCrossEntropy,
        the combined gradient simplifies to (p - target).
        '''
        dot = np.sum(output_grad * self.output, axis=1, keepdims=True)
        return self.output * (output_grad - dot)
