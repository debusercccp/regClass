import typing
from numpy import ndarray
import numpy as np
from .utils import assert_same_shape

class Operation(object):
    '''
    Base class for an "operation" in a neural network.
    An Operation encapsulates a forward computation and its gradient.
    '''
    def __init__(self):
        pass

    def forward(self, input_: ndarray):
        '''
        Stores input in the self._input instance variable
        Calls the self._output() function.
        '''
        self.input_ = input_

        self.output = self._output()

        return self.output


    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        '''
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad


    def _output(self) -> ndarray:
        '''
        The _output method must be defined for each Operation
        '''
        raise NotImplementedError()


    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        The _input_grad method must be defined for each Operation
        '''
        raise NotImplementedError()
    
class ParamOperation(Operation):
    '''
    An Operation with learnable parameters.
    Examples: WeightMultiply (matrix multiplication with weight matrix),
              BiasAdd (addition of bias vector).
    
    Subclasses must implement both _input_grad() and _param_grad().
    '''

    def __init__(self, param: ndarray) -> ndarray:
        '''
        Initialize ParamOperation with learnable parameters.
        
        Args:
            param: The parameter array (e.g., weight matrix, bias vector).
                   This will be updated during training via gradient descent.
        '''
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Computes both input gradient and parameter gradient.
        Calls self._input_grad and self._param_grad.
        Validates shapes of all gradients.
        
        Args:
            output_grad: Gradient of loss w.r.t. output of this operation.
            
        Returns:
            input_grad: Gradient to propagate backward to previous layer.
        '''

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute gradient of loss w.r.t. the parameter.
        
        Every subclass of ParamOperation must implement _param_grad.
        This gradient is used by optimizers to update the parameters.
        '''
        raise NotImplementedError()
    
class Layer(object):
    '''
    A "layer" of neurons in a neural network.
    Layers contain operations (weight multiplication, bias addition, activation).
    '''

    def __init__(self,
                 neurons: int):
        '''
        Initialize a layer.
        
        Args:
            neurons: Number of output units in this layer.
        '''
        self.neurons = neurons
        self.first = True
        self.params: typing.List[ndarray] = []
        self.param_grads: typing.List[ndarray] = []
        # Parallel to self.params: True if weight decay (L2) should apply to that
        # parameter. Weights decay, biases and norm parameters do not.
        self.param_decay: typing.List[bool] = []
        self.operations: typing.List[Operation] = []
        self.output = None

    def _setup_layer(self, num_in: int) -> None:
        '''
        The _setup_layer function must be implemented for each layer.
        Called once on first forward pass to initialize parameters
        based on the actual input shape.
        '''
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        '''
        Passes input forward through a series of operations.
        On first call, initializes layer parameters via _setup_layer().
        ''' 
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Passes output_grad backward through a series of operations.
        Operations are traversed in reverse order.
        Checks appropriate shapes.
        '''

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        
        self._param_grads()

        return input_grad

    def _param_grads(self) -> ndarray:
        '''
        Extracts the param_grads from a layer's ParamOperation instances.
        '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:
        '''
        Extracts the params from a layer's ParamOperation instances.
        '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

class Loss(object):
    '''
    The loss function of a neural network.
    Measures the discrepancy between predictions and targets.
    '''

    def __init__(self):
        '''Initialize Loss'''
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        '''
        Computes the actual loss value.
        
        Args:
            prediction: Model output (shape depends on task).
            target: Ground truth labels (same shape as prediction).
            
        Returns:
            float: Scalar loss value.
        '''
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        '''
        Computes gradient of the loss value w.r.t. the predictions.
        This gradient is propagated backward through the network.
        
        Returns:
            ndarray: Gradient with same shape as prediction.
        '''
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        '''
        Every subclass of "Loss" must implement the _output function.
        '''
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        '''
        Every subclass of "Loss" must implement the _input_grad function.
        '''
        raise NotImplementedError()
