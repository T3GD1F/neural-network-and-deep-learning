"""
layers.py
~~~~~~~~~~

Collection of different layer types.
Each layer provides forward pass and backpropagation.
"""

### --- IMPORTS --- ###
# Standard Import

# Third-Party Import
import numpy as np
from numpy.lib.financial import rate


### --- CODE --- ###
# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                    weight_regularizer_l1=0, weight_regularizer_l2=0,
                    bias_regularizer_l1=0, bias_regularizer_l2=0):
        """Contains the Number of Inputs and Neurons.
        Random initialisation of weights (gaussian) and
        biases = 0."""

        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1   = bias_regularizer_l1
        self.bias_regularizer_l2   = bias_regularizer_l2


    def forward(self, inputs):
        """Forward Pass
        Calculates the output of the Layer"""

        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases


    def backward(self, dvalues):
        """Backward Pass
        Calculates gradient of weights, biases and input"""

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


# Dropout
class Layer_Dropout:
    def __init__(self, rate):
        """Stores Rate of Neurons not firing.
        Input: percentage neurons NOT firinng
        self.rate: success rate of Neurons firing"""

        self.rate = 1 - rate


    def forward(self, inputs):
        """Forward Pass
        Calculates the output of the Layer"""

        self.inputs = inputs 
        
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask


    def backward(self, dvalues):
        """Backward Pass
        Calculates gradient of weights, biases and input"""
        self.dinputs = dvalues * self.binary_mask
