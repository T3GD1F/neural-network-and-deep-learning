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


### --- CODE --- ###
#### Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """Contains the Number of Inputs and Neurons.
        Random initialisation of weights (gaussian) and
        biases = 0."""

        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        """Forward Pass
        Calculates the output of the Layer"""

        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases 

    def backward(self, dvalues):
        """Backward Pass
        Calculates gradient of weights, biases and input"""
        self.dweights = np.dot(self.input.T, dvalues)
        self.biases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
