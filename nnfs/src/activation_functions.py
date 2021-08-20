"""
activation_functions.py
~~~~~~~~~~

Collection of activation functions.
Each function provides forward pass and backpropagation.
"""

### --- IMPORTS --- ###
# Standard Import

# Third-Party Import
import numpy as np


### --- CODE --- ###
# ReLU
class Activation_ReLU:
    def forward(self, inputs):
        """Forward Pass
        Calculates ReLU of Input"""

        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """Backward Pass
        Calculates Gradient for Inputs"""
        
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# SoftMax
class Activation_Softmax:
    def forward(self, inputs):
        """Forward Pass
        Calculates Softmax of Input
        (Normalize Input via Exp with 
        respect to difference rather magnitude)"""

        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs), axis=1, keepdims=True)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


    def backward(self, dvalues):
        """Backward Pass
        Calculates Gradient of Inputs"""
            
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dots(jacobian_matrix, single_dvalues)
