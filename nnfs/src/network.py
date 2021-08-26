"""
network.py
~~~~~~~~~~

Collection of different layer types
and activation functions.
Also includes different learning algorithms
"""

### --- IMPORTS --- ###
# Standard Import

# Own Import
from src import layers
from src import activation_functions
from src import loss
from src import optimizers

# Third-Party Import
import numpy as np


### --- CODE --- ###
# Softmax + Categorical Cross Entropy
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        """Softmax Classifier
        Combination of Softmax and Cross Entropy"""
        
        self.activation = activation_functions.Activation_Softmax()
        self.loss = loss.Loss_CategoricalCrossentropy()
    

    def forward(self, inputs, y_true):
        """Forward Pass
        Calculates Forward via Softmax and CrossEntropy"""
        
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)
    

    def backward(self, dvalues, y_true):
        """Backward Pass
        Calculates Gradient"""

        # Number of samples
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
