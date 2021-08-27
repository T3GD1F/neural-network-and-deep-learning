"""
activation_loss.py
~~~~~~~~~~

Collection of functions, which combine
activation function and loss.
"""

### --- IMPORTS --- ###
# Own Import

# Third-Party Import
import numpy as np


# Softmax + Categorical Cross Entropy
class Activation_Softmax_Loss_CategoricalCrossentropy(): 
    def backward(self, dvalues, y_true):
        """Backward Pass
        Calculates Gradient"""

        # Number of samples
        n_samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(n_samples), y_true] -= 1
        self.dinputs = self.dinputs / n_samples