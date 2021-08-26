"""
layers.py
~~~~~~~~~~

Collection of different Loss functions.
Each loss provides Forward Pass and Backpropagation.
"""

### --- IMPORTS --- ###
# Standard Import

# Third-Party Import
import numpy as np


### --- CODE --- ###
# Common Loss Class
class Loss:
    def calculate(self, output, y):
        """Loss
        Calculates data and regularization loss
        for a given model"""
        
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

# Categorical Cross Entropy Loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        """Forward Pass
        Calculates Categorical Crossentropy Loss"""

        n_samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Mask for categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(n_samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        
        # Loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    def backward(self, dvalues, y_true):
        """Backward Pass
        Calculates Gradient"""

        n_samples = len(dvalues)
        n_labels = len(dvalues[0])

        # Mask for sparse -> one hot
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / n_samples
