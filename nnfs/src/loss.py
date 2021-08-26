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
    def regularization_loss(self, layer):
        """Calculates the Loss of the Regularization"""
        
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss
        

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

        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Mask for categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                                    range(samples), y_true]
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

        samples = len(dvalues)
        labels = len(dvalues[0])

        # Mask for sparse -> one hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        """Forward Pass
        Calculates Binary Cross Entropy"""

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses


    def backward(self, dvalues, y_true):
        """Backward Pass
        Calculates Gradient of Inputs"""

        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


# Mean Squared Error Loss
class Loss_MeanSquaredError(Loss):      # L2 loss
    def forward(self, y_pred, y_true):
        """Forward Pass
        Calculates mean Squared Error ||*||_2^2"""

        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses


    def backward(self, dvalues, y_true):
        """Backward Pass
        Calculates Gradient"""

        n_samples = len(dvalues)
        n_outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / n_outputs
        self.dinputs = self.dinputs / n_samples

    
# Mean Absolute Error Loss
"""L1 is more robust against outliers than L2.
But L1 penalizes the error linearly. Therefor often L2 is used."""
class Loss_MeanAbsoluteError(Loss):      # L1 loss
    def forward(self, y_pred, y_true):
        """Forward Pass
        Calculates Mean Absolute Error ||*||_1"""

        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses


    def backward(self, dvalues, y_true):
        """Backward Pass
        Calculates Gradient"""

        n_samples = len(dvalues)
        n_outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / n_outputs
        self.dinputs = self.dinputs / n_samples