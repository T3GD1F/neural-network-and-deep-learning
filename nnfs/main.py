"""
main.py
~~~~~~~~~~

Init and run the Neural Network.
"""

### --- IMPORTS --- ###
# Standard Import

# Own Import
from src import network

# Third-Party Import
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


### --- CODE --- ###
nnfs.init()
X, y = spiral_data(samples=100, classes=3)

# Init
dense1 = network.layers.Layer_Dense(2, 3)
activation1 = network.activation_functions.Activation_ReLU()

dense2 = network.layers.Layer_Dense(3, 3)
loss_activation = network.Activation_Softmax_Loss_CategoricalCrossentropy()


# Forward Pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)


# Backward Pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)