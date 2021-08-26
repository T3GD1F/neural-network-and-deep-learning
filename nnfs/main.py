"""
main.py
~~~~~~~~~~

Init and run the Neural Network.
"""

### --- IMPORTS --- ###
# Standard Import

# Own Import
from src import network
from src import drawer

# Third-Party Import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import nnfs
from nnfs.datasets import spiral_data


### --- CODE --- ###
nnfs.init()
X, y = spiral_data(samples=100, classes=3)
delta = 0.01
xy = np.mgrid[-1:1.01:delta, -1:1.01:delta].reshape(2,-1).T
new_drawer = drawer.drawer(xy, X, y)

# Init
dense1 = network.layers.Layer_Dense(2, 64)
activation1 = network.activation_functions.Activation_ReLU()

dense2 = network.layers.Layer_Dense(64, 3)
loss_activation = network.Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = network.optimizers.Optimizer_Adam(learning_rate=0.05, decay=5e-7)


# Train
for epoch in range(10001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)


    # Accuracy and Loss
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')
    

    # Backward Pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    # Update Weights and Biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_paramy()


    # Plot
    if (True) and (not epoch % 500):
        dense1.forward(xy)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss_activation.forward(dense2.output, np.argmax(dense2.output, axis=1))
       
        output = loss_activation.output
        new_drawer.update(output, delta)

new_drawer.keep_open()