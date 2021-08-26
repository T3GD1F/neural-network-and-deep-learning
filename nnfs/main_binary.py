# Set-Alias -name python3 -Value C:/Users/felix/AppData/Local/Programs/Python/Python37-32/python.exe
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
X, y = spiral_data(samples=100, classes=2)

delta = 0.01
xy = np.mgrid[-1:1.01:delta, -1:1.01:delta].reshape(2,-1).T
new_drawer = drawer.drawer(xy, X, y)

y = y.reshape(-1, 1)

# Init
dense1 = network.layers.Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = network.activation_functions.Activation_ReLU()

dense2 = network.layers.Layer_Dense(64, 1)
activation2 = network.activation_functions.Activation_Sigmoid()

loss_function = network.loss.Loss_BinaryCrossentropy()

optimizer = network.optimizers.Optimizer_Adam(decay=5e-7)


### --- Train --- ###
for epoch in range(10001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Accuracy and Loss
    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, '+
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')
    

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


    # Plot
    if (True) and (not epoch % 500):
        dense1.forward(xy)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        new_drawer.update_bin(activation2.output, delta)


### --- Validation --- ###
X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y_test)
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


### --- Other Stuff --- ###
new_drawer.keep_open()