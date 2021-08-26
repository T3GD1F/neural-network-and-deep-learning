# Set-Alias -name python3 -Value C:/Users/felix/AppData/Local/Programs/Python/Python37-32/python.exe
"""
main_classification.py
~~~~~~~~~~

Init and run the Neural Network.
Specialised for Classification Problems.
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
dense1 = network.layers.Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = network.activation_functions.Activation_ReLU()

dropout1 = network.layers.Layer_Dropout(0.1)

dense2 = network.layers.Layer_Dense(512, 3)
loss_activation = network.Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = network.optimizers.Optimizer_Adam(learning_rate=0.05, decay=5e-6)


### --- Train --- ###
for epoch in range(10001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)


    # Accuracy and Loss
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss 

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f},' +
              f'reg_loss: {regularization_loss:.3f}),' +
              f'lr: {optimizer.current_learning_rate}')
    

    # Backward Pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)


    # Update Weights and Biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


    # Plot
    if (True) and (not epoch % 500):
        dense1.forward(xy)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss_activation.forward(dense2.output, np.argmax(dense2.output, axis=1))
       
        output = loss_activation.output
        new_drawer.update(output, delta)


### --- Validation --- ###
X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


### --- Other Stuff --- ###
new_drawer.keep_open()
