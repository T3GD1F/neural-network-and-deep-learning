# Set-Alias -name python3 -Value C:/Users/felix/AppData/Local/Programs/Python/Python37-32/python.exe
"""
main_regression.py
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
from nnfs.datasets import sine_data


### --- CODE --- ###
nnfs.init()
X, y = sine_data()

new_drawer = drawer.drawer_reg(X, y)


# Init
dense1 = network.layers.Layer_Dense(1, 64)
activation1 = network.activation_functions.Activation_ReLU()

dense2 = network.layers.Layer_Dense(64, 64)
activation2 = network.activation_functions.Activation_ReLU()

dense3 = network.layers.Layer_Dense(64, 1)
activation3 = network.activation_functions.Activation_Linear()

loss_function = network.loss.Loss_MeanSquaredError()

optimizer = network.optimizers.Optimizer_Adam(learning_rate=0.005, decay=1e-3)

accuracy_precision = np.std(y) / 250

### --- Train --- ###
for epoch in range(10001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)


    # Accuracy and Loss
    data_loss = loss_function.calculate(activation3.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + \
                          loss_function.regularization_loss(dense2) + \
                          loss_function.regularization_loss(dense3)
    loss = data_loss + regularization_loss 

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f},' +
              f'reg_loss: {regularization_loss:.3f}),' +
              f'lr: {optimizer.current_learning_rate}')
    

    # Backward Pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    # Update Weights and Biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()


    # Plot
    if (True) and (not epoch % 500):
        new_drawer.update(predictions)


### --- Other Stuff --- ###
new_drawer.keep_open()
