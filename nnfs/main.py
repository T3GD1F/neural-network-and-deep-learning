# Set-Alias -name python3 -Value C:/Users/felix/AppData/Local/Programs/Python/Python37-32/python.exe
"""
mainy.py
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

import nnfs
from nnfs.datasets import spiral_data


### --- CODE --- ###
nnfs.init()
X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

### Init Model
n = 512
model = network.Model()

model.add(network.layers.Layer_Dense(2, n, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(network.activation_functions.Activation_ReLU())
model.add(network.layers.Layer_Dropout(0.1))
model.add(network.layers.Layer_Dense(n, 3))
model.add(network.activation_functions.Activation_Softmax())

model.set(
    loss = network.loss_functions.Loss_CategoricalCrossentropy(),
    optimizer = network.optimizers.Optimizer_Adam(learning_rate=0.05, decay=5e-7),
    accuracy=network.accuracy.Accuracy_Categorical(binary=True)
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test),
    epochs = 10000,
    print_every = 100
)
