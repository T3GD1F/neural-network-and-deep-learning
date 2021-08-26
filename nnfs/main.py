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
from nnfs.datasets import sine_data


### --- CODE --- ###
nnfs.init()
X, y = sine_data()

### Init Model
n = 64
model = network.Model()

model.add(network.layers.Layer_Dense(1, n))
model.add(network.activation_functions.Activation_ReLU())
model.add(network.layers.Layer_Dense(n, n))
model.add(network.activation_functions.Activation_ReLU())
model.add(network.layers.Layer_Dense(n, 1))
model.add(network.activation_functions.Activation_Linear())

model.set(
    loss = network.loss.Loss_MeanSquaredError(),
    optimizer = network.optimizers.Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    accuracy=network.accuracy.Accuracy_Regression()
)

model.finalize()

model.train(X, y, 
    epochs = 10000,
    print_every = 100
)