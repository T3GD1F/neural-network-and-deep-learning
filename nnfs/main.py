# Set-Alias -name python3 -Value C:/Users/felix/AppData/Local/Programs/Python/Python37-32/python.exe
"""
mainy.py
~~~~~~~~~~

Init and run the Neural Network.
"""

### --- IMPORTS --- ###
# Standard Import
import os

from numpy.core.fromnumeric import reshape

# Own Import
from src import network
from src import loader
from src import drawer

# Third-Party Import
import numpy as np
import matplotlib.pyplot as plt

import nnfs
nnfs.init()


### --- CODE --- ###
### Hyperparameters
EPOCHS = 10
BATCH_SIZE = 128


### Load Data (run get_mnist.py before executing this for first time)
print("% Start Loading Data")
X, y, X_test, y_test = loader.create_data_mnist('fashion_mnist_images')
print("% Successfully loaded Data", "\n")


print("% Preprocess Data")
# Shuffle Data
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and Shape Data
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

print("% Successfully preprocessed Data", "\n")


### Neural Network
# Init Model
n = 128
model = network.Model()

model.add(network.layers.Layer_Dense(X.shape[1], n))
model.add(network.activation_functions.Activation_ReLU())
model.add(network.layers.Layer_Dense(n, n))
model.add(network.activation_functions.Activation_ReLU())
model.add(network.layers.Layer_Dense(n, 10))
model.add(network.activation_functions.Activation_Softmax())

model.set(
    loss = network.loss_functions.Loss_CategoricalCrossentropy(),
    optimizer = network.optimizers.Optimizer_Adam(decay=5e-5),
    accuracy=network.accuracy.Accuracy_Categorical()
)

model.finalize()

# Train Model
model.train(X, y, validation_data=(X_test, y_test),
    epochs = 10,
    batch_size=128,
    print_every = 100
)

model.evaluate(X_test, y_test)

# plt.imshow((X[0], reshape(28, 28)), cmap='gray')
# plt.show()
# exit()