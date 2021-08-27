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


### Labels
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


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

keys = np.array(range(X_test.shape[0]))
np.random.shuffle(keys)
X_test = X_test[keys]
y_test = y_test[keys]

# Scale and Shape Data
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

print("% Successfully preprocessed Data", "\n")


### Neural Network
# Init Model
model = network.Model.load('fashion_mnist.model')


confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)


for X_value, prediction, y_test_label in zip(X_test[:5], predictions, y_test[:5]):

    print("Model:", fashion_mnist_labels[prediction],
          "  Real:", fashion_mnist_labels[y_test_label])

    plt.imshow(X_value.reshape(28, 28), cmap='gray')
    plt.show()

# plt.imshow((X[0], reshape(28, 28)), cmap='gray')
# plt.show()
# exit()