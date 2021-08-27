# Set-Alias -name python3 -Value C:/Users/felix/AppData/Local/Programs/Python/Python37-32/python.exe
"""
mainy.py
~~~~~~~~~~

Init and run the Neural Network.
"""

### --- IMPORTS --- ###
# Standard Import
import os

from matplotlib import image

# Own Import
from src import network
from src import loader
from src import drawer

# Third-Party Import
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
image_data = cv2.imread('example_data/tshirt.png', cv2.IMREAD_GRAYSCALE)        # Load
image_data = cv2.resize(image_data, (28, 28))                                   # Resize
image_data = 255 - image_data                                                   # Invert
image_data = image_data.reshape(1, -1)                                          # Reshape
image_data = (image_data.astype(np.float32) - 127.5) / 127.5                    # Rescale


### Neural Network
# Init Model
model = network.Model.load('fashion_mnist.model')

confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)

prediction = fashion_mnist_labels[predictions[0]]

print(prediction)

# plt.imshow((X[0], reshape(28, 28)), cmap='gray')
# plt.show()
# exit()