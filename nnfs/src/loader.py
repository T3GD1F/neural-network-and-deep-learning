"""
loader.py
~~~~~~~~~~

Loads in MNIST Dataset
"""

### --- IMPORTS --- ###
# Standard Import
import os

# Own Import

# Third-Party Import
import numpy as np
import cv2


### --- CODE --- ###
def load_mnist_dataset(dataset, path):
    """Loads the data from a specific path
    with respect to the dataset type (train/test)"""

    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)
    
    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    """Loads the training and 
    testing dataset"""
    
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test