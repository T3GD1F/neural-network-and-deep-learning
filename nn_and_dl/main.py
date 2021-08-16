"""
main.py
~~~~~~~~~~

Init and run the Neural Network.
"""

### --- Load Data --- ###
### Import
from src import mnist_loader

#### Code
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("MNIST DATA loaded\n")


### --- Network --- ###
### Import
from src import network

### Code
# Create a new NN with 784 Input, 30 Hidden and 10 Out
net = network.Network([784, 30, 10])

# Learning: Use SGD over 30 epoches
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)