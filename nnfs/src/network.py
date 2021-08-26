"""
network.py
~~~~~~~~~~

Collection of different layer types
and activation functions.
Also includes different learning algorithms
"""

### --- IMPORTS --- ###
# Standard Import

# Own Import
from src import layers
from src import activation_functions
from src import loss
from src import optimizers
from src import activation_loss
from src import accuracy

# Third-Party Import
import numpy as np


### --- CODE --- ###
class Model:
    def __init__(self):
        """Model Class
        Class for fast Usage of """ 

        self.layers = []

    
    def add(self, layer):
        """Add one Layer to the network"""

        self.layers.append(layer)

    
    def set(self, *, loss, optimizer, accuracy):
        """Sets Optimizer, Loss and Accuracy function"""

        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    

    def finalize(self):
        """Defines the previous and next layer
        foreach layer in the network"""

        self.input_layer = layers.Layer_Input()
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            # first layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            # middle layers
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # output layer
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        self.loss.remember_trainable_layers(self.trainable_layers)


    def train(self, X, y, *, epochs=1, print_every=1):
        """Method to train the Network"""

        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            # Forward Pass
            output = self.forward(X)
            
            # Loss
            data_loss, regularization_loss = self.loss.calculate(output, y)
            loss = data_loss + regularization_loss

            # Prediction & Accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Backward Pass
            self.backward(output, y)

            # Optimize
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Useful Information
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f},' +
                      f'reg_loss: {regularization_loss:.3f}),' +
                      f'lr: {self.optimizer.current_learning_rate}')

    def forward(self, X):
        """Performs Forward Pass
        via loop, uses prev/next layer from finalize"""

        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        return layer.output

    
    def backward(self, output, y):
        """Backward Pass
        Calculates Gradient, similar to Froward Pass"""

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)