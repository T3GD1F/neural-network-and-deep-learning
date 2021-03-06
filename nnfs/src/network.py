"""
network.py
~~~~~~~~~~

Collection of different layer types
and activation functions.
Also includes different learning algorithms
"""

### --- IMPORTS --- ###
# Standard Import
import pickle
import copy

# Own Import
from src import layers
from src import activation_functions
from src import loss_functions
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
        self.softmax_classifier_output = None

    
    def add(self, layer):
        """Add one Layer to the network"""

        self.layers.append(layer)

    
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        """Sets Optimizer, Loss and Accuracy function"""

        if loss is not None:
            self.loss = loss
        
        if optimizer is not None:
            self.optimizer = optimizer
        
        if accuracy is not None:
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

        # check for faster gradient calculation (Softmax + Categorical Cross Entropy)
        if isinstance(self.layers[-1], activation_functions.Activation_Softmax) and \
           isinstance(self.loss, loss_functions.Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = activation_loss.Activation_Softmax_Loss_CategoricalCrossentropy()
        
        # Update Loss with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)          


    def train(self, X, y, *, 
                epochs=1, batch_size=None, 
                print_every=1, validation_data=None):
        """Method to train the Network"""

        self.accuracy.init(y)


        # default batch size
        train_steps = 1

        
        # checks for validation data
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        
        # number of batches for given batch size
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1


        # actual training Method
        for epoch in range(1, epochs+1):
            # Reset Epoch Parameters
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                # splits data into batches
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Forward Pass
                output = self.forward(batch_X, training=True)

                # Loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Prediction & Accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Backward Pass
                self.backward(output, batch_y)

                # Optimize
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Useful Information
                if not step % print_every or step == train_steps -1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Loss and Accuracy ofer Epoch
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'\n' +
                  f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

        
            # Validation
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)


    def forward(self, X, training):
        """Performs Forward Pass
        via loop, uses prev/next layer from finalize"""

        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output

    
    def backward(self, output, y):
        """Backward Pass
        Calculates Gradient, similar to Froward Pass"""

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    

    def evaluate(self, X_val, y_val, *, batch_size=None):
        """Evaluate the Model"""

        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            # splits data into batches
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Forward Pass
            output = self.forward(batch_X, training=False)

            # Loss
            self.loss.calculate(output, batch_y)

            # Prediction & Accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
    
  
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation: ' + 
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}' + 
              f'\n')


    def get_parameters(self):
        """Gets Weights and Biases for each layer"""
        
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters


    def set_parameters(self, parameters):
        """Sets Weights and Biases for each layer"""

        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)


    def save_parameters(self, path):
        """Writes Parameters to a file"""

        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    

    def load_parameters(self, path):
        """Reads Parameters from a file"""

        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))


    def save(self, path):
        """Hard copy of the complete model"""

        model = copy.deepcopy(self)

        # Clean Model
        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        # Save Model
        with open(path, 'wb') as f:
            pickle.dump(model, f)


    @staticmethod
    def load(path):
        """Loads a model"""

        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    

    def predict(self, X, *, batch_size=None):
        """Predicts on samples"""

        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            
            batch_output = self.forward(batch_X, training=False)

            output.append(batch_output)

        return np.vstack(output)