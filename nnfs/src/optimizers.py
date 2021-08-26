"""
optimizers.py
~~~~~~~~~~

Collection of activation functions.
Each function provides forward pass and backpropagation.
"""

### --- IMPORTS --- ###
# Standard Import

# Third-Party Import
import numpy as np


class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        """Stochastic Gradient Optimizer + Momentum
        Decay is Rate of Reducing the Learning rate
        Momentum is the influence of previous gradients"""
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum


    def pre_update_params(self):
        """Update the Learningrate if
        a Decay is given"""
        
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    

    def update_params(self, layer):
        """Update Weights and Biases
        Will use current Learning Rate and,
        if given, the Momentum"""
        
        # Momentum SGD
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):          # initial init
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums - \
                                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - \
                                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        # Update Weights and Biases
        layer.weights += weight_updates
        layer.biases += bias_updates
    

    def post_update_paramy(self):
        """Update the Iterations"""
        
        self.iterations += 1


# AdaGrad Optimizer
class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        """AdaGrad Optimizer
        Vanilla SGD + Normalize Gradients"""
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    
    def pre_update_params(self):
        """Update the Learningrate if
        a Decay is given"""
        
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    

    def update_params(self, layer):
        """Update Weights and Biases
        Will use current Learning Rate with
        respect to normalized Gradient"""

        if not hasattr(layer, 'weight_cache'):          # initial init
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD + Normalization
        layer.weights += -(self.current_learning_rate*layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -(self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.weight_cache) + self.epsilon)
        

    def post_update_paramy(self):
        """Update the Iterations"""
        
        self.iterations += 1


# RMSprop Optimizer
class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=.9):
        """RMSprop Optimizer
        like AdaGrad, but smoother Cache"""

        self.leraning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    

    def pre_update_params(self):
        """Update the Learningrate if
        a Decay is given"""
        
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))


    def update_params(self, layer):
        """Update Weights and Biases
        Will use current Learning Rate with
        respect to RMSprop Gradient"""

        if not hasattr(layer, 'weight_cache'):          # initial init
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1. - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1. - self.rho) * layer.dbiases**2

        # Vanilla SGD + Normalization
        layer.weights += -(self.current_learning_rate*layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -(self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.weight_cache) + self.epsilon)
        

    def post_update_paramy(self):
        """Update the Iterations"""
        
        self.iterations += 1


# Adam
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        """Adam
        RMSprop + SGD momentum"""
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def pre_update_params(self):
        """Update the Learningrate if
        a Decay is given"""
        
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    
    def update_params(self, layer):
        """Update Weights and Biases
        Will use current Learning Rate with
        respect to RMSprop Gradient + Momentum"""

        if not hasattr(layer, 'weight_cache'):          # initial init
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)

                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_cache + (1. - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_2 * layer.bias_cache   + (1. - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected   = layer.bias_momentums   / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache   = self.beta_2 * layer.bias_cache   + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations +1))
        bias_cache_corrected   = layer.bias_cache   / (1 - self.beta_2 ** (self.iterations +1))

        # Vanilla SGD + normalization
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases  += -self.current_learning_rate * bias_momentums_corrected   / (np.sqrt(bias_cache_corrected)   + self.epsilon)
        

    def post_update_paramy(self):
        """Update the Iterations"""
        
        self.iterations += 1