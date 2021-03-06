"""
accuracy.py
~~~~~~~~~~

Collection of accuracy functions.
"""

### --- IMPORTS --- ###
# Standard Import

# Third-Party Import
import numpy as np


### --- CODE --- ###
# Common Accuracy Class
class Accuracy:
    def calculate(self, predictions, y):
        """Standard Method for calculating accuracy"""
        
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy
    

    def calculate_accumulated(self):
        """Calculates accumulated Loss"""
        
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy
    

    def new_pass(self):
        """Resets Variables for accumulated Loss"""
        
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Accuracy for Regression
class Accuracy_Regression(Accuracy):
    def __init__(self):
        """Calculates the Accuracy
        for Regression Model"""

        self.precision = None
    

    def init(self, y, reinit=False):            # not to be confused with __init__
        """initialises the precision"""

        if (self.precision is None) or reinit:
            self.precision = np.std(y) / 250

    
    def compare(self, predictions, y):
        """Finally Calculates the Accuracy"""
        
        return np.absolute(predictions - y) < self.precision
    

# Accuracy for Categorical Classification
class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        """Init, sets if Binary Classifiaction or not"""
    
        self.binary = binary
    

    def init(self, y):
        """No init needed"""

        pass


    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        
        return predictions == y
