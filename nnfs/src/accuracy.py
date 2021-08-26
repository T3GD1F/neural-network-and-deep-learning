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
        
        comparison = self.compare(predictions, y)
        accuracy = np.mean(comparison)

        return accuracy


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