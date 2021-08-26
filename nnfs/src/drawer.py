"""
drawer.py
~~~~~~~~~~

Drwas Network Output
"""

### --- IMPORTS --- ###
# Third-Party Import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


### --- INIT --- ###
### --- 2D Classification Problem --- ###
class drawer:
    def __init__(self, default, points, y_true):
        plt.ion()

        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        cmap_name = 'my_list'
        self.cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
        self.fig, self.ax = plt.subplots()
        
        self.axim = self.ax.imshow(default, cmap=self.cmap, 
                        interpolation='bicubic', alpha=.25, extent=[-1,1,-1,1], 
                        vmin=0, vmax=2)
        self.ax.scatter(points[:,0], -points[:,1], c=y_true, cmap=self.cmap)


    def update(self, data, delta):
        data = np.argmax(data, axis=1)
        data = np.reshape(data, (int(2./delta) + 1, int(2./delta) + 1))

        self.axim.set_data(data.T)
        self.fig.canvas.flush_events()


    def update_bin(self, data, delta):
        data = (data > 0.5) * 2
        data = np.reshape(data, (int(2./delta) + 1, int(2./delta) + 1))
        self.axim.set_data(data.T)
        self.fig.canvas.flush_events()


    def keep_open(self):
        plt.ioff()
        plt.show()


### --- Regression Problem --- ###
class drawer_reg(drawer):
    def __init__(self, points, y_true):
        plt.ion()

        self.points = points

        self.fig, self.ax = plt.subplots()
        
        self.ax.plot(points, y_true)
        self.line1, = self.ax.plot(points, y_true)
        

    def update(self, data):
        self.line1.set_ydata(data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
