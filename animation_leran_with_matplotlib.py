import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


class Amin_Plot():

    def __init__(self, delay=None, cmap=None, x_grid=None, y_grid=None,
                 z_grid=None, phi=None, theta=None) -> None:
        
        self.delay = delay
        self.cmap = cmap
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid
        self.phi = phi
        self.theta = theta
        self.axes_3d = plt.figure().add_subplot(projection="3d")
        self.anim = animation
    
    def anim(self, i):

        self.phi, self.theta = np.meshgrid(self.phi, self.theta)

        x_grid = np.cos(self.phi) * np.cos(self.theta)
        y_grid = np.sin(self.phi) * np.cos(self.theta)
        z_grid = np.sin(self.phi) * np.sin(self.theta)

        self.axes_3d.scatter(x_grid, y_grid, x_grid, cmap=self.cmap, s=0.98)
    def run(self):

        self.anim.FuncAnimation(self.axes_3d.figure, self.anim, interval=self.delay)
        plt.show()


phi = np.linspace(0, 2.0 * np.pi, 100)
theta = np.linspace(0, 2.0 * np.pi, 100)
new_object = Amin_Plot(delay=1000, phi=phi, theta=theta, cmap="magma").run()

        