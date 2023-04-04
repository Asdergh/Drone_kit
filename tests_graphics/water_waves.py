import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation



class Water_waves():
    def __init__(self, g_velocity: int, depth: int, x_cores=None, y_cores=None) -> None:
        self.G = g_velocity
        self.D = depth
        self.time = np.linspace(0, np.pi, 100)
        if (x_cores != None) and (y_cores != None):
            self.Grid = np.stack([x_cores, y_cores])

        else:
            x_grid = np.linspace(-np.pi, np.pi, 100)
            y_grid = np.linspace(-np.pi, np.pi, 100)
            self.Grid = np.stack([x_grid, y_grid])

        self.figure = plt.figure()
        self.axis_3d = self.figure.add_subplot(projection="3d")
    
    def waves(self, i):

        self.axis_3d.clear()
        x_grid, y_grid = np.meshgrid(self.Grid[0], self.Grid[1])
        eta = ((y_grid * self.time[i]) / 2 * np.pi) * np.tanh((2 * np.pi * self.D) / 2 * np.pi) + x_grid + y_grid 

        self.axis_3d.plot_surface(x_grid, y_grid, eta, cmap="coolwarm", alpha=0.76)
        self.axis_3d.contourf(x_grid, y_grid, eta, zdir="z", cmap="coolwarm")

    def run(self):
        anim = animation.FuncAnimation(self.figure, self.waves, interval=100)
        plt.show()

if __name__ == "__main__":
    obj = Water_waves(g_velocity=12.0, depth=30.0).run()
