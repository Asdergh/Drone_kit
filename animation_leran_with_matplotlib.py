import matplotlib.pyplot as plt
import matplotlib.animation as animo
import numpy as np
import pandas as pd
import random as rd


class AnimExample():

    def __init__(self, surface_mode: bool, X_grid=None,
                  Y_grid=None, Z_grid=None, phi=None,
                    theta=None, cmap=None, gradient_mode=False, alpha=0) -> None:
        self.Gn_mode = gradient_mode
        self.surface_mode = surface_mode
        #self.XYZ = np.dstack((X_grid, Y_grid, Z_grid))
        self.XYmesh = np.column_stack((X_grid, Y_grid))
        self.angles = np.column_stack((phi, theta))
        self.figure = plt.figure()
        self.axis = self.figure.add_subplot(projection="3d")
        self.time_changes = np.linspace(-np.pi, np.pi, 100)
        self.cmap = cmap
        self.alpha = alpha


    def anim_surface_(self, i):
        self.axis.clear()

        X_grid, Y_grid = np.meshgrid(self.XYmesh[:, 0], self.XYmesh[:, 1])
        #X_gradient, Y_gradient = np.gradient(X_grid), np.gradient(Y_grid)
        Z_grid = np.cos(X_grid + self.time_changes[i]) * np.sin(Y_grid + self.time_changes[i]) * np.sin(X_grid + Y_grid + self.time_changes[i]) 
        #Z_gradient = np.cos(X_gradient + self.time_changes[i]) * np.sin(Y_gradient + self.time_changes[i]) * np.sin(X_gradient + Y_gradient + self.time_changes[i])

        """if self.Gn_mode == True:
            self.axis.plot_surface(X_gradient, Y_gradient, Z_gradient, cmap=self.cmap, alpha=self.alpha)
            self.axis.contourf(X_gradient, Y_gradient, Z_gradient, cmap="magma", zdir="z", offset=-2)
            self.axis.scatter(X_gradient, Y_gradient, Z_gradient, cmap="binry", c=Z_gradient, s=0.123)"""
        
        self.axis.plot_surface(X_grid, Y_grid, Z_grid, cmap=self.cmap, alpha=self.alpha)
        self.axis.contourf(X_grid, Y_grid, Z_grid, cmap="magma", zdir="z", offset=-2)
        self.axis.scatter(X_grid, Y_grid, Z_grid, cmap="binary", c=Z_grid, s=0.123)
        
        self.axis.set_xlabel("X label", color="blue")
        self.axis.set_ylabel("Y label", color="red")
        self.axis.set_zlabel("Z label", color="green")
    
    
    def anim_params_(self, i):
        self.axis.clear()
        phi, theta = np.meshgrid(self.angles[:, 0], self.angles[:, 1])
        x_core = -np.cos(phi * self.time_changes[i]) * np.sin(theta * self.time_changes[i])
        y_core = np.sin(phi * self.time_changes[i]) * np.sin(theta * self.time_changes[i])
        z_core = np.cos(phi * self.time_changes[i])
        self.axis.plot_surface(x_core, y_core, z_core, cmap=self.cmap, alpha=self.alpha)
        self.axis.contourf(x_core, y_core, z_core, zdir="z", cmap="plasma", offset=-1)
        self.axis.set_xlabel("X axis", color="blue")
        self.axis.set_ylabel("Y axis", color="red")
        self.axis.set_zlabel("Z axis", color="green")
    
    
        
    


    def run(self):
        if self.surface_mode == True:
            animation = animo.FuncAnimation(self.figure, self.anim_surface_, interval=100)
        else:
            animation = animo.FuncAnimation(self.figure, self.anim_params_, interval=1000)
        
        plt.show()

if __name__ == "__main__":
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    X_grid = np.linspace(0, np.pi, 100)
    Y_grid = np.linspace(0, np.pi, 100)
    print(np.dstack((phi, theta)))
    print(np.column_stack((X_grid, Y_grid))[:, [0, 1]])

    obj = AnimExample(surface_mode=True, phi=phi, theta=theta, X_grid=X_grid, Y_grid=Y_grid, cmap="coolwarm", alpha=0.67, gradient_mode=False).run()





