import matplotlib.pyplot as plt
import numpy as np
import random as rd
import matplotlib.animation as animo

figure = plt.figure()
axis_3d = figure.add_subplot(projection="3d")

theta = np.linspace(0, np.pi * 2, 100)
phi = np.linspace(0, np.pi * 2, 100)
time = np.linspace(0, np.pi, 100)

plt.style.use("seaborn")

def animation(i):
    
    axis_3d.clear()
    x_cores = np.linspace(-np.pi, np.pi, 100)
    y_cores = np.linspace(-np.pi, np.pi, 100)
    
    xx_cores, yy_cores = np.meshgrid(x_cores, y_cores)
    #eta = np.tanh((-1 * (xx_cores * i - 5) ** 2 / 20) - (yy_cores * i - 5) ** 2 / 100) / 2
    eta = 123 * np.sin(xx_cores ** 2 + i - yy_cores ** 2 + i - 35)
    axis_3d.plot_surface(xx_cores, yy_cores, eta, cmap="twilight", alpha=0.76)
    axis_3d.contour(xx_cores, yy_cores, eta, zdir="z", offset=-0.50, cmap="coolwarm")
    axis_3d.contour(xx_cores, yy_cores, eta, zdir="x", offset=0, cmap="coolwarm")
    axis_3d.contour(xx_cores, yy_cores, eta, zdir="y", offset=0, cmap="coolwarm")
    

anim = animo.FuncAnimation(figure, animation, interval=100)
plt.show()