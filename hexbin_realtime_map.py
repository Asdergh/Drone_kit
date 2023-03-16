import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim

plt.style.use("dark_background")
fig, axes = plt.subplots()
def animo(i):

    axes.clear()
    x_grid = np.random.normal(size=(1, 800000))
    y_grid = np.random.normal(size=(1, 800000))
    axes.hexbin(x_grid, y_grid, cmap="twilight", gridsize=100, edgecolor="none")

animation = anim.FuncAnimation(fig, animo, interval=100)
plt.show()

