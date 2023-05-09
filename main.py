import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib.colors as mcolors
import pandas as pd
import random as rd
import os
from progress.bar import IncrementalBar

plt.style.use("seaborn")
def generate_randin_points(P_C=10000):

    x_cores, y_cores, z_cores = np.zeros(P_C + 1), np.zeros(P_C + 1), np.zeros(P_C + 1)
    for index in range(P_C + 1):
        x_cores[index] = x_cores[index - 1] + float(rd.randint(0, 123)) * rd.choice([-1, 1])
        y_cores[index] = y_cores[index - 1] + float(rd.randint(0, 123)) * rd.choice([-1, 1])
        z_cores[index] = z_cores[index - 1] + float(rd.randint(0, 123)) * rd.choice([-1, 1])
    
    cores = np.stack((x_cores, y_cores, z_cores)).T
    with open("data.xyz", "w") as file:
        with IncrementalBar(max=(P_C + 1)) as bar:
            for (index, item) in enumerate(cores):
                bar.next()
                file.write(f"{item[0]}\t{item[1]}\t{item[2]}\n")
    return cores

def graph_plane_plt(time_coef):

    axis_3d.clear()
    time = np.linspace(1, np.pi, 100)
    x_cores = np.linspace(-np.pi, np.pi, 100)
    y_cores = np.linspace(-np.pi, np.pi, 100)
    xy1_grid, xy2_grid = np.meshgrid(x_cores, y_cores)

    z_cores = np.sin(np.sqrt(xy1_grid ** 2 * time_coef + xy2_grid ** 2 * time_coef)) / (np.sqrt(xy1_grid ** 2 * time_coef + xy2_grid ** 2 * time_coef))

    axis_3d.plot_surface(xy1_grid, xy2_grid, z_cores, cmap="bone", alpha=0.56)

def graph_desion(X, Y, classifier, test_idx):

    markers = ["o", "+", "v", "^"]
    colors = ["gray", "blue", "red"]
    cmaps = mcolors.ListedColormap(colors[:len(np.unique(Y))])

    xx1_grid, xx2_grid = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1),
                                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1))
    z_grid = classifier.predict(np.array[xx1_grid.ravel(), xx2_grid.ravel()]).T
    z_grid = z_grid.reshape(xx1_grid.shape)
    plt.contourf(xx1_grid, xx2_grid, z_grid, cmap=cmaps, alpha=0.3)

    for (index, cls) in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y==cls, 0], y=X[Y==cls, 1], 
                    marker=markers[index], color=colors(index))


figure = plt.figure()
axis_3d = figure.add_subplot(1, 1, 1, projection="3d")
array = generate_randin_points(P_C=10000)
#axis_3d.scatter(array[:, 0], array[:, 1], array[:, 2], c=array[:, 2], cmap="twilight", s=0.78, alpha=0.3, marker="v")
#axis_2d_01.plot(range(0, len(array[:, 0])), array[:, 0], color="red")
animation = manimation.FuncAnimation(figure, graph_plane_plt, interval=100)
plt.show()





