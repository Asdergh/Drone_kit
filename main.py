import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolor
import pandas as pd


def generate_plot_foo(X, Y, classifier, frequency=0.02):
    markers = ["s", "x", "o", "^", "v"]
    colors = ["blue", "red", "lightgreen", "gray", "cyan"]

    cmap = mcolor.ListedColormap(colors[:len(np.unique(Y))])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1_grid, xx2_grid = np.meshgrid(np.arange(x1_min, x1_max, frequency), np.arange(x2_min, x2_max, frequency))
    Z = classifier.prediction(np.array(xx1_grid.ravel(), xx2_grid.ravel()).T)
    Z = np.reshape(xx1_grid.shape)

    plt.contourf(xx1_grid, xx2_grid, Z, cmap=cmap)
    plt.xlabel("длинна чашелистника")
    plt.ylabel("длинна лепистка")

    for (index, cls) in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == cls, 0], y=X[Y == cls, 1], c=cmap(index), marker=markers[index], label=cls)





class Perceptron():
    def __init__(self, eta, epochs) -> None:
        self.eta = eta
        self.epochs = epochs
    
    def fit(self, X, Y):
        self.W = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.epochs):
            error = 0
            for (xi, true_label) in zip(X, Y):
                update = self.eta * (true_label - self.prediction(xi))
                self.W[1:] += update * xi
                self.W[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.W[1:]) + self.W[0]

    def prediction(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


if __name__ == "__main__":
    pn = Perceptron(eta=0.0, epochs=30)
    url = "iris.data"
    data_frame = pd.read_csv(url, header=None)
    data_frame.tail()

    y = data_frame.iloc[0:100, 4].values
    y = np.where(y == "Iris-versicolor", -1, 1)
    X = data_frame.iloc[0:100, [0, 2]].values

    pn.fit(X, y)
    
    generate_plot_foo(X, y, classifier=pn)
    plt.xlabel("длинна чашелистика")
    plt.ylabel("длинна лепистка")
    plt.legend(loc="upper left")

    plt.show()
   
