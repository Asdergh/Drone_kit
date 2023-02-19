import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import pandas as pd
import numpy as np



class Perceptron(object):

    def __init__(self, eta=0.0, epochs=20) -> None:
        self.eta = eta
        self.epochs = epochs
    def fit(self, X, Y):
        self.W = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            error = 0
            for(xi, true_label) in zip(X, Y):
                update = self.eta * (true_label - self.prediction(xi))
                self.W[1:] += update * xi
                self.W[:] += update
                errrors = int(update != 0.0)
            self.errors_.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.W[1:]) + self.W[0]
    def prediction(self, X):
        return np.where(self.net_iput(X) >= 0.0, 1, -1)

if __name__ == "__main__":
    pn = Perceptron(eta=0.0, epochs=30)

    url = "iris.data"
    data_set = pd.read_csv(url, header=None)
    data_set.tail()

    Y = data_set.iloc[:100, 4].values
    Y = np.where(Y == "Iris-verisicolor", 1, -1)
    X = data_set.iloc[:100, [0, 1, 2]].values

    axes_3d = plt.figure().add_subplot(projection="3d")
    axes_3d.set_xlabel("длинна чащелистника")
    axes_3d.set_ylabel("ширина чашелистника")
    axes_3d.set_zlabel("длинна лепистка")
    axes_3d.scatter(X[:50, 0], X[:50, 1], X[:50, 2], marker="s", color="red", label="ирис щетинистый")
    axes_3d.scatter(X[50:, 0], X[50:, 1], X[50:, 2], marker="^", color="blue", label="ирис разноцветный")
    axes_3d.legend(loc="upper left")
    plt.show()

    print(X, Y)

