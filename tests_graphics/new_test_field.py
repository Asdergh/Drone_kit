import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

#neural network 1.
class Perceptron():
    def __init__(self, eta=0.01, epochs=40) -> None:
        self.eta = eta
        self.epochs = epochs
    
    def fit(self, X, y):
        self.W_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        for _ in range(self.epochs):
            for (xi, true_label) in zip(X, y):
                update = (true_label -  self.pure_input(xi)) * self.eta
                self.W_[1:] += update * xi
                self.W_[0] += update
                error = int(update != 0.0)
            self.cost_.append(error)
        return self

    def pure_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def prediction(self, X):
        return np.where(self.pure_input(X) >= 0.0, 1, -1)

# neural network 2.
class RegrationADL():
    def __init__(self, eta, epochs) -> None:
        self.eta = eta
        self.epochs = epochs
    
    def fit(self, X, y):
        self.W_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []

        for _ in range(self.epochs):
            tmp_input = self.net_input(X)
            error = y - tmp_input
            self.W_[1:] +=  self.eta * X.T.dot(error)
            self.W_[0] += self.eta * error.sum()
            cost = (error ** 2).sum()
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def activation(self, X):
        return self.net_iput(X)
    
    def prediction(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)    
    

#neral network result graph
def generate_plot_desion(X, y, classifier, frrequency=0.02, ix=0, iy=0):
    
    x1_min = X[:, 0].min() - 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 0].max() + 1
    x1_max = X[:, 1].max() + 1

    markers = ["v", "^", "s", "o", "x"]
    colors = ["red", "green", "lightgreen", "yelow", "gray"]
    cmap = mcolors.ListedColormap(colors[:len(np.unique(y))])

    x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max), np.arange(x2_min, x2_max))
    Z = classifier.prediction(np.array([x1_grid.ravel(), x2_grid.ravel()]).T)
    Z = Z.reshape(x1_grid.shape)
    print(Z.shape)

    axis[ix, iy].contourf(x1_grid, x2_grid, Z, cmap=cmap, alpha=0.67)
    
    for(index, cls) in enumerate(np.unique(y)):
        axis[ix, iy].scatter(x=X[y==cls, 0], y=X[y==cls, 1], color=cmap(index), marker=markers[index], label=cls)

#data preproccesing
plt.style.use("seaborn")
data = pd.read_csv("iris.data")
X = data.iloc[0:100, [0, 2]]
y = data.iloc[0:100, 4]
X = np.array(X)
X_std = X.copy()
X_std[:, 0] = (X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
X_std[:, 1] = (X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()
y = np.where(y == "Iris-setosa", 1, -1)
print(X.shape)

#creating the objects of neural networks
pea = Perceptron(eta=0.0001, epochs=30).fit(X, y)
ada = RegrationADL(eta=0.0001, epochs=30).fit(X, y)

#ploting graphs 
fig, axis = plt.subplots(2, 2)
axis[0, 0].plot(range(0, len(ada.cost_)), np.log(ada.cost_), color="green", marker="^", label="velocity test")
axis[0, 0].legend(loc="upper left")
generate_plot_desion(X_std, y, ada, ix=0, iy=1)

axis[1, 0].plot(range(0, len(pea.cost_)), np.log(pea.cost_), marker="^", label="velocity test", color="blue")
axis[1, 0].legend(loc="upper left")
generate_plot_desion(X_std, y, pea, ix=1, iy=1)
plt.show()



