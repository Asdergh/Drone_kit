import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Perceptron():
    def __init__(self, eta, epochs) -> None:
        self.eta = eta
        self.epochs = epochs
    
    def fit(self, X, y):

        self.W_ = np.zeros(X.shape[1])
        self.cost_ = []
        for _ in range(self.epochs):
            errors = 0
            for (xi, true_label) in zip(X, y):
                update = true_label - self.net_input(xi)
                self.W_[1:] += self.eta * update * xi
                self.W_[0] += self.eta * update
                errors = int(update != 0.0)
            self.cost_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


data = datasets.load_iris()
figure = plt.figure()
axis_3d_01 = figure.add_subplot(2, 2, 1, projection="3d")
axis_2d_02 = figure.add_subplot(2, 2, 2)
X = data.data[:, [0, 1, 2]]
y = data.target
markers = ["v", "^", "x"]
colors = ["red", "blue", "green"]
angles = np.linspace(-np.pi, np.pi, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
snn = StandardScaler()
snn.fit(X_train)
X_train_std, X_test_std = snn.transform(X_train), snn.transform(X_test)
ppn = Perceptron(epochs=40, eta=0.001)
ppn.fit(X_train_std, y_train)
for (index, cls) in enumerate(np.unique(y)):
    axis_3d_01.scatter(X[y == cls, 0], X[y == cls, 1], X[y == cls, 2],
                    color=colors[index], marker=markers[index], label=cls)
    curent_x = X[y == cls, 0]
    curent_y = X[y == cls, 1]
    xx1_grid, xx2_grid = np.meshgrid(curent_x, curent_y)
    Z = (index * xx1_grid + index * xx2_grid) / index
    #Z_01 = np.sin(np.sqrt(xx1_grid ** 2 + xx2_grid ** 2)) / np.tanh(xx1_grid ** 2 + xx2_grid ** 2)
    axis_3d_01.plot_surface(xx1_grid, xx2_grid, Z, color="gray", alpha=0.2)
    #axis_3d.plot_surface(xx1_grid, xx2_grid, Z_01, color="gray", alpha=0.2)
axis_2d_02.plot(range(0, len(ppn.cost_)), ppn.cost_, color="green", marker="x", label="[cost info]")

plt.legend(loc="upper left")
plt.show()


