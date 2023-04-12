import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colorsm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

def generate_desion_plot(X, Y, classifier, frequency=0.02, test_idx=None):

    markers = ["s", "v", "^", "o", "x"]
    colors = ["gray", "red", "blue", "green", "greenlight"]
    cmaps = colorsm.ListedColormap(colors[: len(np.unique(Y))])
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    x1_min = X[:, 1].min() - 1
    x1_max = X[:, 1].max() + 1

    xx_grid, xx1_grid = np.meshgrid(np.arange(x_min, x_max, frequency), 
                                    np.arange(x1_min, x1_max, frequency))
    
    Z =classifier.predict(np.array([xx_grid.ravel(), xx1_grid.ravel()]).T)
    Z = Z.reshape(xx_grid.shape)

    plt.contourf(xx_grid, xx1_grid, Z, cmap=cmaps)
    for (item, cls) in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == cls, 0], y=X[Y == cls, 1], marker=markers[item],
                    cmap=cmaps(item), label=cls)
        
        if test_idx:
            X_test, y_test = X[test_idx, :], Y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", linewidths=1, s=55, label="тестовый набор")


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

ppn = Perceptron(max_iter=40, eta0 = 0.0001, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn.fit(X_train, Y_train)

X_combined = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))
generate_desion_plot(X=X_combined, Y=Y_combined, classifier=ppn, test_idx=(105, 150))
plt.legend(loc="upper left")
plt.show()

