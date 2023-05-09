import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn.datasets as datasets
from sklearn.linear_model import Perceptron, LogisticRegression



plt.style.use("seaborn")
figure = plt.figure()
axis_3d = figure.add_subplot(2, 3, 1,projection="3d")
axis_3d_01 = figure.add_subplot(2, 3, 2, projection="3d")
axis_3d_02 = figure.add_subplot(2, 3, 3, projection="3d")
axis_3d.set_xlabel("параметр [x1]")
axis_3d.set_ylabel("параметр [x2]")
axis_3d.set_zlabel("параметр [x3]")
axis_3d_01.set_xlabel("параметр [x1]")
axis_3d_01.set_ylabel("параметр [x2]")
axis_3d_01.set_zlabel("параметр [x3]")
axis_3d_02.set_xlabel("параметр [x1]")
axis_3d_02.set_ylabel("параметр [x2]")
axis_3d_02.set_zlabel("параметр [x3]")


def generate_desion(X, Y, classifier, test_idx=None, curent_view=axis_3d, basis=2):
    markers = ["o", "x", "v", "^"]
    colors = ["tomato", "blue", "gray", "green"]
    cmaps = mcolors.ListedColormap(colors[:len(np.unique(Y))])
    xx1_grid, xx2_grid = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1),
                                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1))
    z_grid = classifier.predict(np.array([xx1_grid.ravel(), xx2_grid.ravel()]).T)
    z_grid = z_grid.reshape(xx1_grid.shape)
    
    if basis == 2:
        curent_view.contourf(xx1_grid, xx2_grid, z_grid, cmap=cmaps, alpha=0.3)

        for (index, cls) in enumerate(np.unique(Y)):
            Z_array = np.linspace(0, len(X[Y==cls, 0]))
            curent_view.scatter(X[Y==cls, 0], X[Y==cls, 1], marker=markers[index],
                        c=cmaps(index), alpha=0.75, s=12.23, label=cls)

            if test_idx:
                x_test, y_test = X[test_idx, :], Y[test_idx]
                curent_view.scatter(x_test[:, 0], x_test[:, 1], linewidths=1, edgecolors="black", alpha=0.5, marker="o", label=f"test elem [{index}]")

    elif basis == 3:
        curent_view.plot_surface(xx1_grid, xx2_grid, z_grid, cmap=cmaps, alpha=0.3)

        for (index, cls) in enumerate(np.unique(Y)):
            Z_array = np.linspace(0, len(X[Y==cls, 0]))
            curent_view.scatter(X[Y==cls, 0], X[Y==cls, 1], 1, marker=markers[index],
                        c=cmaps(index), alpha=0.75, s=12.23, label=cls)

            if test_idx:
                x_test, y_test = X[test_idx, :], Y[test_idx]
                curent_view.scatter(x_test[:, 0], x_test[:, 1], 1, linewidths=1, edgecolors="black", alpha=0.5, marker="o", label=f"test elem [{index}]")


data = datasets.load_iris()
X = data.data[:, [2, 3]]
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
fnn = StandardScaler()
fnn.fit(X_train)

X_train_std = fnn.transform(X_train)
X_test_std = fnn.transform(X_test)

X_combined = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))

smn = SVC(kernel="rbf", random_state=0, C=1.0)
pnn = Perceptron(max_iter=40, eta0=0.01, random_state=0)
log = LogisticRegression(C=1.0, random_state=0)

log.fit(X_train_std, Y_train)
pnn.fit(X_train_std, Y_train)
smn.fit(X_train_std, Y_train)
generate_desion(X_combined, Y_combined, 
                classifier=smn, test_idx=range(105, 150), curent_view=axis_3d)
generate_desion(X_combined, Y_combined, 
                classifier=pnn, test_idx=range(105, 150), curent_view=axis_3d_01)
generate_desion(X_combined, Y_combined, 
                classifier=log, test_idx=range(105, 150), curent_view=axis_3d_02)

plt.legend(loc="lower center")
plt.show()







