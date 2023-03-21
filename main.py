import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.style.use("seaborn")
class Perceptron():

    def __init__(self, eta=0.01, epochs=40) -> None:
        self.eta = eta
        self.epochs = epochs
    
    def fit(self, X, y):
        
        self.W_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        for _ in range(self.epochs):
            for xi, true_label in zip(X, y):
                error = 0
                ouput = true_label - self.pure_input(xi)
                self.W_[1:] += self.eta * ouput * xi
                self.W_[0] += self.eta * ouput
                error += int(ouput >= 0.0)
            self.cost_.append(error)
        return self
    
    def pure_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def prediciton(self, X):
        return np.where(self.pure_input(X) >= 0.0, 1, -1)
    

data = pd.read_csv("iris.data")

X = data.iloc[0:100, [0, 2]]
X = np.asarray(X)
y = data.iloc[0:100, [4]]
y = np.asarray(y)
y = np.where(y == "Iris-setosa")
plt.xlabel("param 0")
plt.xlabel("param 2")

plt.scatter(X[0:50, 0], X[0:50, 1], color="red", marker="o", label="First elem")
plt.legend(loc="upper left")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", label="Second elem", marker="x")
plt.legend(loc="upper left")

plt.show()








    