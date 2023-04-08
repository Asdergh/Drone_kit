import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")
class Perseptron():
    def __init__(self, eta=0.01, epochs=40) -> None:
        self.eta = eta
        self.epochs = epochs
    
    def fit(self, X, y):
        self.W_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        for i in range(self.epochs):
            print(f"epoche__/{i}")
            errors = 0
            for (xi, true_label) in zip(X, y):
                update = self.eta * (true_label - self.net_input(xi))
                self.W_[1:] += update * xi
                self.W_[0] += update
                errors += int(update != 0.0)
            self.cost_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def prediciton(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    

class Addaline():
    def __init__(self, eta=0.0001, epochs=50) -> None:
        self.eta = eta
        self.epochs = epochs
    
    def fit(self, X, y):
        self.W_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        for _ in range(self.epochs):
            update = y - self.net_input(X)
            self.W_[1:] += self.eta * X.T.dot(update)
            self.W_[0] += self.eta * update.sum()
            cost = (update ** 2).sum() * 0.5
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def prediction(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    
    


data = pd.read_csv("iris.data")
X = np.array(data.iloc[0:100, [0, 2]])
y = np.array(data.iloc[0:100, [4]])
y = np.where(y == "Iris-setosa", 1, -1)

ppn = Perseptron(eta=0.001, epochs=45).fit(X, y)
ada = Addaline(eta=0.001, epochs=45).fit(X, y)
plt.plot(range(0, len(ada.cost_)), ada.cost_, marker="x", color="green", label="errors reasing")
plt.legend(loc="upper left")
plt.show()
