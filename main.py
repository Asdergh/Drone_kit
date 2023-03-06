import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd

class ADA():
    def __init__(self, eta=0.1, epochs=40, random_state=None, shuffle=True) -> None:
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.shuffle = shuffle
        if self.random_state:
            np.random.seed(self.random_state)
    
    def fit(self, X, Y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.epochs):
            if self.shuffle == True:
                self.shuffle_(X, Y)
            cost = []
            for (xi, true_label) in zip(X, Y):
                cost.append(self.update(xi, true_label))
                avg_cost = sum(cost) / len(cost)
                self.cost_.append(avg_cost)
        
        return self

    def shuffle_(self, X, Y):
        rand = np.random.permutation(len(Y))
        return X[rand], Y[rand]
    
    def update(self, xi, true_label):
        output = self.net_input(xi)
        error = (true_label - output)
        self.W_[1:] += self.eta * xi.dot(error)
        self.W_[0] += self.eta * error
        cost = 0.5 * (error ** 2)

        return cost


    def _initialize_weights(self, m):
        self.W_ = np.zeros(1 + m)
        self.W_initialized = True

    def net_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def activation(self, X):
        return self.net_input(X)
    
    def prediction(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    

if __name__ == "__main__":
    data = pd.read_csv("iris.data")
    X = data.iloc[0:100, [1, 2]].values
    Y = data.iloc[0:100, 4].values
    Y = np.where(Y == "Iris-setosa", 1, -1)
    print(Y, X)
    ada = ADA(eta=0.001, epochs=40).fit(X, Y)

    plt.plot(range(0, len(ada.cost_) + 1), ada.cost_, color="red", label="минимизация квдаратичной ошибки и стабилизация нейронной сети")
    plt.legend(loc="upper left")
    plt.show()

