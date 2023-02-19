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
        self.errors = []

        for _ in range(self.epochs):
            error = 0
            for(xi, true_label) in zip(X, Y):
                update = self.eta * ()

    def net_input(self, X):
        return np.dot(X, self.W[1:]) + self.W[0]
    def prediction(self, X):
        return np.where(self.net_iput(X) >= 0.0, 1, -1)