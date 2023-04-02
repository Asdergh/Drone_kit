import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Online_Adaline():
    def __init__(self, eta=0.001, epochs=30, shuffle=False) -> None:
        self.eta = eta
        self.epochs = epochs
        self.shuffle = shuffle
    
    def fit(self, X, y):
        pass
    

    def initialie_weighs(self, m):
        self.W_ = np.zeros(1 + m)
        self.W_inialized = True
    
    def shuffle_(self, X, y):
        rand = np.permitation #<-----------
