import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
import matplotlib.animation as animation

#надо программа должна выводить изменения траектории а сейчас она ее просто дублирует


figure = plt.figure()
axis_3d = figure.add_subplot(projection="3d")
                
def generate_trajactory():

    r = np.linspace(0, np.pi)
    phi = np.linspace(0, np.pi * 2)
    theta = np.linspace(0, np.pi * 2)
    iter = 0
    while True:
        #radiuse_choice = int(input("chouse the radiue of array"))
        
        if iter < 3:
            r *= rd.randint(0, 45)
            X = r * np.cos(phi) 
            Y = r * np.sin(theta)
            Z = np.linspace(0, np.pi, 100)

            with open("data.txt", "a") as file:
                for (x, y, z) in zip(X, Y, Z):
                    file.write(f"{x}\t{y}\t{z}\n")
        else:
             animo = animation.FuncAnimation(figure, trajectory_graph, interval=100)
             plt.show()
             break
        
        iter += 1

def trajectory_graph(i):

    
    with open("data.txt", "r") as file:
                data = pd.read_csv("data.txt", sep="\t")
                x_cores = np.asarray(data.iloc[:, 0] + i)
                y_cores = np.asarray(data.iloc[:, 1] + i)
                z_cores = np.asarray(data.iloc[:, 2] + i)

                axis_3d.plot(x_cores, y_cores, z_cores, color="red", alpha=0.67)

generate_trajactory()


