import pandas as pd
import numpy as np
import open3d as o3d
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PointCloud():
    
    def __init__(self, count_of_points: int) -> None:
        self.count_of_points = count_of_points
        self.figure = plt.figure()
        self.axis_3d = self.figure.add_subplot(projection="3d")


    def generate_data(self):
        
        x_cores = np.zeros(self.count_of_points)
        y_cores = np.zeros(self.count_of_points)
        z_cores = np.zeros(self.count_of_points)

        #set the random walk
        for i in range(1, len(x_cores)):
            x_cores[i] = x_cores[i - 1] + rd.randint(0, 123) * rd.choice([-1, 1])
            y_cores[i] = y_cores[i - 1] + rd.randint(0, 123) * rd.choice([-1, 1])
            z_cores[i] = z_cores[i - 1] + rd.randint(0, 123) * rd.choice([-1, 1])
        

        with open("cloud_data.txt", "w") as file:
            for (x, y, z) in zip(x_cores, y_cores, z_cores):
                file.write(f"{x}\t{y}\t{z}\n")

    def rotation_on_x_(self, i):
        self.axis_3d.clear()
        rotated_cores = self.data.dot(np.array([
            [1, 0, 0],
            [0, np.cos(i), np.sin(i)],
            [0, -np.sin(i), np.cos(i)]
        ]))
         
        self.axis_3d.scatter(rotated_cores[0], rotated_cores[1], rotated_cores[2], c = rotated_cores[2], cmap="twilight", alpha=0.76, s=0.78)


    
    def visualise_cloud(self, file_name):
        
        """pcd = o3d.io.read_point_cloud(file_name)
        o3d.visualization.draw_geometries([pcd], zoom=0.3412, front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6272, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024])"""
        self.data = pd.read_csv("cloud_data.txt", delimiter="\t")
        cores = np.array([self.data.iloc[:, 0], self.data.iloc[:, 1], self.data.iloc[:, 2]])

        animo = animation.FuncAnimation(self.figure, self.rotation_on_x_, interval=100)
        plt.show()

    





pc = PointCloud(count_of_points=10000)
pc.generate_data()
pc.visualise_cloud("coud_data.txt")

data = pd.read_csv("cloud_data.txt", delimiter="\t")
X_cores = np.array(data.iloc[:, 0])
Y_cores = np.array(data.iloc[:, 1])
Z_cores = np.array(data.iloc[:, 2])
print(X_cores, Y_cores, Z_cores, sep="\n")