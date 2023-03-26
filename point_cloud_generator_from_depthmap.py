import cv2
import open3d as o3d
import numpy as np
import json as js
import random as rd



def generate_data_in_json(count_of_points, normal_distrib_mode, sigma_x: float, mean_x: int,
                          sigma_y: float, mean_y: int, sigma_z: float, mean_z: int, default_normal_params=False) -> None:
    data = {}
    if normal_distrib_mode == True:
        if default_normal_params == False:
            x_gaussian = np.random.normal(mean_x, sigma_x, count_of_points)
            y_gaussian = np.random.normal(mean_y, sigma_y, count_of_points)
            z_gaussian = np.random.normal(mean_z, sigma_z, count_of_points)
            core_array = np.dstack((x_gaussian, y_gaussian, z_gaussian))
            print(core_array)
        else:
            sigma_rand = np.arange(0.0, 1.0, 100)
            x_gaussian = np.random.normal(rd.randint(45, 360), rd.choice(sigma_rand), count_of_points)
            y_gaussian = np.random.normal(rd.rnadint(45, 360), rd.choice(sigma_rand), count_of_points)
            z_gaussian = np.random.normal(rd.randint(45, 360), rd.choice(sigma_rand), count_of_points)
            core_array = np.dstack((x_gaussian, y_gaussian, z_gaussian))
            print(core_array)
    else:
        core_array = np.random.rand(count_of_points, 3)
 
    for i in range(count_of_points):
        data[str(i)] = {
            "X": core_array[0, i, 0],
            "Y": core_array[0, i, 1],
            "Z": core_array[0, i, 2]
        }
        print(core_array[0, i, 0])
    with open("json_data_file.json", "w") as file:
        js.dump(data, file)

def generate_data(count_of_points, normal_distrib_mode, sigma_x: float, mean_x: int,
                          sigma_y: float, mean_y: int, sigma_z: float, mean_z: int, default_normal_params=False) -> None:
    if normal_distrib_mode == True:
        if default_normal_params == False:
            x_gaussian = np.random.normal(mean_x, sigma_x, count_of_points)
            y_gaussian = np.random.normal(mean_y, sigma_y, count_of_points)
            z_gaussian = np.random.normal(mean_z, sigma_z, count_of_points)
            core_array = np.dstack((x_gaussian, y_gaussian, z_gaussian))
            print(core_array)
        else:
            sigma_rand = np.arange(0.0, 1.0, 100)
            x_gaussian = np.random.normal(rd.randint(45, 360), rd.choice(sigma_rand), count_of_points)
            y_gaussian = np.random.normal(rd.rnadint(45, 360), rd.choice(sigma_rand), count_of_points)
            z_gaussian = np.random.normal(rd.randint(45, 360), rd.choice(sigma_rand), count_of_points)
            core_array = np.dstack((x_gaussian, y_gaussian, z_gaussian))
            print(core_array)
    else:
        core_array = np.random.rand(count_of_points, 3)
    
    with open("DATA.txt", "w") as file:
        for (x, y, z) in core_array:
            file.write(f"{x}\t{y}\t{z}\n")



#generate_data_in_json(count_of_points=12000, normal_distrib_mode=True, mean_x=12, sigma_x=0.67, 
#                      mean_y=12, sigma_y=1.24, mean_z=67, sigma_z=1.12)

#with open("json_data_file.json", "r") as file:
#    cores_dict = js.load(file)

generate_data(count_of_points=12000, normal_distrib_mode=False, mean_x=12, sigma_x=0.67, mean_y=12, sigma_y=1.24, mean_z=67, sigma_z=1.12)
pcd = o3d.io.read_point_cloud("DATA.txt", format="xyz")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[16.6172, 6.0475, 56.532],
                                  up=[-0.0694, -0.9768, 0.2024])
   
    

        



    