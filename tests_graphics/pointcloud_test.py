import open3d as o3d
import numpy as np
import cv2
import laspy



def generate_random_pointcloud_data(count_of_points, normal_mode=False, x_mean=0, y_mean=0, z_mean=0, deviation=0.01):

    if normal_mode == False:

        core_array = np.random.rand(count_of_points, 3)
        with open("point_cloud.txt", "w") as file:
            for (x, y, z) in core_array:
                file.write(f"{x}\t{y}\t{z}\n")

    elif normal_mode == True:

        x_cores = np.random.normal(x_mean, deviation, count_of_points)
        y_cores = np.random.normal(y_mean, deviation, count_of_points)
        z_cores = np.random.normal(z_mean, deviation, count_of_points)
        core_array = np.dstack((x_cores, y_cores, z_cores))
        print(core_array)
        with open("point_cloud_normal.txt", "w") as file:
            for (x, y, z) in core_array[0]:
                file.write(f"{x}\t{y}\t{z}\n")

generate_random_pointcloud_data(normal_mode=False, count_of_points=120000)
generate_random_pointcloud_data(normal_mode=True, x_mean=5, y_mean=6, z_mean=3, deviation=0.78, count_of_points=120000)


PG = o3d.geometry.PointCloud()
cores = laspy.read("points.las")
core_array = np.stack([cores.X, cores.Y, cores.Z], axis=0).transpose((1, 0))
PG.points = o3d.utility.Vector3dVector(core_array)
o3d.visualization.draw_geometries([core_array])
print(core_array)



        


