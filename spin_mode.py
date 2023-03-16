import numpy as np
import matplotlib.pyplot as plt
import math as mt
import matplotlib.animation as animo

plt.style.use("dark_background")
class Spin_mode():
    def __init__(self, projections=False, cmap=None, alpha=0,
                  scatter_mode=False, cmap_for_projection_z=None, cmap_for_projection_x=None,
                  cmap_for_projection_y=None, cmap_for_scatter=None, pr_onX=False,
                  pr_onY=False, pr_onZ=False, distrib_on_z=False,
                  distrib_on_y=False, distrib_on_x=False, scat_point_size=0.78,
                  rotation_axis=None) -> None:
        
        self.projections = projections
        self.cmap = cmap
        self.cmap_pr_z = cmap_for_projection_z
        self.cmap_pr_x = cmap_for_projection_x
        self.cmap_pr_y = cmap_for_projection_y
        self.cmap_scat = cmap_for_scatter
        self.scatter_mode = scatter_mode
        self.scat_s = scat_point_size
        self.rotation_axis = rotation_axis

        self.distrib_x = distrib_on_x
        self.distrib_y = distrib_on_y
        self.distrib_z = distrib_on_z

        self.on_x = pr_onX
        self.on_y = pr_onY
        self.on_z = pr_onZ

        self.alpha = alpha
        self.figure = plt.figure()
        self.axis_3d = self.figure.add_subplot(projection="3d")

    
    def rotation_matrix_(self, A, angle):
        rot_mat = np.array([
            [-mt.sin(np.pi * angle) ** 2, mt.sin(np.pi * angle) * mt.cos(np.pi * angle), mt.cos(np.pi * angle)],
            [mt.cos(np.pi * angle) ** 2 + mt.sin(np.pi * angle) ** 2, mt.cos(np.pi * angle) * mt.sin(np.pi * angle) + mt.cos(np.pi * angle) ** 2 * mt.sin(np.pi * angle), -mt.sin(np.pi * angle) ** 2],
            [-mt.sin(np.pi * angle) * mt.sin(np.pi * angle) - mt.sin(np.pi * angle) * mt.cos(np.pi * angle) ** 2, -mt.sin(np.pi * angle) ** 2 + mt.cos(np.pi * angle) ** 3, -mt.cos(np.pi * angle) * mt.sin(np.pi * angle)]
            ])
        return rot_mat.dot(A)
    
    def rotation_on_x_(self, A, angle):
        rot_mat = np.array([
            [1, 0, 0,],
            [0, mt.cos(np.pi * angle), mt.sin(np.pi * angle)],
            [0, -mt.sin(np.pi * angle), mt.cos(np.pi * angle)]
        ])
        return rot_mat.dot(A)
    
    def rotation_on_y_(self, A, angle):
        rot_mat = np.array([
            [mt.cos(angle * np.pi), 0, mt.sin(angle * np.pi)],
            [0, 1, 0],
            [-mt.sin(angle * np.pi), 0, mt.cos(np.pi * angle)]
        ])
        return rot_mat.dot(A)
    
    def rotation_on_z_(self, A, angle):
        rot_mat = np.array([
            [0, 0, 1],
            [mt.cos(np.pi * angle), mt.sin(np.pi * angle), 0],
            [-mt.sin(angle * np.pi), mt.cos(angle * np.pi), 0]
        ])
        return rot_mat.dot(A)
    
    def anim_(self, i):

        self.axis_3d.clear()
        phi, theta = np.linspace(np.pi, -np.pi, 100), np.linspace(np.pi, -np.pi)
        phi, theta = np.meshgrid(phi, theta)
        
        x_grid = np.sin(phi) + np.cos(theta) * np.sin(phi)
        y_grid = np.cos(phi) * np.sin(theta) + np.sin(theta)
        z_grid = np.cos(theta)

        core_array = [cords for cords in zip(x_grid, y_grid, z_grid)]
        core_array = np.asarray(core_array)

        for point_number in range(len(core_array)):
            if self.rotation_axis == "x":
                core_array[point_number] = self.rotation_on_x_(core_array[point_number], i)
            elif self.rotation_axis == "y":
                core_array[point_number] = self.rotation_on_y_(core_array[point_number], i)
            elif self.rotation_axis == "z":
                core_array[point_number] = self.rotation_on_z_(core_array[point_number], i)
            else:
                core_array[point_number] = self.rotation_on_y_(core_array[point_number], i)
        
        result_x = core_array[:, 0]
        result_y = core_array[:, 1]
        result_z = core_array[:, 2]

        self.axis_3d.plot_surface(result_x, result_y, result_z, cmap=self.cmap, alpha=self.alpha)
        if self.projections == True:
            if self.on_z == True:
                self.axis_3d.contourf(result_x, result_y, result_z, zdir="z", offset=-2, cmap=self.cmap_pr_z)
            if self.on_x == True:
                self.axis_3d.contourf(result_x, result_y, result_z, zdir="x", offset=-2, cmap=self.cmap_pr_x)
            if self.on_y == True:
                self.axis_3d.contourf(result_x, result_y, result_z, zdir="y", offset=-2, cmap=self.cmap_pr_y)

        if self.scatter_mode == True:
            if self.distrib_x == True:
                self.axis_3d.scatter(result_x, result_y, result_z, c=result_x, cmap=self.cmap_scat, s=self.scat_s)
            elif self.distrib_y == True:
                self.axis_3d.scatter(result_x, result_y, result_z, c=result_y, cmap=self.cmap_scat, s=self.scat_s)
            elif self.distrib_z == True:
                self.axis_3d.scatter(result_x, result_y, result_z, c=result_z, cmap=self.cmap_scat, s=self.scat_s)
        
    def run(self):
        anim = animo.FuncAnimation(self.figure, self.anim_, interval=100)
        plt.show()
        
if __name__ == "__main__":
    obj = Spin_mode(projections=False, pr_onX=False, pr_onY=False, pr_onZ=True, scatter_mode=True, distrib_on_x=True, cmap_for_projection_x="magma",
                    cmap_for_projection_y="magma", cmap_for_projection_z="magma", cmap_for_scatter="binary", cmap="twilight", alpha=0.67, scat_point_size=0.12).run()

    
        