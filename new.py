import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn")
phi, theta = np.linspace(0, np.pi*360, 360), np.linspace(0, np.pi*360, 360)
phi, theta = np.meshgrid(phi, theta)

def B_magnatic_induction(neu, neu1, Ix=[], Iy=[], Iz=[], R=4.5):

    if len(Iz) != 0:
        Bx = ((neu * neu1) * Ix) / (4 * 5 * np.pi)
        By = ((neu * neu1) * Iy) / (4 * 5 * np.pi)
        Bz = ((neu * neu1) * Iz) / (4 * 5 * np.pi)

        return Bx, By, Bz
    
    else:
        Bx = ((neu * neu1) * Ix) / (4 * 5 * np.pi)
        By = ((neu * neu1) * Iy) / (4 * 5 * np.pi)

        return Bx, By

figure = plt.figure()
axis_3d = figure.add_subplot(projection="3d")

axis_3d.set_xlabel("X", color="b")
axis_3d.set_ylabel("Y", color="r")
axis_3d.set_zlabel("Z", color="g")

x, y, z = np.meshgrid(np.linspace(-np.pi, np.pi, 14), np.linspace(-np.pi, np.pi, 14), 
                      np.linspace(-np.pi, np.pi, 14))

i, j, k = B_magnatic_induction(2, 2.5, x, y, z, 4.5)[0], B_magnatic_induction(2, 2.5, x, y, z, 4.5)[1], B_magnatic_induction(2, 2.5, x ,y, z, 4.5)[2]
B_vector = np.stack((i, j, k), axis=0).T
I_vector = np.stack((x, y, z), axis=0).T
Eamper_vector = np.cross(I_vector, B_vector)
print(Eamper_vector, B_vector, I_vector)

axis_3d.quiver(x, z, y, i, j, k, color="gray", alpha=0.78)

axis_3d.scatter(0, 0, 0, color="red", s=200)
axis_3d.quiver(0, 0, 0, 3, 0, 0, color="b")
axis_3d.quiver(0, 0, 0, 0, 3, 0, color="r")
axis_3d.quiver(0, 0, 0, 0, 0, 3, color="g")
axis_3d.quiver(-np.pi, -np.pi, -np.pi, 6, 0, 0, color="b")
axis_3d.quiver(-np.pi, -np.pi, -np.pi, 0, 6, 0, color="r")
axis_3d.quiver(-np.pi, -np.pi, -np.pi, 0, 0, 6, color="g")

plt.show()

