import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim

plt.style.use("dark_background")
figure = plt.figure()
axis_3d = figure.add_subplot(projection="3d")
def animo(i):
    x, y, z = np.meshgrid(np.linspace(-np.pi, np.pi, 8),
                                np.linspace(-np.pi, np.pi, 8),
                                np.linspace(-np.pi, np.pi, 8))

    u = np.sin(x + i) * np.cos(y + i) * np.sin(z + i)
    v = -np.sin(x + i) * np.sin(y + i) * np.sin(z + i)
    w = np.sin(x + i) * np.sin(y + i) * np.cos(z + i)

    axis_3d.quiver(x, y, z, u, v, w, normalize=True, color="b")
animon = anim.FuncAnimation(figure, axis_3d, interval=8)
plt.show()


