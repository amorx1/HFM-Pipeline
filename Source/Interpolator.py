from matplotlib.tri.triinterpolate import LinearTriInterpolator
import meshio
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.tri import CubicTriInterpolator
from matplotlib.tri import Triangulation
import numpy as np

def extract_data():
    mesh = meshio.read("DATA/2D_LA_1inlet_lowRe-152.vtu")
    triangles = mesh.cells_dict["triangle"] # array
    x = mesh.points[:,0]
    y = mesh.points[:,1]
    C = mesh.point_data["Con"]

    return triangles, x, y, C

def visualise(triangles, x, y, C):

    triangulation = Triangulation(x, y, triangles=triangles)
    xi, yi = np.meshgrid(np.linspace(min(x),max(x),num=1000),np.linspace(min(y),max(y),num=1000))
    geom_interpolation = CubicTriInterpolator(triangulation, C, kind="geom")
    min_e_interpolation = CubicTriInterpolator(triangulation, C, kind="min_E")
    linear_interpolation = LinearTriInterpolator(triangulation, C)
    geom_res = geom_interpolation(xi, yi)
    min_e_res = min_e_interpolation(xi, yi)
    linear_res = linear_interpolation(xi, yi)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs = axs.flatten()

    # Triangulation
    axs[0].tricontour(triangulation, C)
    axs[0].triplot(triangulation, 'ko-')

    # 
    axs[1].contourf(xi, yi, geom_res)
    # axs[1].plot(xi, yi, 'k-', lw=0.5, alpha=0.1)
    # axs[1].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.1)
    axs[1].set_title("Cubic interpolation,\nkind='geom")

    axs[2].contourf(xi, yi, min_e_res)
    # axs[2].plot(xi, yi, 'k-', lw=0.5, alpha=0.1)
    # axs[2].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.1)
    axs[2].set_title("Cubic interpolation,\nkind='min_E'")

    axs[3].contourf(xi, yi, linear_res)
    # axs[1].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
    # axs[1].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
    axs[3].set_title("Linear interpolation")

    print("Done")

def main():
    triangles, x, y, C = extract_data()
    visualise(triangles, x, y, C)
    plt.colorbar();
    plt.show()

if __name__ == "__main__":
    main()