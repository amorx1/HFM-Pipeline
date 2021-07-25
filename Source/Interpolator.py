from matplotlib.tri.triinterpolate import LinearTriInterpolator
import meshio
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.tri import CubicTriInterpolator
from matplotlib.tri import Triangulation
import numpy as np
import os
from utilities import parse_args, numericalSort

# returns triangulation object
def triangulate(mesh):
    triangles = mesh.cells_dict["triangle"]
    triangulation = Triangulation(mesh.points[:,0], mesh.points[:,1], triangles=triangles)
    return triangulation

# returns x and y coordinates (constant for all timesteps)
def get_coords(mesh):
    x = mesh.points[:,0]
    y = mesh.points[:,1]
    return x, y

# returns concentration from each file (not constant)
def get_Conc(mesh):
    C = mesh.point_data["Con"]
    return C

def render(triangulation, xi, yi, C, it):

    # geom_interpolation = CubicTriInterpolator(triangulation, C, kind="geom")
    min_e_interpolation = CubicTriInterpolator(triangulation, C, kind="min_E")
    # linear_interpolation = LinearTriInterpolator(triangulation, C)

    # geom_res = geom_interpolation(xi, yi)
    min_e_res = min_e_interpolation(xi, yi)
    # linear_res = linear_interpolation(xi, yi)

    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # axs = axs.flatten()

    # Triangulation
    # plt.tricontour(triangulation, C)
    # plt.triplot(triangulation, 'ko-')

    # axs[1].contourf(xi, yi, geom_res)
    # # axs[1].plot(xi, yi, 'k-', lw=0.5, alpha=0.1)
    # # axs[1].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.1)
    # axs[1].set_title("Cubic interpolation,\nkind='geom")

    plt.contourf(xi, yi, min_e_res, vmax=1, vmin=0)
    # plt.plot(xi, yi, 'k-', lw=0.5, alpha=0.1)
    # plt.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.1)
    # plt.title = "Cubic interpolation,\nkind='min_E'"
    plt.set_cmap('plasma')
    plt.savefig('_'+str(it)+'.png')

    # axs[3].contourf(xi, yi, linear_res)
    # # axs[1].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
    # # axs[1].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
    # axs[3].set_title("Linear interpolation")

    print("Done")

def main():
    try:
        os.chdir("DATA")    # FOR TESTING ONLY
    except:
        print("Invalid directory")

    fNames = parse_args()["fileNames"]
    sampling_mesh = meshio.read(fNames[1])
    x, y = get_coords(sampling_mesh)
    xi, yi = np.meshgrid(np.linspace(min(x),max(x),num=1000),np.linspace(min(y),max(y),num=1000))
    triangulation = triangulate(sampling_mesh)
    
    it = 0
    for file in fNames:
        C = meshio.read(file).point_data["Con"]
        render(triangulation, xi, yi, C, it)
        it += 1

if __name__ == "__main__":
    main()