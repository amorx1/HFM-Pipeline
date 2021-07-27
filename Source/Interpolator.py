from re import X
from typing import IO
from matplotlib.tri.triinterpolate import LinearTriInterpolator
import meshio
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.tri import CubicTriInterpolator
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd
import os
import io
import cv2
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

# interpolator
def render(triangulation, xi, yi, C, it, interpolator:str, smoothing_algo:str=None):

    if interpolator == "linear": interpolation = LinearTriInterpolator(triangulation, C)

    if smoothing_algo == "geom": smoothing = "geom"
    if smoothing_algo == "min_E": smoothing = "min_E"
    if interpolator == "cubic": interpolation = CubicTriInterpolator(triangulation, C, kind=smoothing)
    
    res = interpolation(xi, yi,)
    plt.contourf(xi, yi, res, vmax=1, vmin=0)
    plt.set_cmap('plasma')
    plt.savefig('_'+str(it)+'.png')

    print("Frame " + str(it) + ": done")

def render_array(triangulation, xi, yi, C, it, interpolator:str, smoothing_algo:str=None):

    if interpolator == "linear": interpolation = LinearTriInterpolator(triangulation, C)

    if smoothing_algo == "geom": smoothing = "geom"
    if smoothing_algo == "min_E": smoothing = "min_E"
    if interpolator == "cubic": interpolation = CubicTriInterpolator(triangulation, C, kind=smoothing)
    
    res = interpolation(xi, yi,)
    fig = plt.contourf(xi, yi, res, vmax=1, vmin=0)
    fig.set_cmap('plasma')
    
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    cv2.imwrite(("_"+str(it)+".tiff"), cv2.GaussianBlur(im, (10,10), 0))

    print("Frame " + str(it) + ": done")

# generate frames from data
def generate_frames(fNames, interpolator:str, smoothing:str=None):

    settings = {
        "interpolator": interpolator,
        "smoothing": smoothing,
    }

    sampling_mesh = meshio.read(fNames[1])
    x, y = get_coords(sampling_mesh)
    xi, yi = np.meshgrid(np.linspace(min(x),max(x),num=1000),np.linspace(min(y),max(y),num=1000))
    triangulation = triangulate(sampling_mesh)
    it = 0

    for file in fNames:
        C = meshio.read(file).point_data["Con"]
        render(triangulation, xi, yi, C, it, settings["interpolator"], settings["smoothing"])
        it += 1

def generate_frame_arrays(fNames, interpolator:str, smoothing:str=None):

    settings = {
        "interpolator": interpolator,
        "smoothing": smoothing,
    }

    sampling_mesh = meshio.read(fNames[1])
    x, y = get_coords(sampling_mesh)
    xi, yi = np.meshgrid(np.linspace(min(x),max(x),num=1000),np.linspace(min(y),max(y),num=1000))
    triangulation = triangulate(sampling_mesh)
    it = 0

    for file in fNames:
        C = meshio.read(file).point_data["Con"]
        render_array(triangulation, xi, yi, C, it, settings["interpolator"], settings["smoothing"])
        it += 1

def process():
    return

def mat2vtu(mesh_template, results_mat):

    points = mesh_template.points
    for i in range(251):
        output = pd.DataFrame(points)
        output.loc[:, "Pres"] = results_mat["P_pred"][:,i]
        cells = mesh_template.cells
        point_data = {
            "Pres": output.loc[:, "Pres"]
        }
        mesh = meshio.Mesh(
            points,
            cells,
            point_data
        )
        mesh.write("_"+str(i)+".vtu")


def main():
    try:
        os.chdir("DATA")    # FOR TESTING ONLY
    except:
        raise IOError("Invalid directory")

    mesh_template = meshio.read("misc/mesh_template.vtu")
    results_mat = scipy.io.loadmat("training_120k_results_26_07_2021.mat")
    mat2vtu(mesh_template, results_mat)
    # fNames = parse_args()["fileNames"]

    # if fNames:
    #     try:
    #         generate_frame_arrays(fNames, "linear")

    #     except:
    #         raise RuntimeError("There was a problem rendering the results")

if __name__ == "__main__":
    main()