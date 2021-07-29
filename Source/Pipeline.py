"""
@author: Akshay Mor
"""

from re import X
from typing import IO
from matplotlib.tri.triinterpolate import LinearTriInterpolator
import meshio
from numpy.ma.core import array
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.tri import CubicTriInterpolator
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd
import os
import io
import cv2
from utilities import getFiles, numericalSort, relative_error

class Pipeline:

    def __init__(self, fileNames = None, mesh:dict = None, triangulation = None, inputData:dict = None,
                         predictions: dict=None, xi:np.ndarray = None, yi:np.ndarray = None):

        self.fileNames = fileNames
        self.mesh = mesh
        self.triangulation = triangulation
        self.mesh_temp = None
        self.xi = None
        self.yi = None
        self.input_data = {
            "x_star": None,
            "y_star": None,
            "t_star": None,
            "C_star": None,
            "U_star": None,
            "V_star": None,
            "P_star": None
        }
        self.predictions = {
            "C_pred": None,
            "U_pred": None,
            "V_pred": None,
            "P_pred": None
        }
        self.test_data = {
            "x_test": None,
            "y_test": None,
            "t_test": None,
            "C_test": None,
            "U_test": None,
            "V_test": None,
            "P_test": None
        }
        self.errors = {
            "error_c": None,
            "error_u": None,
            "error_v": None,
            "error_p": None
        }
        self.settings = {
            "interpolator": None,
            "smoothing_algorithm": None
        }


    # get data for training from vtu files
    def extractData(self, fileNames):
        # read files
        x_star = pd.DataFrame().astype(np.float)
        y_star = pd.DataFrame().astype(np.float)
        t_star = pd.DataFrame().astype(np.float)

        # create empty dictionaries for C, U, V and P
        C_star = pd.DataFrame().astype(np.float)
        U_star = pd.DataFrame().astype(np.float)
        V_star = pd.DataFrame().astype(np.float)
        P_star = pd.DataFrame().astype(np.float)

        if (fileNames == None and self.fileNames == None): raise IOError("No files have been provided to the pipeline for reading")
        else:

            # if fileNames is not passed as argument, use existing filenames of object
            if (fileNames == None):
                if len(fileNames) != 0:
                    mesh = meshio.read(fileNames[1])
                    x_star.loc[:,1] = np.array(mesh.points[:,0])
                    y_star.loc[:,1] = np.array(mesh.points[:,1])
                    t_star.loc[:,1] = (np.linspace(0, 25, num=251))

                    it = 0
                    for file in fileNames:

                        # read file
                        mesh = meshio.read(file)

                        # append C, U, V & P from file to column in output
                        C_star.loc[:,it] = np.array(mesh.point_data["Con"])
                        U_star.loc[:,it] = np.array(mesh.point_data["Vel"][:,0]) # note Vel is a dict itself of 3 columns: x, y, z
                        V_star.loc[:,it] = np.array(mesh.point_data["Vel"][:,1])
                        P_star.loc[:,it] = np.array(mesh.point_data["Pres"])
                        it += 1
                else:
                    print("There are no files to get data from!")
                    pass

            # otherwise use the provided fileNames and overwrite existing self.input_data
            else:
                if len(fileNames) != 0:
                    mesh = meshio.read(fileNames[1])
                    x_star.loc[:,0] = np.array(mesh.points[:,0])
                    y_star.loc[:,0] = np.array(mesh.points[:,1])
                    t_star.loc[:,0] = (np.linspace(0, 25, num=251))

                    for i in fileNames:

                        # read file
                        mesh = meshio.read(i)

                        # append C, U, V & P from file to column in output
                        C_star.loc[:,i] = np.array(mesh.point_data["Con"])
                        U_star.loc[:,i] = np.array(mesh.point_data["Vel"][:,0]) # note Vel is a dict itself of 3 columns: x, y, z
                        V_star.loc[:,i] = np.array(mesh.point_data["Vel"][:,1])
                        P_star.loc[:,i] = np.array(mesh.point_data["Pres"])
                else:
                    print("There are no files to get data from!")
                    pass
        
        self.input_data["x_star"] = x_star
        self.input_data["y_star"] = y_star
        self.input_data["t_star"] = t_star
        self.input_data["C_star"] = C_star
        self.input_data["U_star"] = U_star
        self.input_data["V_star"] = V_star
        self.input_data["P_star"] = P_star

    

    # creates a mesh template for the Pipeline object
    # pass in the array of filenames to pick one and use the coordinates and cell format as a template for an output mesh
    def createMeshTemplate(self):
            if (self.fileNames[0].endswith('.vtu')):
                self.mesh_temp = meshio.read(self.fileNames[0])

                # clear points data for the output mesh
                self.mesh_temp.point_data = None
            else:
                raise IOError("Incorrect file type to create mesh template! Only .vtu files are supported at this time")

    # returns to a variable, a mesh template with all the correct xy coordinates and cells; points_data is empty for filling
    def meshTemplate(self):
        if (self.fileNames[0].endswith('.vtu')):
            mesh_template = meshio.read(self.fileNames[0])

            # clear points data
            mesh_template.point_data = None
            return mesh_template
        else:
            raise IOError("Incorrect file type to create mesh template!")

    # creates triangulation for Pipleline object
    # mesh passed in here will override the mesh belonging to the object
    def triangulate(self, mesh):
        if (mesh == None):
            ## assume a mesh exists within the object
            if(self.triangulation == None):
                triangles = self.mesh.cells_dict["triangle"]
                self.triangulation = Triangulation(self.mesh.points[:,0], self.mesh.points[:,1], triangles=triangles)
            else: 
                raise IOError("A triangulation already exists within object")
        else:
            ## if a mesh DOES NOT within the object
            if(self.triangulation == None):
                triangles = mesh.cells_dict["triangle"]
                self.triangulation = Triangulation(mesh.points[:,0], mesh.points[:,1], triangles=triangles)
            else: 
                raise IOError("A triangulation already exists within object")

    # creates and returns a triangulation object to a variable
    def create_trigangulation(self, mesh):
        if (mesh == None):
            if (self.mesh == None):
                raise IOError("There is no mesh to use for triangulation! Try creating one first!")
            # use internal mesh
            triangles = self.mesh.cells_dict["triangle"]
            triangulation = Triangulation(self.mesh.points[:,0], self.mesh.points[:,1], triangles=triangles)
            return triangulation
        else:
            if (self.mesh == None):
                raise IOError("There is no mesh to use for triangulation! Try creating one first!")
            # use provided mesh
            triangles = mesh.cells_dict["triangle"]
            triangulation = Triangulation(mesh.points[:,0], mesh.points[:,1], triangles=triangles)
            return triangulation

    # returns x and y coordinates (constant for all timesteps)
    def get_coords(self, mesh):
        if mesh != None:
            x = mesh.points[:,0]
            y = mesh.points[:,1]
            return x, y
        else:
            raise IOError("No mesh provided!")

    # takes 2d C array and turns it into a series of interpolated surface images
    # def render_frames(self, C, interpolator:str, xi = None, yi = None, it=0, triangulation = None, smoothing_algo:str=None):

    #     if (xi == None or yi == None):
    #         # create some defaults
    #         xi, yi = np.meshgrid(np.linspace(min(self.x),max(self.x),num=1000),np.linspace(min(self.y),max(self.y),num=1000))

    #     if (C == None and self.C == None):
    #         raise os.error("There are no C values to interpolate and render with")
    
    #     if smoothing_algo == "geom" or None: smoothing = "geom"
    #     if smoothing_algo == "min_E": smoothing = "min_E"

    #     if (triangulation != None):
    #         # if triangulation passed
    #         if interpolator == "linear" or None: interpolation = LinearTriInterpolator(self.triangulation, self.C)
    #         if interpolator == "cubic": interpolation = CubicTriInterpolator(self.triangulation, C, kind=smoothing)
    #     else:
    #         # use own triangulation
    #         if interpolator == "linear" or None: interpolation = LinearTriInterpolator(self.triangulation, self.C)
    #         if interpolator == "cubic": interpolation = CubicTriInterpolator(self.triangulation, C, kind=smoothing)
        
    #     os.mkdir("Frames")
    #     os.chdir("Frames")
    #     res = interpolation(xi, yi,)
    #     plt.contourf(xi, yi, res, vmax=1, vmin=0)
    #     plt.set_cmap('plasma')
    #     plt.savefig('Frames/_'+str(it)+'.png')

    #     print("Frame " + str(it) + ": done")

    def calculate_errors(self, predictions=None, test=None):

        # if args passed, override self
        if (predictions != None and test != None):
            errors = {
            "error_c" : relative_error(predictions["C_pred"], test["C_test"]),
            "error_u" : relative_error(predictions["U_pred"], test["U_test"]),
            "error_v" : relative_error(predictions["V_pred"], test["V_test"]),
            "error_p" : relative_error(predictions["P_pred"], test["P_test"])
            }
            return errors

        else:
            self.errors["error_c"] = relative_error(self.predictions["C_pred"], self.test_data["C_test"])
            self.errors["error_u"] = relative_error(self.predictions["U_pred"], self.test_data["U_test"])
            self.errors["error_v"] = relative_error(self.predictions["V_pred"], self.test_data["V_test"])
            self.errors["error_p"] = relative_error(self.predictions["P_pred"], self.test_data["P_test"])


    # def render_array(triangulation, xi, yi, C, it, interpolator:str, smoothing_algo:str=None):

    #     if interpolator == "linear": interpolation = LinearTriInterpolator(triangulation, C)

    #     if smoothing_algo == "geom": smoothing = "geom"
    #     if smoothing_algo == "min_E": smoothing = "min_E"
    #     if interpolator == "cubic": interpolation = CubicTriInterpolator(triangulation, C, kind=smoothing)
        
    #     res = interpolation(xi, yi,)
    #     fig = plt.contourf(xi, yi, res, vmax=1, vmin=0)
    #     fig.set_cmap('plasma')
        
    #     with io.BytesIO() as buff:
    #         fig.savefig(buff, format='raw')
    #         buff.seek(0)
    #         data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    #     w, h = fig.canvas.get_width_height()
    #     im = data.reshape((int(h), int(w), -1))
    #     cv2.imwrite(("_"+str(it)+".tiff"), cv2.GaussianBlur(im, (10,10), 0))

    #     print("Frame " + str(it) + ": done")


    # generate frames from data
    # def generate_frames(self):

    #     settings = {
    #         "interpolator": interpolator,
    #         "smoothing": smoothing,
    #     }

    #     sampling_mesh = meshio.read(self.fileNames[1])
    #     x, y = self.get_coords(sampling_mesh)
    #     xi, yi = np.meshgrid(np.linspace(min(x),max(x),num=1000),np.linspace(min(y),max(y),num=1000))
    #     triangulation = triangulate(sampling_mesh)
    #     it = 0

    #     for file in fNames:
    #         C = meshio.read(file).point_data["Con"]
    #         render_frames(triangulation, xi, yi, C, it, settings["interpolator"], settings["smoothing"])
    #         it += 1

    # def generate_frame_arrays(fNames, interpolator:str, smoothing:str=None):

    #     settings = {
    #         "interpolator": interpolator,
    #         "smoothing": smoothing,
    #     }

    #     sampling_mesh = meshio.read(fNames[1])
    #     x, y = get_coords(sampling_mesh)
    #     xi, yi = np.meshgrid(np.linspace(min(x),max(x),num=1000),np.linspace(min(y),max(y),num=1000))
    #     triangulation = triangulate(sampling_mesh)
    #     it = 0

    #     for file in fNames:
    #         C = meshio.read(file).point_data["Con"]
    #         render_array(triangulation, xi, yi, C, it, settings["interpolator"], settings["smoothing"])
    #         it += 1


    def mat2vtu(self, mesh_template=None, predictions: dict=None):
        if (mesh_template == None and results == None):
            if (self.mesh_temp == None): raise IOError("No mesh template to use! Try creating one first!")
            points = self.mesh_temp.points
            output = pd.DataFrame(float)
            for i in range(251):
                output[:, 1,2] = pd.DataFrame(points)
                output.loc[:, "Con"] = self.predictions["C_pred"][:,i]
                output.loc[:, "Pres"] = self.predictions["P_pred"][:,i]
                output.loc[:, "Vel"] = np.linalg.norm((np.concatenate(self.predictions["U_pred"], self.predictions["V_pred"], axis=1)), axis=1)
                cells = mesh_template.cells
                point_data = {
                    "Con" : output.loc[:, "Con"],
                    "Pres": output.loc[:, "Pres"],
                    "Vel" : output.loc[:, "Con"]
                }
                mesh = meshio.Mesh(
                    points,
                    cells,
                    point_data
                )
                mesh.write("_"+str(i)+".vtu")
        # if neither mesh_template nor mat_data exist,
        else:
            return


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