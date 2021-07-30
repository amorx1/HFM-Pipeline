"""
@author: Akshay Mor
"""

from re import X
import tensorflow as tf
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
import sys
import time
from Objects import *
from utilities import getFiles, numericalSort, relative_error, neural_net, mean_squared_error, Navier_Stokes_2D, tf_session

class Pipeline:

    def __init__(self, fileNames = None, mesh:dict = None, triangulation = None, inputData:dict = None,
                         predictions: dict=None, xi:np.ndarray = None, yi:np.ndarray = None, HFM = None):

        self.HFM = None
        self.fileNames = fileNames
        self.mesh = mesh
        self.triangulation = triangulation
        self.mesh_temp = None
        self.xi = None
        self.yi = None
        self.T = None
        self.N = None
        self.Inputs = InputData()
        self.TrainingData = TrainingData()
        self.TestData = TestData()
        self.Equations = Equations()
        self.Predictions = Predictions()
        self.input_data = {
            "x_star": None,
            "y_star": None,
            "t_star": None,
            "C_star": None,
            "U_star": None,
            "V_star": None,
            "P_star": None
        }
        # InputData()
        self.rearranged_inputs = {
            "X_star": None,
            "Y_star": None,
            "T_star": None,
            "C_star": None,
            "U_star": None,
            "V_star": None,
            "P_star": None
        }
        self.eqns = {
            "t_eqns": None,
            "x_eqns": None,
            "y_eqns": None
        }
        self.predictions = {
            "C_pred": None,
            "U_pred": None,
            "V_pred": None,
            "P_pred": None
        }
        self.train_data = {
            "t_data": None,
            "x_data": None,
            "y_data": None,
            "c_data": None,
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

    def setTN(self):
        self.T = self.Inputs.t_star.shape[0]
        self.N = self.Inputs.x_star.shape[0]

    # HFM object instantiated with input data filled
    def model(self):
        if self.HFM == None:
            if (self.input_data != None):
                self.HFM = HFM(t_data=self.rearranged_inputs["t_star"], x_data=self.rearranged_inputs["x_star"], y_data=self.rearranged_inputs["y_star"], c_data=self.rearranged_inputs["C_star"], t_eqns=self.eqns["t_eqns"], x_eqns=self.eqns["x_eqns"], y_eqns=self.eqns["y_eqns"], layers=self.layers, batch_size=self.batch_size, Pec=self.Pec, Rey=self.Rey)
            else:
                raise IOError("HFM cannot be created without input data!")
        else:
            raise IOError("An HFM object already exists within the Pipeline!")
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
        
        self.Inputs.x_star = x_star
        self.Inputs.y_star = y_star
        self.Inputs.t_star = t_star
        self.Inputs.C_star = C_star
        self.Inputs.U_star = U_star
        self.Inputs.V_star = V_star
        self.Inputs.P_star = P_star
    #rearrange data
    def rearrangeData(self):

        if(self.input_data != None):
            T = (self.input_data["t_star"]).shape[0]
            N = (self.input_data["x_star"]).shape[0]

            self.rearranged_inputs["T_star"] = np.tile(self.input_data["t_star"], (1,N)).T
            self.rearranged_inputs["X_star"] = np.tile(self.input_data["x_star"], (1,T))
            self.rearranged_inputs["Y_star"] = np.tile(self.input_data["y_star"], (1,T))

            # Training Data
            T_data = T
            N_data = N
            idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
            idx_x = np.random.choice(N, N_data, replace=False)
            t_data = self.rearranged_inputs["T_star"][:, idx_t][idx_x,:].flatten()[:,None]
            x_data = self.rearranged_inputs["X_star"][:, idx_t][idx_x,:].flatten()[:,None]
            y_data = self.rearranged_inputs["Y_star"][idx_x,:].flatten()[:,None]
            c_data = self.input_data["C_Star"][:, idx_t][idx_x,:].flatten()[:,None]

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
        if (mesh_template == None and predictions == None):
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
        # if mesh_template and predictions are provided
        else:
            return

class HFM:
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions
    
    def __init__(hfm, t_data, x_data, y_data, c_data,
                       t_eqns, x_eqns, y_eqns,
                       layers, batch_size,
                       Pec, Rey):
        
        # specs
        hfm.layers = layers
        hfm.batch_size = batch_size
        
        # flow properties
        hfm.Pec = Pec
        hfm.Rey = Rey
        
        # data
        [hfm.t_data, hfm.x_data, hfm.y_data, hfm.c_data] = [t_data, x_data, y_data, c_data]
        [hfm.t_eqns, hfm.x_eqns, hfm.y_eqns] = [t_eqns, x_eqns, y_eqns]
        
        # placeholders
        [hfm.t_data_tf, hfm.x_data_tf, hfm.y_data_tf, hfm.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [hfm.t_eqns_tf, hfm.x_eqns_tf, hfm.y_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        
        # physics "uninformed" neural networks
        hfm.net_cuvp = neural_net(hfm.t_data, hfm.x_data, hfm.y_data, layers = hfm.layers)
        
        [hfm.c_data_pred,
         hfm.u_data_pred,
         hfm.v_data_pred,
         hfm.p_data_pred] = hfm.net_cuvp(hfm.t_data_tf,
                                           hfm.x_data_tf,
                                           hfm.y_data_tf)
         
        # physics "informed" neural networks
        [hfm.c_eqns_pred,
         hfm.u_eqns_pred,
         hfm.v_eqns_pred,
         hfm.p_eqns_pred] = hfm.net_cuvp(hfm.t_eqns_tf,
                                           hfm.x_eqns_tf,
                                           hfm.y_eqns_tf)
        
        [hfm.e1_eqns_pred,
         hfm.e2_eqns_pred,
         hfm.e3_eqns_pred,
         hfm.e4_eqns_pred] = Navier_Stokes_2D(hfm.c_eqns_pred,
                                               hfm.u_eqns_pred,
                                               hfm.v_eqns_pred,
                                               hfm.p_eqns_pred,
                                               hfm.t_eqns_tf,
                                               hfm.x_eqns_tf,
                                               hfm.y_eqns_tf,
                                               hfm.Pec,
                                               hfm.Rey)
        
        # loss
        hfm.loss = mean_squared_error(hfm.c_data_pred, hfm.c_data_tf) + \
                    mean_squared_error(hfm.e1_eqns_pred, 0.0) + \
                    mean_squared_error(hfm.e2_eqns_pred, 0.0) + \
                    mean_squared_error(hfm.e3_eqns_pred, 0.0) + \
                    mean_squared_error(hfm.e4_eqns_pred, 0.0)
        
        # optimizers
        hfm.learning_rate = tf.placeholder(tf.float32, shape=[])
        hfm.optimizer = tf.train.AdamOptimizer(learning_rate = hfm.learning_rate)
        hfm.train_op = hfm.optimizer.minimize(hfm.loss)
        
        hfm.sess = tf_session()
        
    def train(hfm, total_time, learning_rate):
        
        N_data = hfm.t_data.shape[0]
        N_eqns = hfm.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, min(hfm.batch_size, N_data))
            idx_eqns = np.random.choice(N_eqns, hfm.batch_size)
            
            (t_data_batch,
             x_data_batch,
             y_data_batch,
             c_data_batch) = (hfm.t_data[idx_data,:],
                              hfm.x_data[idx_data,:],
                              hfm.y_data[idx_data,:],
                              hfm.c_data[idx_data,:])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch) = (hfm.t_eqns[idx_eqns,:],
                              hfm.x_eqns[idx_eqns,:],
                              hfm.y_eqns[idx_eqns,:])


            tf_dict = {hfm.t_data_tf: t_data_batch,
                       hfm.x_data_tf: x_data_batch,
                       hfm.y_data_tf: y_data_batch,
                       hfm.c_data_tf: c_data_batch,
                       hfm.t_eqns_tf: t_eqns_batch,
                       hfm.x_eqns_tf: x_eqns_batch,
                       hfm.y_eqns_tf: y_eqns_batch,
                       hfm.learning_rate: learning_rate}
            
            hfm.sess.run([hfm.train_op], tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = hfm.sess.run([hfm.loss,
                                                       hfm.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learnsing Rate: %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1
    
    def predict(hfm, t_star, x_star, y_star):
        
        tf_dict = {hfm.t_data_tf: t_star, hfm.x_data_tf: x_star, hfm.y_data_tf: y_star}
        
        c_star = hfm.sess.run(hfm.c_data_pred, tf_dict)
        u_star = hfm.sess.run(hfm.u_data_pred, tf_dict)
        v_star = hfm.sess.run(hfm.v_data_pred, tf_dict)
        p_star = hfm.sess.run(hfm.p_data_pred, tf_dict)
        
        return c_star, u_star, v_star, p_star

def main():
    try:
        os.chdir("DATA")    # FOR TESTING ONLY
    except:
        raise IOError("Invalid directory")

    mesh_template = meshio.read("misc/mesh_template.vtu")
    results_mat = scipy.io.loadmat("training_120k_results_26_07_2021.mat")
    #mat2vtu(mesh_template, results_mat)
    # fNames = parse_args()["fileNames"]
    
    # if fNames:
    #     try:
    #         generate_frame_arrays(fNames, "linear")

    #     except:
    #         raise RuntimeError("There was a problem rendering the results")

if __name__ == "__main__":
    main()