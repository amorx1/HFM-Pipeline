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
from scipy.io import savemat
import matplotlib.pyplot as plt
from matplotlib.tri import CubicTriInterpolator
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd
import os
import sys
import time
import glob
from Objects import *
from utilities import numericalSort, relative_error, neural_net, mean_squared_error, Navier_Stokes_2D, tf_session, Gradient_Velocity_2D

class Pipeline:

    def __init__(self, options: dict = None):

        # Attributes
        self.fileNames = None
        self.mesh = None
        self.mesh_temp = None

        # Options
        self.data = None
        self.training_mode = options["training"]["mode"]
        self.training_augment_loss_at = options["training"]["augment_loss_at"]
        self.training_sequence = options["training"]["sequence"]
        self.checkpoint_every = options["checkpoint_every"]
        self.stop_early = options["stop_early"]
        self.BCs_no_slip = options["BCs"]["no-slip"]
        self.BCs_dirichlet = options["BCs"]["dirichlet"]
        self.lambdas = options["lambdas"]
        self.batch_size = options["batch_size"]
        self.layers = options["layers"]
        self.Pec = options["Pec"]
        self.Rey = options["Rey"]
        self.output_mat = options["output_mat"]
        self.patch_ID = scipy.io.loadmat(options["boundary_file_path"])["patchID"]

        if type(self.training_mode) is not str or \
            (self.training_augment_loss_at is not None and type(self.training_augment_loss_at) is not int) or \
                (self.training_sequence is not None and type(self.training_sequence) is not list) or \
                    type(self.checkpoint_every) is not int or \
                        type(self.stop_early) is not int or \
                            type(self.BCs_no_slip) is not bool or \
                                type(self.BCs_dirichlet) is not bool or \
                                    type(self.lambdas) is not list or \
                                        type(self.batch_size) is not int or \
                                            type(self.Pec) is not int or \
                                                type(self.Rey) is not int or \
                                                    type(self.output_mat) is not bool: raise TypeError("Type error in Pipeline options dictionary")

        if self.training_sequence is None:
            self.training_sequence = [250]

        if self.training_mode == "std":
            #  set slices to [250]
            self.training_sequence = [250]

        if self.training_mode == "curr":
            if self.training_augment_loss_at == None:
                self.training_augment_loss_at = 150000
            self.training_sequence = [250]

        # Objects
        self.Inputs = InputData(no_slip=self.BCs_no_slip, dirichlet=self.BCs_dirichlet, patch_ID=self.patch_ID)
        self.TrainingData = TrainingData()
        self.TestData = TestData()
        self.Equations = Equations()
        self.Predictions = Predictions()
        self.HFM = None

    def getFiles(self):
        # sort and get list of .vtu files in correct order
        path = os.getcwd()
        if(self.fileNames == None):
            # parse file names and registration name
            self.fileNames = sorted(glob.glob(os.path.join(path, "*.vtu")), key = numericalSort)
        else: raise IOError("A set of files already exists for this pipeline!") 
    def setTN(self):
        if (self.T == None and self.N == None):
            self.T = self.Inputs.t_star.shape[0]
            self.N = self.Inputs.x_star.shape[0]
        else: raise IOError("T and N already exist for this pipeline!")
    
    # Instantiate HFM object with existing input data
    def model(self):
        if self.HFM == None:
            self.HFM = HFM(data=self.data, patch_ID=self.patch_ID, layers=self.layers, batch_size=self.batch_size, mode=self.training_mode, slice_array=self.training_sequence, augment_loss_at=self.training_augment_loss_at, stop_early=self.stop_early, no_slip=self.BCs_no_slip, dirichlet=self.BCs_dirichlet, Pec=self.Pec, Rey=self.Rey)
        else:
            raise IOError("An HFM object already exists within the Pipeline!")

    # Read and write input data to Inputs
    def extractData(self, fileNames=None):
        
        # create empty dictionaries for data
        x_star = pd.DataFrame().astype(np.float)
        y_star = pd.DataFrame().astype(np.float)
        t_star = pd.DataFrame().astype(np.float)
        C_star = pd.DataFrame().astype(np.float)
        U_star = pd.DataFrame().astype(np.float)
        V_star = pd.DataFrame().astype(np.float)
        P_star = pd.DataFrame().astype(np.float)

        if (fileNames == None and self.fileNames == None): raise IOError("No files have been provided to the pipeline for reading")
        else:

            # if fileNames is not passed as argument, use existing filenames of object
            if (fileNames == None):
                if len(self.fileNames) != 0:
                    mesh = meshio.read(self.fileNames[1])
                    x_star.loc[:,1] = np.array(mesh.points[:,0])
                    y_star.loc[:,1] = np.array(mesh.points[:,1])
                    t_star.loc[:,1] = (np.linspace(0, 25, num=251))

                    it = 0
                    for file in self.fileNames:

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

        savemat("4-inlets-mat.mat", {
            "C_star": C_star.to_numpy(),
            "U_star": U_star.to_numpy(),
            "V_star": V_star.to_numpy(),
            "P_star": P_star.to_numpy(),
            "x_star": x_star.to_numpy(),
            "y_star": y_star.to_numpy(),
            "t_star": t_star.to_numpy() 
        })

        self.data = {
            "C_star": C_star.to_numpy(),
            "U_star": U_star.to_numpy(),
            "V_star": V_star.to_numpy(),
            "P_star": P_star.to_numpy(),
            "x_star": x_star.to_numpy(),
            "y_star": y_star.to_numpy(),
            "t_star": t_star.to_numpy() 
        }

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

    # returns x and y coordinates (constant for all timesteps)
    # def get_coords(self, mesh):
    #     if mesh != None:
    #         x = mesh.points[:,0]
    #         y = mesh.points[:,1]
    #         return x, y
    #     else:
    #         raise IOError("No mesh provided!")
    def Train(self, total_time, learning_rate):
        self.HFM.train(mode=self.training_mode, total_time=total_time, learning_rate=learning_rate, end=self.training_sequence[0])
        # check if curr
            #  check for aug point
            #  aug
        # else  std training
        return

    def Predict(self):

        predictions = Predictions()
        errors = Errors()
        predictions.c_pred = 0*self.Inputs.C_star
        predictions.u_pred = 0*self.Inputs.U_star
        predictions.v_pred = 0*self.Inputs.V_star
        predictions.p_pred = 0*self.Inputs.P_star

        T_star = np.tile(self.Inputs.t_star, (1,self.Inputs.N)).T
        X_star = np.tile(self.Inputs.x_star, (1,self.Inputs.N))
        Y_star = np.tile(self.Inputs.y_star, (1,self.Inputs.N))

        for snap in range(0, self.Inputs.T):
            t_test = T_star[:,snap:snap+1]
            x_test = X_star[:,snap:snap+1]
            y_test = Y_star[:,snap:snap+1]
            
            c_test = (self.Inputs.C_star.to_numpy())[:,snap:snap+1]
            u_test = (self.Inputs.U_star.to_numpy())[:,snap:snap+1]
            v_test = (self.Inputs.V_star.to_numpy())[:,snap:snap+1]
            p_test = (self.Inputs.P_star.to_numpy())[:,snap:snap+1]

            c_pred, u_pred, v_pred, p_pred = self.HFM.predict(t_test, x_test, y_test)

            errors.error_c.loc[snap, "Error"] = relative_error(c_pred, c_test)
            errors.error_u.loc[snap, "Error"] = relative_error(u_pred, u_test)
            errors.error_v.loc[snap, "Error"] = relative_error(v_pred, v_test)
            errors.error_p.loc[snap, "Error"] = relative_error(p_pred, p_test)
            
            (predictions.c_pred.to_numpy())[:,snap:snap+1] = c_pred
            (predictions.u_pred.to_numpy())[:,snap:snap+1] = u_pred
            (predictions.v_pred.to_numpy())[:,snap:snap+1] = v_pred
            (predictions.p_pred.to_numpy())[:,snap:snap+1] = p_pred
        
        if(self.output_mat):
            os.chdir("/Users/akshay/Downloads/TESTINGWRITE")
            scipy.io.savemat('4_inlet_results_%s.mat' %(time.strftime('%d_%m_%Y')), {'C_pred':predictions.c_pred.to_numpy(), 'U_pred':predictions.u_pred.to_numpy(), 'V_pred':predictions.v_pred.to_numpy(), 'P_pred':predictions.p_pred.to_numpy(), 'Losses':self.HFM.track_loss})
            scipy.io.savemat('4_inlet_errors_%s.mat' %(time.strftime('%d_%m_%Y')), {'C_error': errors.error_c.to_numpy(), 'U_error': errors.error_u.to_numpy(), 'V_error': errors.error_v.to_numpy(), 'P_error': errors.error_p.to_numpy()})

        return predictions, errors

    def writeVTU(self, mesh_template=None, predictions: dict=None):
        os.chdir("/Users/akshay/Downloads/TESTINGWRITE")
        if (mesh_template == None and predictions == None):
            if (self.Predictions != None):
                if (self.mesh_temp == None): raise IOError("No mesh template to use! Try creating one first!")
                points = self.mesh_temp.points
                for i in range(251):
                    output = pd.DataFrame(points)
                    output.loc[:, "Con"] = self.Predictions.c_pred.iloc[:,i]
                    output.loc[:, "Pres"] = self.Predictions.p_pred.iloc[:,i]
                    # output.loc[:, "Vel"] = np.linalg.norm((np.concatenate(self.predictions["U_pred"], self.predictions["V_pred"], axis=1)), axis=1)
                    cells = self.mesh_temp.cells
                    point_data = {
                        "Con" : output.loc[:, "Con"],
                        "Pres": output.loc[:, "Pres"],
                        #"Vel" : output.loc[:, "Vel"]
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

class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _inlet: input-output data at the inlet
    # _cyl: points used to regress the no slip boundary condition
    # _star: predictions

    def __init__(self, data, patch_ID, layers, batch_size, mode, slice_array, augment_loss_at, stop_early, no_slip, dirichlet, Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey

        # data
        self.data = data
        self.patch_ID = patch_ID
        self.slice_count = 0
        self.slice_array = slice_array
        self.augment_loss_at = augment_loss_at
        self.stop_early = stop_early
        self.total_its = 0

        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.c_data_tf, self.p_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1])
                                                                            for _ in range(5)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in
                                                            range(3)]
        [self.t_inlet_tf, self.x_inlet_tf, self.y_inlet_tf, self.u_inlet_tf, self.v_inlet_tf] = [
            tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]

        [self.t_cyl_tf, self.x_cyl_tf, self.y_cyl_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        self.confirm_start = self.extract_data(0,self.slice_array[self.slice_count])
        self.chkpt_num = 0

        # physics "uninformed" neural networks
        self.net_cuvp = neural_net(self.t_data, self.x_data, self.y_data, layers=self.layers)

        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_cuvp(self.t_data_tf,
                                           self.x_data_tf,
                                           self.y_data_tf)

        # physics "uninformed" neural networks (data at the inlet)
        [_,
         self.u_inlet_pred,
         self.v_inlet_pred,
         _] = self.net_cuvp(self.t_inlet_tf,
                            self.x_inlet_tf,
                            self.y_inlet_tf)



        # physics "uninformed" neural networks (data on the cylinder)
        [_,
         self.u_cyl_pred,
         self.v_cyl_pred,
         _] = self.net_cuvp(self.t_cyl_tf,
                            self.x_cyl_tf,
                            self.y_cyl_tf)

        # physics "informed" neural networks
        [self.c_eqns_pred,
         self.u_eqns_pred,
         self.v_eqns_pred,
         self.p_eqns_pred] = self.net_cuvp(self.t_eqns_tf,
                                           self.x_eqns_tf,
                                           self.y_eqns_tf)


        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred,
         self.e5_eqns_pred] = Navier_Stokes_2D(self.c_eqns_pred,
                                               self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.p_eqns_pred,
                                               self.t_eqns_tf,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.Pec,
                                               self.Rey)


        # gradients required for the lift and drag forces
        [self.u_x_eqns_pred,
         self.v_x_eqns_pred,
         self.u_y_eqns_pred,
         self.v_y_eqns_pred] = Gradient_Velocity_2D(self.u_eqns_pred,
                                                    self.v_eqns_pred,
                                                    self.x_eqns_tf,
                                                    self.y_eqns_tf)

        if (mode == "std" or mode == "s2s") and not dirichlet and not no_slip:
            self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0) + \
                        mean_squared_error(self.e3_eqns_pred, 0.0) + \
                        mean_squared_error(self.e4_eqns_pred, 0.0) + \
                        mean_squared_error(self.e5_eqns_pred, 0.0)

        if (mode == "std" or mode == "s2s") and dirichlet and not no_slip:
            self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                        mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0) + \
                        mean_squared_error(self.e3_eqns_pred, 0.0) + \
                        mean_squared_error(self.e4_eqns_pred, 0.0) + \
                        mean_squared_error(self.e5_eqns_pred, 0.0)


        if (mode == "std" or mode == "s2s") and no_slip and not dirichlet:
            self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_cyl_pred, 0.0) + \
                        mean_squared_error(self.v_cyl_pred, 0.0) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0) + \
                        mean_squared_error(self.e3_eqns_pred, 0.0) + \
                        mean_squared_error(self.e4_eqns_pred, 0.0) + \
                        mean_squared_error(self.e5_eqns_pred, 0.0)
        

        if (mode == "std" or mode == "s2s") and no_slip and dirichlet:
            self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                        mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                        mean_squared_error(self.u_cyl_pred, 0.0) + \
                        mean_squared_error(self.v_cyl_pred, 0.0) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0) + \
                        mean_squared_error(self.e3_eqns_pred, 0.0) + \
                        mean_squared_error(self.e4_eqns_pred, 0.0) + \
                        mean_squared_error(self.e5_eqns_pred, 0.0)


        if mode == "curr" and not dirichlet and not no_slip:
            self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0)

            self.loss2 = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0) + \
                        mean_squared_error(self.e3_eqns_pred, 0.0) + \
                        mean_squared_error(self.e4_eqns_pred, 0.0) + \
                        mean_squared_error(self.e5_eqns_pred, 0.0)

        if mode == "curr" and dirichlet and not no_slip:
            self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                        mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0)

            self.loss2 = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                        mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0) + \
                        mean_squared_error(self.e3_eqns_pred, 0.0) + \
                        mean_squared_error(self.e4_eqns_pred, 0.0) + \
                        mean_squared_error(self.e5_eqns_pred, 0.0)

        if mode == "curr" and no_slip and not dirichlet:
            self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_cyl_pred, 0.0) + \
                        mean_squared_error(self.v_cyl_pred, 0.0) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0)

            self.loss2 = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_cyl_pred, 0.0) + \
                        mean_squared_error(self.v_cyl_pred, 0.0) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0) + \
                        mean_squared_error(self.e3_eqns_pred, 0.0) + \
                        mean_squared_error(self.e4_eqns_pred, 0.0) + \
                        mean_squared_error(self.e5_eqns_pred, 0.0)

        
        if mode == "curr" and no_slip and  dirichlet:
            self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                        mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                        mean_squared_error(self.u_cyl_pred, 0.0) + \
                        mean_squared_error(self.v_cyl_pred, 0.0) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0)

            self.loss2 = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                        mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                        mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                        mean_squared_error(self.u_cyl_pred, 0.0) + \
                        mean_squared_error(self.v_cyl_pred, 0.0) + \
                        mean_squared_error(self.e1_eqns_pred, 0.0) + \
                        mean_squared_error(self.e2_eqns_pred, 0.0) + \
                        mean_squared_error(self.e3_eqns_pred, 0.0) + \
                        mean_squared_error(self.e4_eqns_pred, 0.0) + \
                        mean_squared_error(self.e5_eqns_pred, 0.0)


        self.track_loss = []

        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
        self.saver = tf.train.Saver(max_to_keep=100)
    
    def extract_data(self, start, end):
        
        try:
          self.patch_ID = self.patch_ID[:, start:end].flatten()[:,None]

          t_star = self.data['t_star'][start:end,:]#[0:58,0:] # T x 1
          x_star = self.data['x_star']#[0:,0:58] # N x 1
          y_star = self.data['y_star']#[0:,0:58] # N x 1

          T = t_star.shape[0]
          N = x_star.shape[0]
          
          U_star = self.data['U_star'][:,start:end]#[0:,0:58] # N x T
          V_star = self.data['V_star'][:,start:end]#[0:,0:58] # N x T
          P_star = self.data['P_star'][:,start:end]#[0:,0:58] # N x T
          C_star = self.data['C_star'][:,start:end]#[0:,0:58] # N x T

          # Rearrange Data 
          T_star = np.tile(t_star, (1,N)).T # N x T
          X_star = np.tile(x_star, (1,T)) # N x T
          Y_star = np.tile(y_star, (1,T)) # N x T
          
          t = T_star.flatten()[:,None] # NT x 1
          x = X_star.flatten()[:,None] # NT x 1
          y = Y_star.flatten()[:,None] # NT x 1
          u = U_star.flatten()[:,None] # NT x 1
          v = V_star.flatten()[:,None] # NT x 1
          p = P_star.flatten()[:,None] # NT x 1
          c = C_star.flatten()[:,None] # NT x 1
          
          ######################################################################
          ######################## Training Data ###############################
          ######################################################################
          
          T_data = T # int(sys.argv[1])
          N_data = N # int(sys.argv[2])
          idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
          idx_x = np.random.choice(N, N_data, replace=False)
          t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
          x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
          y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
          c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
          p_data = P_star[:, idx_t][idx_x,:].flatten()[:,None]

          T_eqns = T
          N_eqns = N
          idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
          idx_x = np.random.choice(N, N_eqns, replace=False)
          t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
          x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
          y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
          
          # Training Data on velocity (inlet)
          t_inlet = t[np.where((self.patch_ID == 2) | (self.patch_ID == 3) | (self.patch_ID == 4) | (self.patch_ID == 5))][:,None]
          x_inlet = x[np.where((self.patch_ID == 2) | (self.patch_ID == 3) | (self.patch_ID == 4) | (self.patch_ID == 5))][:,None]
          y_inlet = y[np.where((self.patch_ID == 2) | (self.patch_ID == 3) | (self.patch_ID == 4) | (self.patch_ID == 5))][:,None]
          u_inlet = u[np.where((self.patch_ID == 2) | (self.patch_ID == 3) | (self.patch_ID == 4) | (self.patch_ID == 5))][:,None]
          v_inlet = v[np.where((self.patch_ID == 2) | (self.patch_ID == 3) | (self.patch_ID == 4) | (self.patch_ID == 5))][:,None]

          # Training Data on pressure (outlet)
          t_outlet = t[(np.where(self.patch_ID == 6))][:, None]
          x_outlet = x[(np.where(self.patch_ID == 6))][:, None]
          y_outlet = y[(np.where(self.patch_ID == 6))][:, None]
          p_outlet = p[(np.where(self.patch_ID == 6))][:, None]

          # Training Data on velocity (cylinder)
          t_cyl = t[np.where((self.patch_ID==1) | (self.patch_ID==7))][:,None]
          x_cyl = x[np.where((self.patch_ID==1) | (self.patch_ID==7))][:,None]
          y_cyl = y[np.where((self.patch_ID==1) | (self.patch_ID==7))][:,None]

          [self.t_data, self.x_data, self.y_data, self.c_data, self.p_data] = [t_data, x_data, y_data, c_data, p_data]
          [self.t_eqns, self.x_eqns, self.y_eqns] = [t_eqns, x_eqns, y_eqns]
          [self.t_inlet, self.x_inlet, self.y_inlet, self.u_inlet, self.v_inlet] = [t_inlet, x_inlet, y_inlet, u_inlet,
                                                                                    v_inlet]

          [self.t_cyl, self.x_cyl, self.y_cyl] = [t_cyl, x_cyl, y_cyl]

        except:
          return False

        return True

    def train(self, mode, total_time, learning_rate, end):
        
        if mode == "std" or mode == "curr":
            if self.confirm_start:

                N_data = self.t_data.shape[0]
                N_eqns = self.t_eqns.shape[0]
                
                start_time = time.time()
                running_time = 0
                it = 0
                
                while running_time < total_time:
                    
                    idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
                    idx_eqns = np.random.choice(N_eqns, self.batch_size)

                    (t_data_batch,
                    x_data_batch,
                    y_data_batch,
                    c_data_batch,
                    p_data_batch) = (self.t_data[idx_data, :],
                                        self.x_data[idx_data, :],
                                        self.y_data[idx_data, :],
                                        self.c_data[idx_data, :],
                                        self.p_data[idx_data, :])

                    (t_eqns_batch,
                    x_eqns_batch,
                    y_eqns_batch) = (self.t_eqns[idx_eqns, :],
                                        self.x_eqns[idx_eqns, :],
                                        self.y_eqns[idx_eqns, :])

                    tf_dict = {self.t_data_tf: t_data_batch,
                                self.x_data_tf: x_data_batch,
                                self.y_data_tf: y_data_batch,
                                self.c_data_tf: c_data_batch,
                                self.t_eqns_tf: t_eqns_batch,
                                self.x_eqns_tf: x_eqns_batch,
                                self.y_eqns_tf: y_eqns_batch,
                                self.t_inlet_tf: self.t_inlet,
                                self.x_inlet_tf: self.x_inlet,
                                self.y_inlet_tf: self.y_inlet,
                                self.u_inlet_tf: self.u_inlet,
                                self.v_inlet_tf: self.v_inlet,
                                self.t_cyl_tf: self.t_cyl,
                                self.x_cyl_tf: self.x_cyl,
                                self.y_cyl_tf: self.y_cyl,
                                self.learning_rate: learning_rate}
                    
                    self.sess.run([self.train_op], tf_dict)
                    
                    # Print
                    if it % 10 == 0:
                        elapsed = time.time() - start_time
                        running_time += elapsed/3600.0

                        if (it > self.augment_loss_at and mode == "cur"):
                            [loss_value,
                            learning_rate_value] = self.sess.run([self.loss2,
                                                                self.learning_rate], tf_dict)
                            
                        else:
                            [loss_value,
                            learning_rate_value] = self.sess.run([self.loss,
                                                                self.learning_rate], tf_dict)
                            
                        self.track_loss.append(loss_value)
                        print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                                %(it, loss_value, elapsed, running_time, learning_rate_value))
                        sys.stdout.flush()
                        start_time = time.time()

                    if mode == "curr":
                        if it == self.augment_loss_at:
                            print("AUGMENTING LOSS FUNCTION")
                            self.train_op = self.optimizer.minimize(self.loss2)

                    if it % 50000 == 0 and it != 0:
                        self.saver.save(self.sess, '/content/drive/MyDrive/06-04/Checkpoints/chckpnt'+str(self.chkpt_num))
                        self.chkpt_num = self.chkpt_num+1

                    if it == self.stop_early:
                        self.confirm_start = False
                        break

                    it += 1
                    
                self.saver.save(self.sess, '/Users/akshay/Downloads/TESTINGSAVE')
                self.slice_count = self.slice_count+1
                # 50-100

                try:
                    proceed = self.extract_data(0,self.slice_array[self.slice_count])
                    if proceed:
                        print("TRAINING 0-"+str(self.slice_array[self.slice_count]))
                        self.train(12, 1e-3, 0, self.slice_array[self.slice_count])

                    else:
                        print("COMPLETED " + mode + " TRAINING")
                except:
                    print("COMPLETED " + mode + " TRAINING")
                
            else:
                print("Training failed - error extracting data")

        if mode == "s2s":
            if self.confirm_start:

                N_data = self.t_data.shape[0]
                N_eqns = self.t_eqns.shape[0]
                
                start_time = time.time()
                running_time = 0
                it = 0
                
                while running_time < total_time and self.total_its < self.stop_early:
                    
                    idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
                    idx_eqns = np.random.choice(N_eqns, self.batch_size)

                    (t_data_batch,
                    x_data_batch,
                    y_data_batch,
                    c_data_batch,
                    p_data_batch) = (self.t_data[idx_data, :],
                                        self.x_data[idx_data, :],
                                        self.y_data[idx_data, :],
                                        self.c_data[idx_data, :],
                                        self.p_data[idx_data, :])

                    (t_eqns_batch,
                    x_eqns_batch,
                    y_eqns_batch) = (self.t_eqns[idx_eqns, :],
                                        self.x_eqns[idx_eqns, :],
                                        self.y_eqns[idx_eqns, :])

                    tf_dict = {self.t_data_tf: t_data_batch,
                                self.x_data_tf: x_data_batch,
                                self.y_data_tf: y_data_batch,
                                self.c_data_tf: c_data_batch,
                                self.t_eqns_tf: t_eqns_batch,
                                self.x_eqns_tf: x_eqns_batch,
                                self.y_eqns_tf: y_eqns_batch,
                                self.t_inlet_tf: self.t_inlet,
                                self.x_inlet_tf: self.x_inlet,
                                self.y_inlet_tf: self.y_inlet,
                                self.u_inlet_tf: self.u_inlet,
                                self.v_inlet_tf: self.v_inlet,
                                self.t_cyl_tf: self.t_cyl,
                                self.x_cyl_tf: self.x_cyl,
                                self.y_cyl_tf: self.y_cyl,
                                self.learning_rate: learning_rate}
                    
                    self.sess.run([self.train_op], tf_dict)
                    
                    # Print
                    if it % 10 == 0:
                        elapsed = time.time() - start_time
                        running_time += elapsed/3600.0

                        [loss_value,
                        learning_rate_value] = self.sess.run([self.loss,
                                                            self.learning_rate], tf_dict)
                            
                        self.track_loss.append(loss_value)
                        print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                                %(it, loss_value, elapsed, running_time, learning_rate_value))
                        sys.stdout.flush()
                        start_time = time.time()

            
                    if it % 50000 == 0 and it != 0:
                        self.saver.save(self.sess, '/content/drive/MyDrive/06-04/Checkpoints/chckpnt'+str(self.chkpt_num))
                        self.chkpt_num = self.chkpt_num+1

                    if it == 20:
                        print("NEXT SLICE")
                        break

                    if it == self.stop_early:
                        self.confirm_start = False
                        break

                    it += 1
                    
                self.saver.save(self.sess, '/Users/akshay/Downloads/TESTINGSAVE/complete_model')
                self.slice_count = self.slice_count+1
                self.total_its = self.total_its + it
                # 50-100

                try:
                    proceed = self.extract_data(0,self.slice_array[self.slice_count])
                    if proceed:
                        print("TRAINING 0-"+str(self.slice_array[self.slice_count]))
                        self.train(mode=mode, total_time=12, learning_rate=1e-3, end=self.slice_array[self.slice_count])

                    else:
                        print("COMPLETED SEQ2SEQ TRAINING")
                except:
                    print("COMPLETED SEQ2SEQ TRAINING")
                
            else:
                print("Training failed or early stopping limit hit")

    def predict(self, t_star, x_star, y_star):
        
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star}
        
        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return c_star, u_star, v_star, p_star
    
    def predict_drag_lift(self, t_cyl):
        
        viscosity = (1.0/self.Rey)
        
        theta = np.linspace(0.0,2*np.pi,200)[:,None] # N x 1
        d_theta = theta[1,0] - theta[0,0]
        x_cyl = 0.5*np.cos(theta) # N x 1
        y_cyl = 0.5*np.sin(theta) # N x 1
            
        N = x_cyl.shape[0]
        T = t_cyl.shape[0]
        
        T_star = np.tile(t_cyl, (1,N)).T # N x T
        X_star = np.tile(x_cyl, (1,T)) # N x T
        Y_star = np.tile(y_cyl, (1,T)) # N x T
        
        t_star = np.reshape(T_star,[-1,1]) # NT x 1
        x_star = np.reshape(X_star,[-1,1]) # NT x 1
        y_star = np.reshape(Y_star,[-1,1]) # NT x 1
        
        tf_dict = {self.t_eqns_tf: t_star, self.x_eqns_tf: x_star, self.y_eqns_tf: y_star}
        
        [p_star,
         u_x_star,
         u_y_star,
         v_x_star,
         v_y_star] = self.sess.run([self.p_eqns_pred,
                                    self.u_x_eqns_pred,
                                    self.u_y_eqns_pred,
                                    self.v_x_eqns_pred,
                                    self.v_y_eqns_pred], tf_dict)
        
        P_star = np.reshape(p_star, [N,T]) # N x T
        P_star = P_star - np.mean(P_star, axis=0)
        U_x_star = np.reshape(u_x_star, [N,T]) # N x T
        U_y_star = np.reshape(u_y_star, [N,T]) # N x T
        V_x_star = np.reshape(v_x_star, [N,T]) # N x T
        V_y_star = np.reshape(v_y_star, [N,T]) # N x T
    
        INT0 = (-P_star[0:-1,:] + 2*viscosity*U_x_star[0:-1,:])*X_star[0:-1,:] + viscosity*(U_y_star[0:-1,:] + V_x_star[0:-1,:])*Y_star[0:-1,:]
        INT1 = (-P_star[1: , :] + 2*viscosity*U_x_star[1: , :])*X_star[1: , :] + viscosity*(U_y_star[1: , :] + V_x_star[1: , :])*Y_star[1: , :]
            
        F_D = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
    
        
        INT0 = (-P_star[0:-1,:] + 2*viscosity*V_y_star[0:-1,:])*Y_star[0:-1,:] + viscosity*(U_y_star[0:-1,:] + V_x_star[0:-1,:])*X_star[0:-1,:]
        INT1 = (-P_star[1: , :] + 2*viscosity*V_y_star[1: , :])*Y_star[1: , :] + viscosity*(U_y_star[1: , :] + V_x_star[1: , :])*X_star[1: , :]
            
        F_L = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
            
        return F_D, F_L


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