"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import meshio
import os
import glob
import re

numbers = re.compile(r'(\d+)')

def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)
    
    # init
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return sess

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

class neural_net(object):
    def __init__(self, *inputs, layers):
        
        self.layers = layers
        self.num_layers = len(self.layers)
        
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)
        
        self.weights = []
        self.biases = []
        self.gammas = []
        
        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))
            
    def __call__(self, *inputs):
                
        H = (tf.concat(inputs, 1) - self.X_mean)/self.X_std
    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.sigmoid(H)
                
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y

def Navier_Stokes_2D(c, u, v, p, t, x, y, Pec, Rey):
    
    Y = tf.concat([c, u, v, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    p = Y[:,3:4]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    p_x = Y_x[:,3:4]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    p_y = Y_y[:,3:4]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    
    e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
    e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy) 
    e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    e4 = u_x + v_y
    
    return e1, e2, e3, e4

def Gradient_Velocity_2D(u, v, x, y):
    
    Y = tf.concat([u, v], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    
    return [u_x, v_x, u_y, v_y]

def Strain_Rate_2D(u, v, x, y):
    
    [u_x, v_x, u_y, v_y] = Gradient_Velocity_2D(u, v, x, y)
    
    eps11dot = u_x
    eps12dot = 0.5*(v_x + u_y)
    eps22dot = v_y
    
    return [eps11dot, eps12dot, eps22dot]

def Navier_Stokes_3D(c, u, v, w, p, t, x, y, z, Pec, Rey):
    
    Y = tf.concat([c, u, v, w, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    w = Y[:,3:4]
    p = Y[:,4:5]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    w_t = Y_t[:,3:4]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    w_x = Y_x[:,3:4]
    p_x = Y_x[:,4:5]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    w_y = Y_y[:,3:4]
    p_y = Y_y[:,4:5]
       
    c_z = Y_z[:,0:1]
    u_z = Y_z[:,1:2]
    v_z = Y_z[:,2:3]
    w_z = Y_z[:,3:4]
    p_z = Y_z[:,4:5]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    w_xx = Y_xx[:,3:4]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    w_yy = Y_yy[:,3:4]
       
    c_zz = Y_zz[:,0:1]
    u_zz = Y_zz[:,1:2]
    v_zz = Y_zz[:,2:3]
    w_zz = Y_zz[:,3:4]
    
    e1 = c_t + (u*c_x + v*c_y + w*c_z) - (1.0/Pec)*(c_xx + c_yy + c_zz)
    e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z
    
    return e1, e2, e3, e4, e5

def Gradient_Velocity_3D(u, v, w, x, y, z):
    
    Y = tf.concat([u, v, w], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    
    return [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z]

def Shear_Stress_3D(u, v, w, x, y, z, nx, ny, nz, Rey):
        
    [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z] = Gradient_Velocity_3D(u, v, w, x, y, z)

    uu = u_x + u_x
    uv = u_y + v_x
    uw = u_z + w_x
    vv = v_y + v_y
    vw = v_z + w_y
    ww = w_z + w_z
    
    sx = (uu*nx + uv*ny + uw*nz)/Rey
    sy = (uv*nx + vv*ny + vw*nz)/Rey
    sz = (uw*nx + vw*ny + ww*nz)/Rey
    
    return sx, sy, sz

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def parse_args() -> dict:

    # sort and get list of .vtu files in correct order
    path = os.getcwd()

    # parse file names and registration name
    fileNames = sorted(glob.glob(os.path.join(path, "*.vtu")), key = numericalSort)
    # registrationName = fileNames[0].replace('0.vtu','*')
    
    # create args
    args = {
        "fileNames": fileNames
    #    "registrationName": registrationName
    }

    return args


def extract_data(args:dict):

    fileNames = args["fileNames"]

    # create empty dictionary for spatio-temporal data

    x_star = pd.DataFrame().astype(np.float)
    y_star = pd.DataFrame().astype(np.float)
    t_star = pd.DataFrame().astype(np.float)

    # create empty dictionaries for C, U, V and P
    C_star = pd.DataFrame().astype(np.float)
    U_star = pd.DataFrame().astype(np.float)
    V_star = pd.DataFrame().astype(np.float)
    P_star = pd.DataFrame().astype(np.float)

    # get x,y,t data and input into output dictionary
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
        print("There are no files")
        pass
    
    # create dictionary containing all data
    data = {
        "x_star": x_star,
        "y_star": y_star,
        "t_star": t_star,
        "C_star": C_star,
        "U_star": U_star,
        "V_star": V_star,
        "P_star": P_star
    }

    return data
