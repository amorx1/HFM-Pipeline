"""
@author: Akshay Mor, Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys
import os
from Pipeline import *
from utilities import neural_net, Navier_Stokes_2D, \
                      tf_session, mean_squared_error, relative_error, getFiles, extract_data


def main():

    # change working directory to locate files to be read
    try:
        os.chdir("DATA/input_data")    # FOR TESTING ONLY
    except:
        print("Invalid directory")

    # initialise the pipeline
    pipeline = Pipeline()
    pipeline.extractData(getFiles()["fileNames"])
    pipeline.Inputs.setTN()

    pipeline.TrainingData, pipeline.Equations, pipeline.TestData = pipeline.Inputs.splitTrainTestEqns()
    print("Done")
    
    # batch_size = 10000
    
    # layers = [3] + 10*[4*50] + [4]
    
    # # Load Data
    # # data = scipy.io.loadmat('DATA/data_final.mat')
    
    # t_star = pipeline.input_data["t_star"] # T x 1
    # x_star = pipeline.input_data["x_star"] # N x 1
    # y_star = pipeline.input_data["y_star"] # N x 1
    
    # T = t_star.shape[0]
    # N = x_star.shape[0]
        
    # U_star = pipeline.input_data["U_star"] # N x T
    # V_star = pipeline.input_data["V_star"] # N x T
    # P_star = pipeline.input_data["P_star"] # N x T
    # C_star = pipeline.input_data["C_star"] # N x T
    
    # # Rearrange Data 
    # T_star = np.tile(t_star, (1,N)).T # N x T
    # X_star = np.tile(x_star, (1,T)) # N x T
    # Y_star = np.tile(y_star, (1,T)) # N x T
    
    # ######################################################################
    # ######################## Training Data ###############################
    # ######################################################################
    
    # T_data = T 
    # N_data = N 
    # idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    # idx_x = np.random.choice(N, N_data, replace=False)
    # t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    # x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    # y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    # c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
        
    # T_eqns = T
    # N_eqns = N
    # idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    # idx_x = np.random.choice(N, N_eqns, replace=False)
    # t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    # x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    # y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    # # Training
    # model = HFM(t_data, x_data, y_data, c_data,
    #             t_eqns, x_eqns, y_eqns,
    #             layers, batch_size,
    #             Pec = 100, Rey = 100)
        
    # model.train(total_time = 0.05, learning_rate=1e-3)
    
    # # Test Data
    # snap = np.array([100])
    # t_test = T_star[:,snap]
    # x_test = X_star[:,snap]
    # y_test = Y_star[:,snap]    
    
    # c_test = C_star[:,snap]
    # u_test = U_star[:,snap]
    # v_test = V_star[:,snap]
    # p_test = P_star[:,snap]
    
    # # Prediction
    # c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
    
    # # Error
    # error_c = relative_error(c_pred, c_test)
    # error_u = relative_error(u_pred, u_test)
    # error_v = relative_error(v_pred, v_test)
    # error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

    # print('Error c: %e' % (error_c))
    # print('Error u: %e' % (error_u))
    # print('Error v: %e' % (error_v))
    # print('Error p: %e' % (error_p))
    
    # ################# Save Data ###########################
    
    # C_pred = 0*C_star
    # U_pred = 0*U_star
    # V_pred = 0*V_star
    # P_pred = 0*P_star
    # for snap in range(0,t_star.shape[0]):
    #     t_test = T_star[:,snap:snap+1]
    #     x_test = X_star[:,snap:snap+1]
    #     y_test = Y_star[:,snap:snap+1]
        
    #     c_test = C_star[:,snap:snap+1]
    #     u_test = U_star[:,snap:snap+1]
    #     v_test = V_star[:,snap:snap+1]
    #     p_test = P_star[:,snap:snap+1]
    
    #     # Prediction
    #     c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
        
    #     C_pred[:,snap:snap+1] = c_pred
    #     U_pred[:,snap:snap+1] = u_pred
    #     V_pred[:,snap:snap+1] = v_pred
    #     P_pred[:,snap:snap+1] = p_pred
    
    #     # Error
    #     error_c = relative_error(c_pred, c_test)
    #     error_u = relative_error(u_pred, u_test)
    #     error_v = relative_error(v_pred, v_test)
    #     error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))
    
    #     print('Error c: %e' % (error_c))
    #     print('Error u: %e' % (error_u))
    #     print('Error v: %e' % (error_v))
    #     print('Error p: %e' % (error_p))
    
    # scipy.io.savemat('../Results/Cylinder2D_flower_results_%s.mat' %(time.strftime('%d_%m_%Y')),
    #                 {'C_pred':C_pred, 'U_pred':U_pred, 'V_pred':V_pred, 'P_pred':P_pred})
 
if __name__ == "__main__":
    main()
  