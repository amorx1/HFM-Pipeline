"""
@author: Akshay Mor, Maziar Raissi
"""

import logging
import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys
import os
from Pipeline import *
from utilities import neural_net, Navier_Stokes_2D, \
                      tf_session, mean_squared_error, relative_error


def main():

    # change working directory to locate .vtu files
    os.chdir("DATA/input_data")

    # initialise the pipeline
    pipeline = Pipeline()

    # pipeline settings
    pipeline.useGPUAcceleration = True  # Note: GPU acceleration requires a CUDA-enabled NVidia GPU
    pipeline.outputMAT = True

    # model settings
    pipeline.batch_size = 10000
    pipeline.layers = [3] + 10*[4*50] + [4]
    pipeline.Pec = 1000000
    pipeline.Rey = 450
    
    # prepare all data
    pipeline.getFiles()
    pipeline.extractData()
    pipeline.Inputs.setTN()
    pipeline.TrainingData, pipeline.Equations, pipeline.TestData = pipeline.Inputs.splitTrainTestEqns()

    # create model inside Pipeline instance
    pipeline.model()

    # train model
    pipeline.HFM.train(total_time = 0.01, learning_rate = 1e-3)

    # make predictions and calculate errors
    pipeline.Predictions, pipeline.Errors = pipeline.Predict()

    # os.chdir("/Users/akshay/Documents/GitHub/HFM-Pipeline/DATA/input_data")
    # pipeline.createMeshTemplate()
    # pipeline.writeVTU()

    print("Done")
    
    # # Training
    # model = HFM(t_data, x_data, y_data, c_data,
    #             t_eqns, x_eqns, y_eqns,
    #             layers, batch_size,
    #             Pec = 100, Rey = 100)
        
    # model.train(total_time = 0.05, learning_rate=1e-3)
    
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
