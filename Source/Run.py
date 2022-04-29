"""
@author: Akshay Mor, Maziar Raissi
"""

import tensorflow as tf
import os
from Pipeline import *
from utilities import neural_net, Navier_Stokes_2D, \
                      tf_session, mean_squared_error, relative_error


def main():

    # configure model options & parameters
    options = {
        "training": {
            "mode": "curr",
            "augment_loss_at": 5,
            "sequence": [50, 100, 150, 200, 250]
            },
        "checkpoint_every": 50000,
        "stop_early": 10,
        "BCs": {
            "no-slip": True,
            "dirichlet": True
                },
        "lambdas": [1.0, 1.0, 1.0],
        "batch_size": 10000,
        "layers": [3] + 10*[4*50] + [4],
        "Pec": 1500000,
        "Rey": 450,
        "output_mat": True,
        "boundary_file_path": "/Users/akshay/Downloads/patchID.mat"
    }

    # initialise the pipeline
    pipeline = Pipeline(options=options)
    
    # change working directory to locate .vtu files
    os.chdir("/Users/akshay/Documents/GitHub/HFM-Pipeline/DATA/4-inlets-5of5")

    # prepare all data
    pipeline.getFiles()
    pipeline.extractData()
    pipeline.Inputs.setTN()
    pipeline.TrainingData, pipeline.Equations, pipeline.TestData = pipeline.Inputs.splitTrainTestEqns()

    # # create model inside Pipeline instance
    pipeline.model()

    # # train model
    pipeline.Train(total_time = 0.01, learning_rate = 1e-3)

    # # make predictions and calculate errors
    pipeline.Predictions, pipeline.Errors = pipeline.Predict()

    os.chdir("/Users/akshay/Documents/GitHub/HFM-Pipeline/DATA/4-inlets-5of5")

    # create mesh and write data
    pipeline.createMeshTemplate()
    pipeline.writeVTU()

    print("Done")
    
if __name__ == "__main__":
    main()
