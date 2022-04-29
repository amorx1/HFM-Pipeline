import meshio
import os
import scipy.io
import pandas as pd
import numpy as np

mesh_template = meshio.read('/Users/akshay/Downloads/2D_LA_chick_low_Re/2D_LA_chick_low_Re-0.vtu')
mesh_template.point_data = None

predictions = scipy.io.loadmat('/Users/akshay/Documents/Project_models/4_inlets/No_BC_450_15e6/300k_its/preds.mat')

def writeVTU(mesh_template=None, predictions: dict=None):
    os.chdir('/Users/akshay/Documents/Project_models/4_inlets/No_BC_450_15e6/300k_its/Prediction')
    points = mesh_template.points
    for i in range(250):
        output = pd.DataFrame(points)
        output.loc[:, "Con"] = predictions["C_pred"][:,i]
        output.loc[:, "Pres"] = predictions["P_pred"][:,i]
        # output.loc[:, "U"] = predictions["U_pred"][:,i]
        # output.loc[:, "V"] = predictions["V_pred"][:,i]
        # output.loc[:, "Vel"] = np.linalg.norm((np.concatenate(predictions["U_pred"], predictions["V_pred"], axis=1)), axis=1)
        cells = mesh_template.cells
        point_data = {
            "Con" : output.loc[:, "Con"],
            "Pres": output.loc[:, "Pres"],
            # "U": output.loc[:, "U"],
            # "V": output.loc[:, "V"]
            # "Vel" : output.loc[:, "Vel"]
        }
        mesh = meshio.Mesh(
            points,
            cells,
            point_data
        )
        mesh.write("_"+str(i)+".vtu")

writeVTU(mesh_template, predictions)