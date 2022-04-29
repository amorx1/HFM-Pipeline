import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from utilities import relative_error


def plot_errors(err_dict, save_location, its):

    colors = ['r', 'g', 'b', 'c']
    c = 0

    t_star = err_dict["Std. N.S+Dir C=1.0"]["t_star"]

    for key2 in err_dict["Std. N.S+Dir C=1.0"]:

        if key2 is "t_star":
            continue

        fig, ax = plt.subplots()
        for key1 in err_dict:
            ax.plot(t_star[10:230,0], err_dict[key1][key2][10:230], colors[c], label=key1)
            # if err_dict[key1] is not None:
            #     ax.plot(t_star[10:230,0], err_dict[key1][key2][10:230], 'r', label=key1)
            c = c+1
        plt.title(key2 + " " + its)
        plt.ylabel("Average Relative L2 Error")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.savefig(save_location + key2 + ".png")
        c = 0
        plt.show()



def compute_errors(ref_file, pred_file):

    preds = scipy.io.loadmat(pred_file)
    ref = scipy.io.loadmat(ref_file)

    C_pred = preds["C_pred"]
    U_pred = preds["U_pred"]
    V_pred = preds["V_pred"]
    P_pred = preds["P_pred"]

    C_star = ref["C_star"]
    U_star = ref["U_star"]
    V_star = ref["V_star"]
    P_star = ref["P_star"]
    t_star = ref["t_star"]

    errors_c = np.zeros(len(t_star))
    errors_u = np.zeros(len(t_star))
    errors_v = np.zeros(len(t_star))
    errors_p = np.zeros(len(t_star))

    for i in range(len(t_star)):
        errors_c[i] =relative_error(C_pred[:,i], C_star[:,i]) 
        errors_u[i] = relative_error(U_pred[:,i], U_star[:,i])
        errors_v[i] = relative_error(V_pred[:,i], V_star[:,i])
        errors_p[i] = relative_error((P_pred[:,i] - np.mean(P_pred[:,i])), (P_star[:,i]-np.mean(P_star[:,i])))
    
    return {"C Error": errors_c, "U Error":errors_u, "V Error":errors_v, "P Error":errors_p, "t_star":t_star}

def main():

    save_location = "/Users/akshay/Downloads/"
    ref_file = "/Users/akshay/Downloads/4-inlets-5of5.mat"
    pred_file1 = "/Users/akshay/Documents/Project_models/4_inlets/Dir_no_slip_450_15e6_CORRECTED/300k_its/preds.mat"
    pred_file2 = "/Users/akshay/Downloads/preds(1).mat"
    # pred_file3 = "/Users/akshay/Documents/Project_models/4_inlets/Weighted-175/300k_its/preds.mat"
    # pred_file4 = "/Users/akshay/Documents/Project_models/4_inlets/Weighted-050/300k_its/preds.mat"

    err_dict = {
        "Std. N.S+Dir C=1.0" : compute_errors(ref_file, pred_file1),
        "Curr. N.S+Dir" : compute_errors(ref_file, pred_file2),
        # "Std. N.S+Dir C=1.5" : compute_errors(ref_file, pred_file3),
        # "Std. N.S+Dir C=0.5" : compute_errors(ref_file, pred_file4)
    }

    plot_errors(err_dict=err_dict, save_location=save_location, its="300k Its.")

if __name__ == "__main__":
    main()