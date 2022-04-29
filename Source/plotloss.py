import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

pred_file1 = "/Users/akshay/Downloads/preds(1).mat"
pred_file2 = "/Users/akshay/Documents/Project_models/4_inlets/Weighed-200/300k_its/preds.mat"
pred_file3 = "/Users/akshay/Documents/Project_models/4_inlets/Weighted-175/300k_its/preds.mat"
pred_file4 = "/Users/akshay/Documents/Project_models/4_inlets/Weighted-050/300k_its/preds.mat"


def plot_loss():
    # preds1 = scipy.io.loadmat(pred_file1)
    preds2 = scipy.io.loadmat(pred_file2)
    preds3 = scipy.io.loadmat(pred_file3)
    preds4 = scipy.io.loadmat(pred_file4)

    # loss1 = preds1["Losses"].T
    # loss1 = loss1[30:len(loss1):100]

    loss2 = preds2["Losses"].T
    loss2 = loss2[0:len(loss2):100]

    loss3 = preds3["Losses"].T
    loss3 = loss3[0:len(loss3):100]

    loss4 = preds4["Losses"].T
    loss4 = loss4[0:len(loss4):100]

    its = np.linspace(1, 301, 301)
    its2 = np.linspace(1, 301, 267)

    fig, ax = plt.subplots()
    # ax.plot(its, loss1, label="C=1.0")
    ax.plot(its2, loss2, label="C=2.0")
    ax.plot(its, loss3, label="C=1.5")
    ax.plot(its, loss4, label="C=0.5")
    plt.title("Loss profile for varying C")
    plt.ylabel("Loss")
    plt.xlabel("Iteration (x1000)")
    plt.legend()
    plt.show()
    return

def main():
    plot_loss()

if __name__ == "__main__":
    main()