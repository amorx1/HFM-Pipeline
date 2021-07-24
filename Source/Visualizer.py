from matplotlib.colors import rgb2hex
from numpy.lib.function_base import meshgrid
from pandas.core.reshape.pivot import pivot
import scipy.io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from ipywidgets import interactive
import seaborn as sb
import pandas as pd

data = scipy.io.loadmat("DATA/data_final.mat")

t = data['t_star']
x = data['x_star']
y = data['y_star']

U = data['U_star'] # N x T
V = data['V_star'] # N x T
P = data['P_star'] # N x T
C = data['C_star'] # N x T

# create a single frame to begin with.
# this is going to be a 2x2 array with x-y coordinates
# each frame is 1 out of the 250 timesteps

def generate_frames(x, y, t, C, U, V, P):
    frames = dict.fromkeys(range(251))
    for key in frames:
        frames[key] = pd.DataFrame()

        frames[key].loc[:,"x"] = x[:, 0]
        frames[key].loc[:,"y"] = y[:, 0]
        frames[key].loc[:,"C"] = C[:, key]
    
    return frames

def show(frames):
    pivotted = frames[100].pivot('y', 'x', 'C')
    sb.heatmap(pivotted)
    plt.show()

def main():
    frames = generate_frames(x, y, t, C, U, V, P)
    show(frames)

if __name__ == "__main__":
    main()