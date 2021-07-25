import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cupy

data = scipy.io.loadmat("DATA/data_final.mat")

t = data['t_star']
x = data['x_star']
y = data['y_star']

U = data['U_star'] # N x T
V = data['V_star'] # N x T
P = data['P_star'] # N x T
C = data['C_star'] # N x T

def write_vtu(x, y, t, C, U, V, P):
    grid_x, grid_y = cp.mgrid[min(cp.array(x)):max(cp.array(x)):10000j, min(cp.array(y)):max(cp.array(y)):10000j]
    points = np.concatenate((x, y), axis=1)
    values = C[:,150]
    
    grid_linear = griddata(points, values, (cp.asnumpy(grid_y), cp.asnumpy(grid_x)), method='linear', fill_value=0.0)
    
    plt.imshow(grid_linear)
    plt.show()

def main():
    # frames = generate_frames(x, y, t, C, U, V, P)
    # show(frames)
    write_vtu(x, y, t, C, U, V, P)
    print("Done")

if __name__ == "__main__":
    main()