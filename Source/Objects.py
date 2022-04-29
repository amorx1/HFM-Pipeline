import pandas as pd
import numpy as np
import csv

class InputData:
    def __init__(input, no_slip, dirichlet, patch_ID):
        input.x_star = pd.DataFrame()
        input.y_star = pd.DataFrame()
        input.t_star = pd.DataFrame()
        input.C_star = pd.DataFrame()
        input.U_star = pd.DataFrame()
        input.V_star = pd.DataFrame()
        input.P_star = pd.DataFrame()
        input.T = None
        input.N = None
        input.no_slip = no_slip
        input.dirichlet = dirichlet
        input.patch_ID = patch_ID

    def setTN(input):
        if (input.x_star.empty == False and input.t_star.empty == False):
            input.T = input.t_star.shape[0]
            input.N = input.x_star.shape[0]


    def splitTrainTestEqns(input):
        patch_ID = input.patch_ID.flatten()[:,None]

        t_star = input.t_star#[0:58,0:] # T x 1
        x_star = input.x_star#[0:,0:58] # N x 1
        y_star = input.y_star#[0:,0:58] # N x 1

        T = t_star.shape[0]
        N = x_star.shape[0]
        
        U_star = input.U_star#[0:,0:58] # N x T
        V_star = input.V_star#[0:,0:58] # N x T
        P_star = input.P_star#[0:,0:58] # N x T
        C_star = input.C_star#[0:,0:58] # N x T

        # Rearrange Data 
        T_star = np.tile(t_star, (1,N)).T # N x T
        X_star = np.tile(x_star, (1,T)) # N x T
        Y_star = np.tile(y_star, (1,T)) # N x T
        
        t = T_star.flatten()[:,None] # NT x 1
        x = X_star.flatten()[:,None] # NT x 1
        y = Y_star.flatten()[:,None] # NT x 1
        u = U_star.to_numpy().flatten()[:,None] # NT x 1
        v = V_star.to_numpy().flatten()[:,None] # NT x 1
        p = P_star.to_numpy().flatten()[:,None] # NT x 1
        c = C_star.to_numpy().flatten()[:,None] # NT x 1
        
        # Training data
        train = TrainingData()
        T_data = T
        N_data = N
        idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
        idx_x = np.random.choice(N, N_data, replace=False)
        train.t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
        train.x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
        train.y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
        train.c_data = (input.C_star.to_numpy())[:, idx_t][idx_x,:].flatten()[:,None]

        if input.dirichlet:
            # Inlet
            train.t_inlet = t[np.where((patch_ID == 2) | (patch_ID == 3) | (patch_ID == 4) | (patch_ID == 5))][:,None]
            train.x_inlet = x[np.where((patch_ID == 2) | (patch_ID == 3) | (patch_ID == 4) | (patch_ID == 5))][:,None]
            train.y_inlet = y[np.where((patch_ID == 2) | (patch_ID == 3) | (patch_ID == 4) | (patch_ID == 5))][:,None]
            train.u_inlet = u[np.where((patch_ID == 2) | (patch_ID == 3) | (patch_ID == 4) | (patch_ID == 5))][:,None]
            train.v_inlet = v[np.where((patch_ID == 2) | (patch_ID == 3) | (patch_ID == 4) | (patch_ID == 5))][:,None]

            # Outlet
            train.t_outlet = t[(np.where(patch_ID == 6))][:, None]
            train.x_outlet = x[(np.where(patch_ID == 6))][:, None]
            train.y_outlet = y[(np.where(patch_ID == 6))][:, None]
            train.p_outlet = p[(np.where(patch_ID == 6))][:, None]

    
        if input.no_slip:
            train.t_ns = t[(np.where(patch_ID == 6))][:, None]
            train.x_ns = x[(np.where(patch_ID == 6))][:, None]
            train.y_ns = y[(np.where(patch_ID == 6))][:, None]

        # Equations
        eqns = Equations()
        T_eqns = T
        N_eqns = N
        idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
        idx_x = np.random.choice(N, N_eqns, replace=False)
        eqns.t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
        eqns.x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
        eqns.y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]

        # Test Data
        test = TestData()
        for snap in range(0,input.T):
            test.t_test = T_star[:,snap:snap+1]
            test.x_test = X_star[:,snap:snap+1]
            test.y_test = Y_star[:,snap:snap+1]
            
            test.c_test = (input.C_star.to_numpy())[:,snap:snap+1]
            test.u_test = (input.U_star.to_numpy())[:,snap:snap+1]
            test.v_test = (input.V_star.to_numpy())[:,snap:snap+1]
            test.p_test = (input.P_star.to_numpy())[:,snap:snap+1]

        return train, eqns, test

class TrainingData:
    def __init__(train):

        train.t_data = pd.DataFrame()
        train.x_data = pd.DataFrame()
        train.y_data = pd.DataFrame()
        train.c_data = pd.DataFrame()
        train.p_data = pd.DataFrame()

        # Inlet for dirichlet
        train.t_inlet = pd.DataFrame()
        train.x_inlet = pd.DataFrame()
        train.y_inlet = pd.DataFrame()
        train.u_inlet = pd.DataFrame()
        train.v_inlet = pd.DataFrame()

        # Outlet for dirichlet
        train.t_outlet = pd.DataFrame()
        train.x_outlet = pd.DataFrame()
        train.y_outlet = pd.DataFrame()
        train.p_outlet = pd.DataFrame()

        # No-slip boundary
        train.t_ns = pd.DataFrame()
        train.x_ns = pd.DataFrame()
        train.y_ns = pd.DataFrame()

class Predictions:
    def __init__(preds):
        preds.c_pred = pd.DataFrame()
        preds.u_pred = pd.DataFrame()
        preds.v_pred = pd.DataFrame()
        preds.p_pred = pd.DataFrame()

class Equations:
    def __init__(eqns):
        eqns.t_eqns = pd.DataFrame()
        eqns.x_eqns = pd.DataFrame()
        eqns.y_eqns = pd.DataFrame()
        eqns.c_eqns = pd.DataFrame()
        eqns.u_eqns = pd.DataFrame()
        eqns.v_eqns = pd.DataFrame()
        eqns.p_eqns = pd.DataFrame()

class TestData:
    def __init__(test):
        test.t_test = pd.DataFrame()
        test.x_test = pd.DataFrame()
        test.y_test = pd.DataFrame()
        test.c_test = pd.DataFrame()
        test.u_test = pd.DataFrame()
        test.v_test = pd.DataFrame()
        test.p_test = pd.DataFrame()

    def get(test):
        return test.t_test, test.x_test, test.y_test

class Errors:
    def __init__(errs):
        errs.error_c = pd.DataFrame()
        errs.error_u = pd.DataFrame()
        errs.error_v = pd.DataFrame()
        errs.error_p = pd.DataFrame()

    def plot(errs):
        return 