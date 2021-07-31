import pandas as pd
import numpy as np

class InputData:
    def __init__(input):
        input.x_star = pd.DataFrame()#np.empty(33974,)
        input.y_star = pd.DataFrame()#np.empty(33974,)
        input.t_star = pd.DataFrame()#np.empty(251,)
        input.C_star = pd.DataFrame()#np.empty(8527474)
        input.U_star = pd.DataFrame()#np.empty(8527474)
        input.V_star = pd.DataFrame()#np.empty(8527474)
        input.P_star = pd.DataFrame()#np.empty(8527474)
        input.T = None
        input.N = None

    # we can operate on these inputs to get the rest of the data
    # the rest of these objects will be populated from input, so we can just write methods to do that
    def setTN(input):
        if (input.x_star.empty == False and input.t_star.empty == False):
            input.T = input.t_star.shape[0]
            input.N = input.x_star.shape[0]

    def splitTrainTestEqns(input):

        T, N = input.T, input.N
        T_star = np.tile(input.t_star, (1,N)).T
        X_star = np.tile(input.x_star, (1,N))
        Y_star = np.tile(input.y_star, (1,N))
        
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
        snap = np.array([100])
        test.t_test = T_star[:,snap]
        test.x_test = X_star[:,snap]
        test.y_test = Y_star[:,snap]    
        test.c_test = (input.C_star.to_numpy())[:,snap]
        test.u_test = (input.U_star.to_numpy())[:,snap]
        test.v_test = (input.V_star.to_numpy())[:,snap]
        test.p_test = (input.P_star.to_numpy())[:,snap]

        return train, eqns, test

class TrainingData:
    def __init__(train):
        train.t_data = pd.DataFrame()
        train.x_data = pd.DataFrame()
        train.y_data = pd.DataFrame()
        train.c_data = pd.DataFrame()

class Predictions:
    def __init__(preds):
        preds.c_star = pd.DataFrame()
        preds.u_star = pd.DataFrame()
        preds.v_star = pd.DataFrame()
        preds.p_star = pd.DataFrame()

class Equations:
    def __init__(eqns):
        eqns.t_eqns = pd.DataFrame()
        eqns.x_eqns = pd.DataFrame()
        eqns.y_eqns = pd.DataFrame()

class TestData:
    def __init__(test):
        test.t_test = pd.DataFrame()
        test.x_test = pd.DataFrame()
        test.y_test = pd.DataFrame()
        test.c_test = pd.DataFrame()
        test.u_test = pd.DataFrame()
        test.v_test = pd.DataFrame()
        test.p_test = pd.DataFrame()

    def __call__(test):
        return test.t_test, test.x_test, test.y_test

class Errors:
    def __init__(errs):
        errs.x = 1