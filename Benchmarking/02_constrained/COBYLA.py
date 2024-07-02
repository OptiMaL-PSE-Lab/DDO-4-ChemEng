from scipy.optimize import minimize
from scipy.optimize import direct, Bounds
import scipy
import numpy as np

def COBYLA(
        f,
        x_dim, #TODO here the functionality of multiple input dimensions has to be implemented
        bounds, # given by the benchmarking, probably not used in COBYLA
        f_eval_, # length of trajectory (objective function evaluation budget)
        i_rep
        ): 
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    take_x_start: boolean to either receive the starting point from f or to randomly sample it
    in case the starting point is randomly sampled, the number of samples is deducted from the evaluation budget. Mind, that 
    the number of samples 
    in case the starting point is taken from the function, this is treated as us providing the algorithm with a starting point.
    Therefore this counts as a single evaluation from the evaluation budget
    '''

    constraints = [{'type': 'ineq', 'fun': f.con_test}]
    iter_ = f_eval_
    x_start = f.x0[i_rep].flatten()

    # Record intermediate solutions
    # x_start needs to be added manually to the trajectory because it is not stored by scipy.minimize
    trajectory = [x_start]

    def record_trajectory(x):
        trajectory.append(x)

    opt = minimize(
        f.fun_test, 
        x_start, 
        method='COBYLA', 
        constraints = constraints,
        bounds=bounds,
        options={'maxfev': iter_, 'disp': False},
        callback = record_trajectory,
        ) 
    
    X_opt_plot = np.array(trajectory)
    TR_l = [] # TODO find a way to plot the trust region from cobyla
    backtrck_l = [] # TODO find out what this is for
    samples_number = len(X_opt_plot)
    xnew = X_opt_plot[-1]
    team_names = ['9','10']
    cids = ['01234567']
    return opt.x, opt.fun, team_names, cids, X_opt_plot, TR_l, xnew, backtrck_l, samples_number
