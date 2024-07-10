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
    if f.func_type == 'WO_f': constraints = [
        {'type': 'ineq', 'fun': f.WO_con1_test}, 
        {'type': 'ineq', 'fun': f.WO_con2_test}
        ]
    else: constraints = [{'type': 'ineq', 'fun': f.con_test}]
    iter_ = f_eval_
    x_start = f.x0[i_rep].flatten()

    opt = minimize(
        f.fun_test, 
        x_start, 
        method='COBYLA', 
        constraints = constraints,
        bounds=bounds,
        options={'maxiter': iter_, 'disp': False},
        ) 
    

    TR_l = [] # TODO find a way to plot the trust region from cobyla
    backtrck_l = [] # TODO find out what this is for
    xnew = f.f_list[-1]
    team_names = ['9','10']
    cids = ['01234567']
    return opt.x, opt.fun, team_names, cids, None, TR_l, xnew, backtrck_l, None
