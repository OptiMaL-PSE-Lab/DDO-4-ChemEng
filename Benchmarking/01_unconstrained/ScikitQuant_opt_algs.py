# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:12:20 2023

@author: adelrioc
"""

# requires https://scikit-quant.readthedocs.io/en/latest/installation.html
# also 

from skquant.opt import minimize
import numpy as np

#########################
# --- Random search --- #
#########################

# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float 
def Random_search(f, n_p, bounds_rs, iter_rs):
    '''
    This function is a naive optimization routine that randomly samples the 
    allowed space and returns the best value.
    '''

    # arrays to store sampled points
    localx   = np.zeros((n_p,iter_rs))  # points sampled
    localval = np.zeros((iter_rs))        # function values sampled
    # bounds
    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        x_trial = np.random.uniform(0, 1, n_p)*bounds_range + bounds_bias # sampling
        localx[:,sample_i] = x_trial
        localval[sample_i] = f.fun_test(x_trial) # f
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return f_b,x_b

####################
# SnobFit algorithm #
####################

def opt_SnobFit(f, x_dim, bounds, iter_tot, has_x0 = False):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    if has_x0 == True:

        iter_          = iter_tot - 1
        result, history = minimize(f.fun_test, f.x0[0].flatten() , bounds, iter_, method='SnobFit') 

    else:

        # n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point
        n_rs = int(max(x_dim+1,iter_tot*.05))

        # evaluate first point
        f_best, x_best = Random_search(f, x_dim, bounds, n_rs)
        iter_          = iter_tot - n_rs
        result, history = minimize(f.fun_test, x_best, bounds, iter_, method='SnobFit') 

    return result.optpar, result.optval, None, None

####################
# Bobyqa  algorithm #
####################

def opt_Bobyqa(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best = Random_search(f.fun_test, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs

    result, history = \
    minimize(f.fun_test, x_best, bounds, iter_, method='Bobyqa') 

    return result.optpar, result.optval
























