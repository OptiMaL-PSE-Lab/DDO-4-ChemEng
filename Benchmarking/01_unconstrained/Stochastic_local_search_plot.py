# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 22:16:01 2023

@author: adelrioc
"""

import numpy as np
import random

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
        localval[sample_i] = f(x_trial) # f
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    team_names = ['5','6']
    cids = ['01234567']
    return f_b, x_b, team_names, cids

###############################
# --- Random local search --- #
###############################

# --- Random Sampling in ball --- #

def Ball_sampling(ndim, r_i):
    '''
    This function samples randomly withing a ball of radius r_i
    '''
    u      = np.random.normal(0,1,ndim)  # random sampling in a ball
    norm   = np.sum(u**2)**(0.5)
    r      = random.random()**(1.0/ndim)
    d_init = r*u/norm*r_i*2      # random sampling in a ball

    return d_init

def Random_local_search_PE(f, f_best, x_best, n_ls, N_x, bounds_rs):
    '''
    This function is an optimization routine following a local random search

    n_i: total number of interations
    n_s: number of samples per iteration
    '''
    # algorithm hyperparameters
    r     = (bounds_rs[:,1] - bounds_rs[:,0])*.5
    n_s   = 10
    n_i   = int(n_ls/n_s)
    gamma = .9

    # extract dimension
    x_dim = N_x

    for iter_i in range(n_i):

        # sampling inside the radius r
        localx   = np.zeros((x_dim,n_s))  # points sampled
        localval = np.zeros((n_s))        # function values sampled
        # sampling loop
        for sample_i in range(n_s):
            x_trial = x_best + Ball_sampling(x_dim, r) # sampling
            localx[:,sample_i] = x_trial
            localval[sample_i] = f(x_trial)
        # choosing the best
        minindex = np.argmin(localval)
        f_b      = localval[minindex]
        x_b      = localx[:,minindex]

        # comparing vs best point so far
        if f_b < f_best:
            x_best, f_best = x_b, f_b
        else:
            r = gamma*r

    return x_best, f_best


#############################
# --- Stochastic search --- #
#############################
# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float 
def SS_alg_plot(f, N_x, bounds_ss, iter_ss):

    iter_rs = 3
    n_ls    = iter_ss - iter_rs

    f_best, x_best, team_names, cids = Random_search(f, N_x, bounds_ss, iter_rs)

    f_best, x_best = Random_local_search_PE(f, f_best, x_best, n_ls, N_x, bounds_ss)
    
    team_names = ['5','6']
    cids = ['01234567']
    return x_best, f_best, team_names, cids