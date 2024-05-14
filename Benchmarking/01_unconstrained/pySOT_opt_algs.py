# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:12:20 2023

@author: adelrioc
"""

# requires https://pysot.readthedocs.io/en/latest/options.html
# also 
# commented lines 115 - 148 in utils
# changed  self.objective ->  self.objective.eval in controller.py line 166
# also replaced np.int with np.int32
# in sop_strategy changed if rank == 1 -> if rank.any() == 1 : line 347 (this I am unsure of consequence)

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import DYCORSStrategy, SRBFStrategy, SOPStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley, Levy
from pySOT.utils import progress_plot
from poap.controller import ThreadController, SerialController, BasicWorkerThread
import numpy as np



#########################
# --- Random search --- #
#########################

# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float 
def Random_searchDYCORS(f, n_p, bounds_rs, iter_rs):
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
        localval[sample_i] = f.eval(x_trial) # f
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return f_b,x_b

####################
# DYCORS algorithm #
####################

def opt_DYCORS(f, x_dim, bounds, iter_tot):
    '''
    Combining radial basis function surrogates and dynamic coordinate search 
    in high-dimensional expensive black-box optimization
    '''

    n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best = Random_searchDYCORS(f, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs
    
    # DYCORS
    # define surrogate
    rbf = RBFInterpolant(
        dim=x_dim, lb=bounds[:,0], ub=bounds[:,1], kernel=CubicKernel(), 
        tail=LinearTail(x_dim))
    slhd = SymmetricLatinHypercube(dim=x_dim, num_pts=2*(x_dim+1))

    # optimization
    controller = SerialController(objective=f)
    controller.strategy = DYCORSStrategy(
        max_evals=iter_, opt_prob=f, asynchronous=False, 
        exp_design=slhd, surrogate=rbf, num_cand=100*x_dim,
        batch_size=1)

    result = controller.run()
    #print(result)

    return result, None, None, None

####################
# DYCORS algorithm #
####################

def opt_SRBF(f, x_dim, bounds, iter_tot):
    '''
    Combining radial basis function surrogates and dynamic coordinate search 
    in high-dimensional expensive black-box optimization
    '''

    n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best = Random_searchDYCORS(f, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs
    
    # DYCORS
    # define surrogate
    rbf = RBFInterpolant(
        dim=x_dim, lb=bounds[:,0], ub=bounds[:,1], kernel=CubicKernel(), 
        tail=LinearTail(x_dim))
    slhd = SymmetricLatinHypercube(dim=x_dim, num_pts=2*(x_dim+1))

    # optimization
    controller = SerialController(objective=f)
    controller.strategy = SRBFStrategy(
        max_evals=iter_, opt_prob=f, asynchronous=False, 
        exp_design=slhd, surrogate=rbf, num_cand=100*x_dim,
        batch_size=1)

    result = controller.run()
    #print(result)

    return result, None, None, None

####################
# SOPStrategy algorithm #
####################

def opt_SOP(f, x_dim, bounds, iter_tot):
    '''
    Combining radial basis function surrogates and dynamic coordinate search 
    in high-dimensional expensive black-box optimization
    '''

    n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best = Random_searchDYCORS(f, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs
    
    # DYCORS
    # define surrogate
    rbf = RBFInterpolant(
        dim=x_dim, lb=bounds[:,0], ub=bounds[:,1], kernel=CubicKernel(), 
        tail=LinearTail(x_dim))
    slhd = SymmetricLatinHypercube(dim=x_dim, num_pts=2*(x_dim+1))

    # optimization
    controller = SerialController(objective=f)
    controller.strategy = SOPStrategy(
        max_evals=iter_, opt_prob=f, asynchronous=False, 
        exp_design=slhd, surrogate=rbf, num_cand=100*x_dim,
        batch_size=1)

    result = controller.run()
    #print(result)

    return result, None, None, None
























