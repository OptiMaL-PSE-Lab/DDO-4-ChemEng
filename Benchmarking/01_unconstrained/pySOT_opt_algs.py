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

# class Exp_dsgn_custom:

#     def generate_points(lb, ub, int_var):

#     '''
#     this algorithm creates a list of initial points based on a latin hypercube. The idea for the plotting ist
#     that I generate a starting point which is taken from the function and then find out, how many more points are generated
#     by the algorithm (here it does latin hypercube sampling) and then it probably selects the best point out of these to start the 
#     optimization. So in order for the plotting function I have to make sure, that a) the point from the test-function
#     is somehow inserted in this list of initial points, and secondly, that the lines connecting the points in the plot all start from 
#     this starting point.

#     ''' 


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
        localval[sample_i] = f.fun_test(x_trial) # f
        
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return f_b,x_b

####################
# DYCORS algorithm #
####################

def opt_DYCORS(f, x_dim, bounds, iter_tot, has_x0=False):
    '''
    Combining radial basis function surrogates and dynamic coordinate search 
    in high-dimensional expensive black-box optimization
    '''

    # define surrogate
    rbf = RBFInterpolant(
        dim=x_dim, lb=bounds[:,0], ub=bounds[:,1], kernel=CubicKernel(), 
        tail=LinearTail(x_dim))

    # define controller
    controller = SerialController(objective=f)

    # for plotting
    if has_x0 == True:

        # receive first point from function
        x_start = f.x0[0].reshape((1,2))
        
        # evaluate first point from function
        f_start = np.array(f.fun_test(x_start)).reshape((1,1))

        # subtract from budget
        iter_          = iter_tot - 1

        # define experimental design
        slhd_minus1 = SymmetricLatinHypercube(dim=x_dim, num_pts=2*(x_dim+1)-1)

        # define strategy
        controller.strategy = DYCORSStrategy(
            max_evals=iter_, 
            opt_prob=f, 
            asynchronous=False, 
            exp_design=slhd_minus1, 
            extra_points=np.array(x_start),
            extra_vals=np.array(f_start),
            surrogate=rbf, 
            num_cand=100*x_dim,
            batch_size=1)

        result = controller.run()
        print(result)

        return result, None, None, None

    else:

        n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

        # evaluate first point
        f_best, x_best = Random_searchDYCORS(f, x_dim, bounds, n_rs)
        iter_          = iter_tot - n_rs

        # define experimental design
        slhd = SymmetricLatinHypercube(dim=x_dim, num_pts=2*(x_dim+1))

        # optimization
        controller = SerialController(objective=f)
        controller.strategy = DYCORSStrategy(
            max_evals=iter_, 
            opt_prob=f, 
            asynchronous=False, 
            exp_design=slhd, 
            surrogate=rbf, 
            num_cand=100*x_dim,
            batch_size=1)

        result = controller.run()
        print(result)

        return result, None, None, None

####################
# DYCORS algorithm #
####################

def opt_SRBF(f, x_dim, bounds, iter_tot, has_x0 = False):
    '''
    Combining radial basis function surrogates and dynamic coordinate search 
    in high-dimensional expensive black-box optimization
    '''

    if has_x0 == True:
    
        # receive first point from function
        x_start = f.x0[0].reshape((1,2))
        
        # evaluate first point from function
        f_start = np.array(f.fun_test(x_start)).reshape((1,1))

        # subtract from budget
        iter_          = iter_tot - 1

        # define surrogate
        rbf = RBFInterpolant(
            dim=x_dim, lb=bounds[:,0], ub=bounds[:,1], kernel=CubicKernel(), 
            tail=LinearTail(x_dim))
        
        # define experimental design
        slhd_minus1 = SymmetricLatinHypercube(dim=x_dim, num_pts=2*(x_dim+1)-1)

        # optimization
        controller = SerialController(objective=f)
        controller.strategy = SRBFStrategy(
            max_evals=iter_, 
            opt_prob=f, 
            asynchronous=False, 
            exp_design=slhd_minus1, 
            extra_points=np.array(x_start),
            extra_vals=np.array(f_start),
            surrogate=rbf, 
            num_cand=100*x_dim,
            batch_size=1
            )

        result = controller.run()
        #print(result)

        return result, None, None, None

    else:
        n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

        # evaluate first point
        f_best, x_best = Random_searchDYCORS(f, x_dim, bounds, n_rs)
        iter_          = iter_tot - n_rs
        
        # define surrogate
        rbf = RBFInterpolant(
            dim=x_dim, lb=bounds[:,0], ub=bounds[:,1], kernel=CubicKernel(), 
            tail=LinearTail(x_dim))
        
        # define experimental design
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

def opt_SOP(f, x_dim, bounds, iter_tot, has_x0 = False):
    '''
    Combining radial basis function surrogates and dynamic coordinate search 
    in high-dimensional expensive black-box optimization
    '''
    
    if has_x0 == True:
    
        # receive first point from function
        x_start = f.x0[0].reshape((1,2))
        
        # evaluate first point from function
        f_start = np.array(f.fun_test(x_start)).reshape((1,1))

        # subtract from budget
        iter_          = iter_tot - 1

        # define surrogate
        rbf = RBFInterpolant(
            dim=x_dim, lb=bounds[:,0], ub=bounds[:,1], kernel=CubicKernel(), 
            tail=LinearTail(x_dim))
        
        # define experimental design
        slhd_minus1 = SymmetricLatinHypercube(dim=x_dim, num_pts=2*(x_dim+1)-1)

        # optimization
        controller = SerialController(objective=f)
        controller.strategy = SOPStrategy(
            max_evals=iter_, 
            opt_prob=f, 
            asynchronous=False, 
            exp_design=slhd_minus1,
            extra_points=np.array(x_start),
            extra_vals=np.array(f_start),
            surrogate=rbf, 
            num_cand=100*x_dim,
            batch_size=1
            )

        result = controller.run()
        #print(result)

        return result, None, None, None

    
    else:
        n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

        # evaluate first point
        f_best, x_best = Random_searchDYCORS(f, x_dim, bounds, n_rs)
        iter_          = iter_tot - n_rs
        
        # DYCORS
        # define surrogate
        rbf = RBFInterpolant(
            dim=x_dim, lb=bounds[:,0], ub=bounds[:,1], kernel=CubicKernel(), 
            tail=LinearTail(x_dim))
        
        # define experimental design
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
























