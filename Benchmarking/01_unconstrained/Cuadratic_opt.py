# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:30:21 2023

@author: adelrioc
"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
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
        localval[sample_i] = f.fun_test(x_trial) # f
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return f_b,x_b

###################################
# --- Random Sampling in ball --- #
###################################

def Ball_sampling(ndim, r_i):
    '''
    This function samples randomly withing a ball of radius r_i
    '''
    u      = np.random.normal(0,1,ndim)  # random sampling in a ball
    norm   = np.sum(u**2)**(0.5)
    r      = random.random()**(1.0/ndim)
    d_init = r*u/norm*r_i*2      # random sampling in a ball

    return d_init

###########################################
# --- Weighted Least Squares function --- #
###########################################

def Wighted_LS(params, X, Y):
    '''
    X: matrix: [x^(1),...,x^(n_d)]
    Y: vecor:  [f(x^(1)),...,f(x^(n_d))]
    The weighted least squares assumes heterostochasticity to the data and takes that into account when updating the model. 
    Technically this means that a new point influences the new model with respect to the variance observed in the region 
    where the sample is taken from.
    '''
    # number of datapoints & dimensions
    n_d = Y.shape[0] 
    n_x = X.shape[1]
    # weight structuring
    b = params[0]
    c = params[1:n_x+1].reshape((n_x,1))
    Q = toeplitz(params[n_x+1:2*n_x+1]) + np.diag(params[2*n_x+1:3*n_x+1])
    # weighted least squares
    WLS = (np.sum([(b + c.T@X[i,:] + X[i,:].T@Q@X[i,:] - Y[i])**2/(Y[i]**2+1) for i in range(n_d)])/n_d + 
           1e-3*np.sum(params**2)/(3*n_x))

    return WLS

##########################################
# --- Estimating the quadratic model --- #
##########################################

def quadratic_model_estimation(X, Y, p0, n_x):
    '''
    X: matrix: [x^(1),...,x^(n_d)]
    Y: vecor:  [f(x^(1)),...,f(x^(n_d))]
    p0: initial guess, preferably from last iteration

    TODO: check if different optimization method (i.e. BFGS) is better
    '''
    # minimizing weighted least squares function with scipy
    res = minimize(Wighted_LS, args=(X,Y), x0=p0, method='SLSQP')
    # obtaining solution
    params = res.x
    #f_val  = res.fun
    b = params[0]
    c = params[1:n_x+1].reshape((n_x,1))
    Q = toeplitz(params[n_x+1:2*n_x+1]) + np.diag(params[2*n_x+1:3*n_x+1])

    return b, c, Q, params

########################################
# --- quadratic surrogate function --- #
########################################

def quad_func(d, b,c,Q,x0var):
    '''
    d: vector:  distance from centre x0var
    '''

    return b + c.T@(x0var+d) + (x0var+d).T@Q@(x0var+d)

##########################################
# --- optimising the quadratic model --- #
##########################################

def opt_quadratic_model(b,c,Q,x0var,r_t):
    '''
    a,b,c: parameters estimated for the quadratic model
    x0var: initial point: last
    '''
    
    # minimising quadratic model
    res = minimize(quad_func, args=(b,c,Q,x0var), x0=np.zeros(c.shape[0]), 
                   method='SLSQP', bounds=([[-r_t,r_t]]*c.shape[0]))
    # Note: bounds are added: nonlinear trust region is handled poorly by SLSQP

    # retrieving solution
    d_sol = res.x
    #print('res = ',res) print status
    # returning solution
    return x0var + d_sol

################################
# --- TR and center update --- #
################################

def update_tr_center(quad_func, b,c,Q,xold, xnew,f, g_r,g_i,r):
    '''
    
    '''
    # for simplicity we will re-evaluate functions
    # evaluating function
    f_new = f.fun_test(xnew)
    f_old = f.fun_test(xold)
    # if new point is worse than old point
    if f_new>=f_old:
        return xold, r*g_r

    # evaluating model
    m_new = quad_func(0, b,c,Q,xnew)
    m_old = quad_func(0, b,c,Q,xold)

    rho = (f_old - f_new)/(m_old - m_new)
    if rho > 0.5:
        return xnew, r*g_i
    else:
        return xnew, r*g_r

##################################################
# --- Local search with quadratic surrogate  --- #
##################################################

# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float
def LS_QM(f, x_dim, bounds, iter_tot, x_start = False):
    '''
    This function is an optimization routine following a local random search
    '''
    r    = np.mean((bounds[:,1] - bounds[:,0]))*0.1  # function on bounds_ss
    
    n_s  = x_dim+1                               # initial iterations to construct surrogate
    gamma_r = 0.5
    gamma_i = 1.5



    if x_start == False:

        # iterations to find good starting point
        n_rs = int(min(20,max(iter_tot*.1,5)))

        # function of iter_ss
        n_i  = iter_tot - x_dim - n_rs

        # array for estimating models
        f_list  = np.zeros((iter_tot-n_rs))
        x_list  = np.zeros((iter_tot-n_rs,x_dim))

        # evaluate first point
        f_best, x_best = f_list[0], x_list[0,:] = Random_search(f, x_dim, bounds, n_rs)
        print(f_best)
        print(x_best)

    else: 

        # subtract for given starting point
        n_rs = 1

        # function of iter_ss
        n_i  = iter_tot - x_dim - n_rs

        # array for estimating models
        f_list  = np.zeros((iter_tot-n_rs))
        x_list  = np.zeros((iter_tot-n_rs,x_dim))

        print('test')
        print(f)
        x_best = x_list[0,:] = f.x0[0].flatten() # TODO this needs to be automatized and adjusted to higher dimensions - works only for 2
        f_best = f_list[0] = f.fun_test(x_best)

    # === first sampling inside the radius === #
    # - (similar to stochastic local search: with proper programming should be a function) - #
    localx   = np.zeros((n_s,x_dim))  # points sampled
    localval = np.zeros((n_s))        # function values sampled
    # sampling loop
    for sample_i in range(n_s):
        x_trial            = x_best + Ball_sampling(x_dim, r) # sampling
        localx[sample_i,:] = x_trial
        localval[sample_i] = f.fun_test(x_trial)
    # tracking evaluations
    f_list[1:n_s+1]   = localval
    x_list[1:n_s+1,:] = localx

    # === Estimate quadratic model === #
    p0       = np.ones(3*x_dim+1)
    b,c,Q,p0 = quadratic_model_estimation(x_list[0:n_s+1,:], f_list[0:n_s+1], p0, x_dim)

    # === main loop === #
    for iter_i in range(n_i):

        # minimise the surrogate model
        x_trial = opt_quadratic_model(b,c,Q,x_best,r)
        # evaluate function
        f_trial = f.fun_test(x_trial)
        # add new points to trainig set
        x_list[n_s+1+iter_i:n_s+1+iter_i+1,:] = x_trial
        f_list[n_s+1+iter_i:n_s+1+iter_i+1]   = f_trial

        # update trust region and center point
        x_best, r = update_tr_center(quad_func, b,c,Q,x_best, x_trial,f, 
                                     gamma_r, gamma_i, r) 

        # === re-Estimate quadratic model === #
        b,c,Q,p0 = quadratic_model_estimation(x_list[0:n_s+1+iter_i+1,:],
                                              f_list[0:n_s+1+iter_i+1], p0, x_dim)
        
        # tracking best (this in theory should not be re-evaluated)
        f_best = f.fun_test(x_best)

    #print('best value found ',f_best)
    #print('best point found ',x_best)

    team_names = ['3','4']
    cids = ['01234567']
    return x_best, f_best, team_names, cids



##################################################
# --- PLOTTING  --- #
##################################################

# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float
def LS_QM_plot(f, x_dim, bounds, iter_tot):
    '''
    This function is an optimization routine following a local random search
    '''
    r    = np.mean((bounds[:,1] - bounds[:,0]))*0.1  # function on bounds_ss
    n_rs = int(min(20,max(iter_tot*.1,5)))       # iterations to find good starting point
    n_s  = x_dim+1                               # initial iterations to construct surrogate
    n_i  = iter_tot - x_dim - n_rs               # function of iter_ss
    gamma_r = 0.5
    gamma_i = 1.5
    
    # array for estimating models
    f_list  = np.zeros((iter_tot-n_rs))
    x_list  = np.zeros((iter_tot-n_rs,x_dim))

    # evaluate first point
    f_best, x_best = f_list[0], x_list[0,:] = Random_search(f, x_dim, bounds, n_rs)

    # === first sampling inside the radius === #
    # - (similar to stochastic local search: with proper programming should be a function) - #
    localx   = np.zeros((n_s,x_dim))  # points sampled
    localval = np.zeros((n_s))        # function values sampled
    # sampling loop
    for sample_i in range(n_s):
        x_trial            = x_best + Ball_sampling(x_dim, r) # sampling
        localx[sample_i,:] = x_trial
        localval[sample_i] = f.fun_test(x_trial)
    # tracking evaluations
    f_list[1:n_s+1]   = localval
    x_list[1:n_s+1,:] = localx

    # === Estimate quadratic model === #
    p0       = np.ones(3*x_dim+1)
    b,c,Q,p0 = quadratic_model_estimation(x_list[0:n_s+1,:], f_list[0:n_s+1], p0, x_dim)

    # === main loop === #
    for iter_i in range(n_i):

        # minimise the surrogate model
        x_trial = opt_quadratic_model(b,c,Q,x_best,r)
        # evaluate function
        f_trial = f.fun_test(x_trial)
        # add new points to trainig set
        x_list[n_s+1+iter_i:n_s+1+iter_i+1,:] = x_trial
        f_list[n_s+1+iter_i:n_s+1+iter_i+1]   = f_trial

        # update trust region and center point
        x_best, r = update_tr_center(quad_func, b,c,Q,x_best, x_trial,f, 
                                     gamma_r, gamma_i, r) 

        # === re-Estimate quadratic model === #
        b,c,Q,p0 = quadratic_model_estimation(x_list[0:n_s+1+iter_i+1,:],
                                              f_list[0:n_s+1+iter_i+1], p0, x_dim)
        
        # tracking best (this in theory should not be re-evaluated)
        f_best = f.fun_test(x_best)

    X_opt = np.array((x_list))

    #print('best value found ',f_best)
    #print('best point found ',x_best)

    team_names = ['3','4']
    cids = ['01234567']
    return x_best, f_best, team_names, cids, X_opt