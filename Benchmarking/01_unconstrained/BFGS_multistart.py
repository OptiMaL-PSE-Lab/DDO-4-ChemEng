# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:38:36 2023

@author: adelrioc
"""

import numpy as np
import sobol_seq

######################################
# Forward finite differences 
######################################

def forward_finite_diff(f, x, Delta, f_old):
    n     = np.shape(x)[0]
    x     = x.reshape((n,1))
    dX    = np.zeros((n,1))
    
    for j in range(n):
        x_d_f    = np.copy(x)
        x_d_f[j] = x_d_f[j] + Delta
        dX[j]    = (f(x_d_f) - f_old)/Delta
    #
    return dX

#############################
# Line search function
#############################

def line_search_f(direction, x, f, lr, grad_k, f_old, armijo_wolfe=0):
    '''
    - function that determines optimal step with linesearch
    - Note: f and lr must be given
    - armijo_wolfe: 0 is naive, 1 is armijo, 2 is armijo-wolfe
    - for large iteration values (say >1000) A-W or A start being good options
    '''
    old_f = f_old ; new_f = old_f + 1.
    ls_i  = 0     ; lr_i  = 2.*lr
    c_1   = 1e-4  ;

    # if doing multistart this can be se to <10
    LS_max_iter = 8

    x_i   = x # if gradient is nan

    # --- Armijo line search --- #
    if armijo_wolfe == 1:
        # remember: 0<c_1<1
        armijo_ = old_f - c_1*lr_i*grad_k.T@direction
        while new_f>armijo_ and ls_i<LS_max_iter:
            lr_i  = lr_i/2.
            x_i   = x - lr_i*direction 
            new_f = f(x_i)
            ls_i += 1

    # ---  naive line-search --- #
    elif armijo_wolfe == 0:
        while new_f>old_f and ls_i<LS_max_iter:
            lr_i  = lr_i/2.
            x_i   = x - lr_i*direction 
            new_f = f(x_i)
            ls_i += 1

    # ---  Armijo and Wolfe line search --- #

    return x_i, ls_i, new_f

#############################
# Approximating Hessian
#############################

def Hk_f(x, x_past, grad_i, grad_i_past, Hk_past, Imatrix):
    '''
    function that approximates the iverse of the Hessian
    '''
    sk  = x - x_past 
    yk  = grad_i - grad_i_past
    rho = 1./(yk.T@sk+1e-7)

    Hinv = (Imatrix-rho*sk@yk.T)@Hk_past@(Imatrix-rho*yk@sk.T) + rho*sk@sk.T
    
    return Hinv

#############################
# First step 
#############################

def BFGS_step1(f, x0, n, grad_f, Imatrix, Delta, f_old):
    '''
    function that computes the first step for BFGS, because there is no Hk_past
    in the first interation, for x_past or grad_i_past. For this a steepest descent
    step is taken.
    '''
    grad_i      = grad_f(f,x0,Delta,f_old)
    x           = x0 - 1e-6*grad_i
    f_old1      = f(x)
    # past values
    x_past      = x0.reshape((n,1))
    grad_i_past = grad_i
    # new gradient
    grad_i      = grad_f(f,x,Delta,f_old1)
    # sk, yk, rho
    sk          = x - x_past 
    f_old2      = f(x)
    yk          = grad_i - grad_i_past
    # initial guess for H0
    Hk_past     = ((yk.T@sk)/(yk.T@yk))*Imatrix
    
    return Hk_past, grad_i_past, x_past, grad_i, x, f_old2

#############################
# multistart 
#############################

def x0_startf(bounds, n_s, N_x):
    '''
    Give starting points
    array([[0.  , 2.  , 7.5 ],
       [0.5 , 1.  , 8.75]])
    '''
    bounds_l = np.array([[ bounds[n_ix,1]-bounds[n_ix,0] ] for n_ix in range(len(bounds))])
    sobol_l  = sobol_seq.i4_sobol_generate(N_x, n_s)
    lb_l     = np.array([[bounds[i,0] for i in range(len(bounds))]])
    x0_start = lb_l  + sobol_l*bounds_l.T
    
    return x0_start

###################################
# BFGS for 'global search'
###################################

# [(ub, lb) for i in range(N_x)]
# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float 
def BFGS_gs(f, N_x, bounds, max_eval):
    '''
    Optimization algorithm: BFGS for global search with linesearch.
    Function evaluations are counted as if the algorithm was smartly coded - which it is not :)

    Notes:
    - if many iterations are possible switch to A-W or A line-search
    - N_x is input dimension
    '''
    # algorithm hyperparameters
    ns       = 5 
    lr       = 2.
    grad_f   = forward_finite_diff
    grad_tol = 1e-6

    # evaluate starting points
    x0_candidates = x0_startf(bounds, ns, N_x)
    f_l = []; 
    for xii in range(ns):
        f_l.append(f(x0_candidates[xii]))

    f_eval      = ns
    best_point  = ['none',1e15]    
    ns_eval     = ns
    Delta       = np.sqrt(np.finfo(float).eps)

    # multi-starting point loop
    while len(f_l)>=1 and f_eval<=max_eval:
        minindex      = np.argmin(f_l)
        x0            = x0_candidates[minindex]
        f_old         = f_l[minindex]
        #pop
        x0_candidates = x0_candidates.tolist()
        f_l.pop(minindex); x0_candidates.pop(minindex)
        x0_candidates = np.asarray(x0_candidates)
    
        # initialize problem
        n       = np.shape(x0)[0]
        x       = np.copy(x0); x = x.reshape((n,1))
        iter_i  = 0
        Imatrix = np.identity(n)
               
        # first step: gradient descent
        # compute gradient   
        Hk_past, grad_i_past, x_past, grad_i, x, f_old = BFGS_step1(f, x, n, grad_f, 
                                                                    Imatrix, Delta, f_old)
        f_eval                                        += 2*N_x + 2
        
        # optimization loop
        while np.sum(np.abs(grad_i)) > grad_tol and f_eval < max_eval:    

            # compute gradient   
            grad_i  = grad_f(f,x,Delta,f_old)
            f_eval += N_x
            # compute Hessian
            Hinv    = Hk_f(x, x_past, grad_i, grad_i_past, Hk_past, Imatrix)
            x_past  = x
            # step direction
            Df_i    = Hinv@grad_i 

            # line-search
            x_i, ls_i, new_f = line_search_f(Df_i, x, f, lr, grad_i, f_old)
            f_eval   += ls_i

            # record past points and gradients
            grad_i_past = grad_i
            Hk_past     = Hinv

            x       = x_i                
            iter_i += 1 

            if best_point[1] > new_f:
                best_point = [x, new_f]

        # if multistart points finish before iteration count
        if len(f_l)<=0:
            ns_eval      += ns
            x0_candidates = x0_startf(bounds, ns_eval, N_x)
            x0_candidates = x0_candidates[(ns_eval-ns):]
            for xii in range(ns):
                f_l.append(f(x0_candidates[xii]))  
    
    team_names = ['1','2']
    cids = ['01234567']
    return x, new_f, team_names, cids
