
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:30:21 2023

@author: adelrioc
"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
import random
import sobol_seq
from scipy.stats import qmc

########################
# Quadratic Regression #
########################

class Quad_Reg_model:

    def __init__(self, X, Y, x_centre):
        
        # Quad model variable definitions
        self.X, self.Y              = X, Y
        self.n_point, self.nx_dim   = X.shape[0], X.shape[1]
        self.ny_dim                 = Y.shape[1]
        
        # normalize data
        self.X_mean, self.X_std    = np.mean(X, axis=0), np.std(X, axis=0)
        self.Y_mean, self.Y_std    = np.mean(Y, axis=0), np.std(Y, axis=0)
        self.X_norm, self.Y_norm   = (X-self.X_mean)/self.X_std, (Y-self.Y_mean)/self.Y_std
        # print(self.X_norm)
        self.x_c                   = (x_centre - self.X_mean)/self.X_std

        # GP variable definitions
        Qp                        = np.zeros((self.nx_dim,self.nx_dim))
        cp                        = np.zeros((self.nx_dim))
        dp                        = 0
        multi_start               = 3
        n_par                     = int(self.nx_dim*(self.nx_dim+1)/2 + self.nx_dim + 1)
        nx_dim                    = self.nx_dim
        self.Mat_n                = int(nx_dim*(nx_dim+1)/2)
        
        pOpt           = np.zeros((n_par))
        localsol       = [0.]*multi_start                        
        localval       = np.zeros((multi_start))   
        # Initialize Sobol generator
        sobol = qmc.Sobol(d=n_par, scramble=False)

        # Generate Sobol sequence
        multi_startvec = sobol.random(n=multi_start)


        # multi_startvec = sobol_seq.i4_sobol_generate(n_par, multi_start)
        # multi_startvec = np.random.uniform(0,1, size=(n_par, multi_start))

        # print(multi_startvec)
        lb, ub         = 0, 1
        bounds         = np.array([[-10,10] for i in range(n_par)])
        
        # carry out optimization
        for j in range(multi_start):
            #print('multi_start hyper parameter optimization iteration = ',j,'  input = ',i)
            hyp_init    = lb + (ub-lb)*multi_startvec[j,:]
            # --- hyper-parameter optimization --- #
            res = minimize(self.LS, hyp_init, args=(self.X_norm,self.Y_norm),method='SLSQP',
                           bounds=bounds)
            localsol[j] = res.x
            localval[j] = res.fun

        # --- choosing best solution --- #
        minindex = np.argmin(localval)
        pOpt     = localsol[minindex]
        
        Mat_n = self.Mat_n
        # assign parameters
        Qp = np.zeros((nx_dim,nx_dim))
        # constructing Q 
        symmVect      = pOpt[:Mat_n]
        indices       = np.tril_indices(nx_dim)
        Qp[indices]   = symmVect 
        Qp.T[indices] = symmVect 
        Qp            = Qp@Qp.T
        # internal parameters
        self.Qp       = Qp + np.eye(nx_dim)*np.matrix.trace(Qp)/nx_dim
        self.cp       = pOpt[Mat_n:Mat_n+nx_dim]
        self.dp       = pOpt[Mat_n+nx_dim:Mat_n+nx_dim+1]
        

    #########################
    # --- Least Squares --- #
    #########################

    def LS(self, pOpt, Xdat, Ydat):
        '''
        Predicts Ydat based on dataset Xdat
        '''
        nx_dim = self.nx_dim
        Mat_n  = self.Mat_n
        
        Qp_ = np.zeros((nx_dim,nx_dim))
        cp_ = pOpt[Mat_n:Mat_n+nx_dim]
        dp_ = pOpt[Mat_n+nx_dim:Mat_n+nx_dim+1]
        # constructing Q 
        symmVect       = pOpt[:Mat_n]
        indices        = np.tril_indices(nx_dim)
        Qp_[indices]   = symmVect 
        Qp_.T[indices] = symmVect 
        Qp_            = Qp_@Qp_.T
        Qp_            = Qp_ + np.eye(nx_dim)*np.matrix.trace(Qp_)/nx_dim
        # penalty on objective function value
        Yw             = Ydat + np.abs(np.min(Ydat))
        # distance penalty
        X_d            = np.sum((Xdat-self.x_c)**2, axis=(1))# X_d, Yw (points,dim)
        X_d            = X_d.reshape(Yw.shape)
        
        # Y prediction
        Y_pred = self.predict_dataset(Xdat, Qp_,cp_,dp_)
        Y_pred = Y_pred.reshape(Ydat.shape)
        
        
        # least-squares == notice the weight (Yw+1e-2+X_d) and regularization
        LS_f   = np.sum((Y_pred-Ydat)**2/(Yw+1e-4+X_d), axis=(0)) + np.sum(pOpt**2)
        #LS_f   = np.sum((Y_pred-Ydat)**2, axis=(0)) + np.sum(pOpt**2)

        return LS_f[0]            

    ######################
    # --- Prediction --- #
    ######################

    def predict_dataset(self, Xdat, Qp,cp,dp):
        '''
        Predicts Ydat based on dataset Xdat
        '''
        Ypred = np.sum(Xdat * np.dot(Xdat,Qp), axis=(1)) + cp@Xdat.T + dp

        return Ypred
            
    ######################
    # --- Prediction --- #
    ######################

    def predict_YNonNorm(self, xnew):
        '''
        redicts y based on xnew
        '''
        xnew = xnew.reshape(1,self.nx_dim)
        
        ypred = np.sum(xnew * np.dot(xnew,self.Qp), axis=(1)) + self.cp@xnew.T + self.dp

        return ypred*self.Y_std + self.Y_mean
    
    ########################################
    # --- Prediction for optimization --- #
    ########################################
    
    def quad_func_2opt(self, d, x0var):
        '''
        d: vector:  distance from centre x0var
        '''
        x_trial = x0var+d
        x_trial = x_trial.reshape(-1,1)
    
        return x_trial.T@self.Qp@x_trial + self.cp@x_trial + self.dp
    
    ##########################################
    # --- optimising the quadratic model --- #
    ##########################################
    
    def opt_quadratic_model(self, x04opt, r_t):
        '''
        a,b,c: parameters estimated for the quadratic model
        x0var: initial point: last
        '''
        x04opt_norm = (x04opt - self.X_mean)/self.X_std
        
        # minimising quadratic model
        res = minimize(self.quad_func_2opt, args=(x04opt_norm), x0=np.zeros(self.nx_dim), 
                       method='SLSQP', bounds=([[-r_t,r_t]]*self.nx_dim))
        # Note: bounds are added: nonlinear trust region is handled poorly by SLSQP
    
        # retrieving solution
        d_sol      = res.x
        x_opt_norm = x04opt_norm + d_sol
        # un normalize
        x_         = x_opt_norm*self.X_std + self.X_mean
        return x_

#########################
# --- Random search --- #
#########################

# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float 
def Random_search_Qopt(f, n_p, bounds_rs, iter_rs, has_x0=False):
    '''
    This function is a naive optimization routine that randomly samples the 
    allowed space and returns the best value.
    '''

    # arrays to store sampled points
    localx   = np.zeros((iter_rs,n_p))  # points sampled
    localval = np.zeros((iter_rs))      # function values sampled

    # bounds
    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    if has_x0 == True:

        for sample_i in range(iter_rs):
            x_trial = np.random.uniform(0, 1, n_p) + f.x0[0].flatten() # sampling
            localx[sample_i,:] = x_trial
            localval[sample_i] = f.fun_test(x_trial) # f

    else:

        for sample_i in range(iter_rs):
            x_trial = np.random.uniform(0, 1, n_p)*bounds_range + bounds_bias # sampling
            localx[sample_i,:] = x_trial
            localval[sample_i] = f.fun_test(x_trial) # f

    return localval,localx



################################
# --- TR and center update --- #
################################

def update_tr_center(quad_M, x_best, f_best, f_list_i, iter_i, n_rs,
                     x_list_i, g_r, g_i, r):
    '''
    
    '''
    # for simplicity we will re-evaluate functions
    # evaluating function
    f_new = f_list_i[n_rs + iter_i]
    f_old = f_list_i[n_rs + iter_i-1]
    xnew  = x_list_i[n_rs + iter_i,:]
    xold  = x_list_i[n_rs + iter_i-1,:]
    
    # if new point is worse than old point
    if f_new >= f_best:
        return x_best, r*g_r, f_best

    # evaluating model
    xnew_norm = (xnew - quad_M.X_mean)/quad_M.X_std
    xold_norm = (xold - quad_M.X_mean)/quad_M.X_std
    
    m_new = quad_M.predict_YNonNorm(xnew_norm)
    m_old = quad_M.predict_YNonNorm(xold_norm)

    rho = (f_old - f_new)/(m_old - m_new)
    if rho > 0.5:
        return xnew, r*g_i, f_new
    else:
        return xnew, r*g_r, f_new

##################################################
# --- Local search with quadratic surrogate  --- #
##################################################
from tqdm import tqdm
# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float
def LS_QM_v2(f, x_dim, bounds, iter_tot, has_x0 = False):
    '''
    This function is an optimization routine following a local random search
    '''
    r_t  = 0.5                            # function on bounds_ss 

    # store f as Test_function object
    o = f

    # read f as function
    f = f.fun_test

    gamma_r = 0.5
    gamma_i = 2.

    # array for estimating models
    f_list  = np.ones((iter_tot))*1e13
    x_list  = np.zeros((iter_tot,x_dim))

    if has_x0 == True:

        # supply starting point
        x_list[0,:] = o.x0[0].flatten()
        f_list[0] = f(x_list[0,:])

        # no of evals to build quadratic model
        n_rs = int(max(x_dim+1,iter_tot*.05)) - 1

        # remaining evals
        n_i = iter_tot - n_rs

        # create sampling around starting point and add to list
        f_list[1:n_rs+1], x_list[1:n_rs+1,:] = Random_search_Qopt(o, x_dim, bounds, n_rs, has_x0=True)   

    else:

        # iterations to find good starting point (and also no of points to build the model around)
        n_rs = int(max(x_dim+1,iter_tot*.05))

        # remaining evals
        n_i  = iter_tot - n_rs

        # evaluate first points
        f_list[:n_rs], x_list[:n_rs,:] = Random_search_Qopt(o, x_dim, bounds, n_rs)

    # find best point
    minindex                       = np.argmin(f_list)
    f_best                         = f_list[minindex]
    x_best                         = x_list[minindex,:]

    print('im here')
    # === Estimate quadratic model === #
    LSQM_m = Quad_Reg_model(x_list[0:n_rs,:], f_list[0:n_rs].reshape(-1,1), x_best)
    # build optimization routine inside Quad_Reg_model, remember to de-normalize
    iter_i_ = range(n_i)
    for iter_i in tqdm(iter_i_):
    # # === main loop === #
    # for iter_i in range(n_i):
        # print('iter ',iter_i,'/',n_i,' radius = ',r_t)
        # minimise the surrogate model
        x_trial = LSQM_m.opt_quadratic_model(x_best, r_t)
        # evaluate function
        f_trial = f(x_trial)
        # add new points to trainig set
        x_list[n_rs + iter_i :n_rs + iter_i+1,:] = x_trial
        f_list[n_rs + iter_i :n_rs + iter_i+1]   = f_trial

        # update trust region and center point
        # MAKE SURE TO PASS THE BEST POINT AND BEST FUNCTION VALUE SO FAR, NOT THE PREVIOUS TO COMPARE
        x_best, r_t, f_best = update_tr_center(LSQM_m, x_best, f_best, f_list, 
                                             iter_i, n_rs, x_list, gamma_r, 
                                             gamma_i, r_t) 
        r_t = max(r_t,10e-5)

        # === re-Estimate quadratic model === #
        LSQM_m = Quad_Reg_model(x_list[:n_rs+iter_i+1,:], 
                                f_list[:n_rs+iter_i+1].reshape(-1,1), 
                                x_best)
        
        # check that f_best corresponds to x_best otherwise there is something strange
        # check
        minindex_ = np.argmin(f_list)
        f_b_      = f_list[minindex_]
        x_b_      = x_list[minindex_,:]

        # re build quadratic model
    # print('x_list = ',x_list)
    # print('f_list = ',f_list)
    return x_best, f_best, None, None #, f_list, x_list























