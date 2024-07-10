import time
import copy
import random
import numpy as np
import numpy.random as rnd
from scipy.spatial.distance import cdist
# !pip install sobol_seq
import sobol_seq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Ellipse
from scipy.optimize import differential_evolution
# !pip install tqdm
from tqdm import tqdm
from pylab import grid
from matplotlib.colors import LogNorm
from utils import *



class GP_model: # _2

    ###########################
    # --- initializing GP --- #
    ###########################
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True):

        # GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim   = X.shape[0], X.shape[1]
        self.ny_dim                 = Y.shape[1]
        self.multi_hyper            = multi_hyper
        self.var_out                = var_out

        # normalize data
        self.X_mean, self.X_std     = np.mean(X, axis=0), np.std(X, axis=0)
        self.Y_mean, self.Y_std     = np.mean(Y, axis=0), np.std(Y, axis=0)
        self.X_norm, self.Y_norm    = (X-self.X_mean)/self.X_std, (Y-self.Y_mean)/self.Y_std

        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()

    #############################
    # --- Covariance Matrix --- #
    #############################

    def Cov_mat(self, kernel, X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        --- decription ---
        '''

        if kernel == 'RBF':
            dist       = cdist(X_norm, X_norm, 'seuclidean', V=W)**2
            cov_matrix = sf2*np.exp(-0.5*dist)
            return cov_matrix
            # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        else:
            print('ERROR no kernel with name ', kernel)

    ################################
    # --- Covariance of sample --- #
    ################################

    def calc_cov_sample(self,xnorm,Xnorm,ell,sf2):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        --- decription ---
        '''
        # internal parameters
        nx_dim = self.nx_dim

        dist = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)**2
        cov_matrix = sf2 * np.exp(-.5*dist)

        return cov_matrix

    ###################################
    # --- negative log likelihood --- #
    ###################################

    def negative_loglikelihood(self, hyper, X, Y):
        '''
        --- decription ---
        '''
        # internal parameters
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel          = self.kernel

        W               = np.exp(2*hyper[:nx_dim])   # W <=> 1/lambda
        sf2             = np.exp(2*hyper[nx_dim])    # variance of the signal
        sn2             = np.exp(2*hyper[nx_dim+1])  # variance of noise

        K       = self.Cov_mat(kernel, X, W, sf2)  # (nxn) covariance matrix (noise free)
        K       = K + (sn2 + 1e-8)*np.eye(n_point) # (nxn) covariance matrix
        K       = (K + K.T)*0.5                    # ensure K is simetric
        L       = np.linalg.cholesky(K)            # do a cholesky decomposition
        logdetK = 2 * np.sum(np.log(np.diag(L)))   # calculate the log of the determinant of K the 2* is due to the fact that L^2 = K
        invLY   = np.linalg.solve(L,Y)             # obtain L^{-1}*Y
        alpha   = np.linalg.solve(L.T,invLY)       # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL     = np.dot(Y.T,alpha) + logdetK      # construct the NLL

        return NLL

    ############################################################
    # --- Minimizing the NLL (hyperparameter optimization) --- #
    ############################################################

    def determine_hyperparameters(self):
        '''
        --- decription ---
        Notice we construct one GP for each output
        '''
        # internal parameters
        X_norm, Y_norm  = self.X_norm, self.Y_norm
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel, ny_dim  = self.kernel, self.ny_dim
        Cov_mat         = self.Cov_mat


        lb               = np.array([-4.]*(nx_dim+1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub               = np.array([4.]*(nx_dim+1) + [ -2.])   # lb on parameters (this is inside the exponential)
        bounds           = np.hstack((lb.reshape(nx_dim+2,1),
                                      ub.reshape(nx_dim+2,1)))
        multi_start      = self.multi_hyper                   # multistart on hyperparameter optimization
        multi_startvec   = sobol_seq.i4_sobol_generate(nx_dim + 2,multi_start)

        options  = {'disp':False,'maxiter':10000}          # solver options
        hypopt   = np.zeros((nx_dim+2, ny_dim))            # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol = [0.]*multi_start                        # values for multistart
        localval = np.zeros((multi_start))                 # variables for multistart

        invKopt = []
        # --- loop over outputs (GPs) --- #
        for i in range(ny_dim):
            # --- multistart loop --- #
            for j in range(multi_start):
                #print('multi_start hyper parameter optimization iteration = ',j,'  input = ',i)
                hyp_init    = lb + (ub-lb)*multi_startvec[j,:]
                # --- hyper-parameter optimization --- #
                res = minimize(self.negative_loglikelihood,hyp_init,args=(X_norm,Y_norm[:,i])\
                               ,method='SLSQP',options=options,bounds=bounds,tol=1e-12)
                localsol[j] = res.x
                localval[j] = res.fun

            # --- choosing best solution --- #
            minindex    = np.argmin(localval)
            hypopt[:,i] = localsol[minindex]
            ellopt      = np.exp(2.*hypopt[:nx_dim,i])
            sf2opt      = np.exp(2.*hypopt[nx_dim,i])
            sn2opt      = np.exp(2.*hypopt[nx_dim+1,i]) + 1e-8

            # --- constructing optimal K --- #
            Kopt        = Cov_mat(kernel, X_norm, ellopt, sf2opt) + sn2opt*np.eye(n_point)
            # --- inverting K --- #
            invKopt     += [np.linalg.solve(Kopt,np.eye(n_point))]

        return hypopt, invKopt

    ########################
    # --- GP inference --- #
    ########################

    def GP_inference_np(self, x):
        '''
        --- decription ---
        '''
        nx_dim                   = self.nx_dim
        kernel, ny_dim           = self.kernel, self.ny_dim
        hypopt, Cov_mat          = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        calc_cov_sample          = self.calc_cov_sample
        invKsample               = self.invKopt
        Xsample, Ysample         = self.X_norm, self.Y_norm
        var_out                  = self.var_out
        # Sigma_w                = self.Sigma_w (if input noise)

        xnorm = (x - meanX)/stdX
        mean  = np.zeros(ny_dim)
        var   = np.zeros(ny_dim)
        # --- Loop over each output (GP) --- #
        for i in range(ny_dim):
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = np.exp(2*hyper[:nx_dim]), np.exp(2*hyper[nx_dim])

            # --- determine covariance of each output --- #
            k       = calc_cov_sample(xnorm,Xsample,ellopt,sf2opt)
            #print(' mean', np.matmul(np.matmul(k.T,invK),Ysample[:,i]))
            #print(' var',max(0, sf2opt - np.matmul(np.matmul(k.T,invK),k)))
            #stop
            mean[i] = np.matmul(np.matmul(k.T,invK),Ysample[:,i])[0]
            var[i]  = max(0, sf2opt - np.matmul(np.matmul(k.T,invK),k)) # numerical error
            #var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

        # --- compute un-normalized mean --- #
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2

        if var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]
        

class BO:

    ###################################
    # --- initializing ITR_GP_RTO --- #
    ###################################

    def __init__(self, obj_system, cons_system, x0,
                 Delta0, Delta_max, eta0, eta1, gamma_red, gamma_incr,
                 n_iter, data, bounds, multi_opt=5, multi_hyper=10, TR_scaling=True,
                 store_data=True, inner_TR=False, safe_con=False, safe_TR=False,
                 TR_method=True):
        '''
        data = ['int', bound_list=[[0,10],[-5,5],[3,8]], samples_number] <=> d = ['int', np.array([[-12, 8]]), 3]
        data = ['data0', Xtrain]

        Note 1: remember the data collected

        '''

        # intrnal variable definitions
        self.n_iter                                   = n_iter
        self.multi_opt, self.multi_hyper              = multi_opt, multi_hyper
        self.store_data, self.data, self.bounds       = store_data, data, bounds
        self.x0, self.obj_system                      = x0, obj_system
        self.cons_system                              = cons_system
        self.TR_scaling                               = TR_scaling
        # TR adjustment variables
        self.Delta_max, self.eta0, self.eta1          = Delta_max, eta0, eta1
        self.gamma_red, self.gamma_incr               = gamma_red, gamma_incr
        self.safe_con, self.safe_TR, self.TR_method   = safe_con, safe_TR, TR_method
        # tracking non-accepted samples
        self.violation_list, self.suboptimal_list     = [],[]
        
        # other definitions
        self.ng     = len(self.cons_system)
        # self.ng     = 1 #TODO automatize
        self.Delta0 = Delta0
        # data creating
        self.Xtrain, self.ytrain               = self.data_handling()
        self.ndim, self.ndat                   = self.Xtrain.shape[1], self.Xtrain.shape[0]
        # alerts
        print('note: remember constraints are set as positive, so they should be set as -g(x)')


    #########################
    # --- training data --- #
    #########################

    def data_handling(self):
        '''
        --- description ---
        '''
        data = self.data

        # Training data
        if data[0]=='int':
            print('- No preliminar data supplied, computing data by sobol sequence')
            Xtrain         = np.array([])
            Xtrain, ytrain = self.compute_data(data, Xtrain)
            return Xtrain, ytrain

        elif data[0]=='data0':
            print('- preliminar data supplied, computing objective and constraint values')
            Xtrain         = data[1]
            Xtrain, ytrain = self.compute_data(data, Xtrain)
            return Xtrain, ytrain

        else:
            print('- error, data ragument ',data,' is of wrong type; can be int or ')
            return None

    ##########################
    # --- computing data --- #
    ##########################

    def compute_data(self, data, Xtrain):
        '''
        --- description ---
        '''
        # internal variable calls
        data[1], cons_system = np.array(data[1]), self.cons_system
        ng, obj_system       = self.ng, self.obj_system


        if Xtrain.shape == (0,): # no data suplied
            # data arrays
            ndim          = data[1].shape[0]
            x_max, x_min  = data[1][:,1], data[1][:,0]
            ndata         = data[2]
            Xtrain        = np.zeros((ndata, ndim))
            ytrain        = np.zeros((ng+1, ndata))
            funcs_system  = [obj_system] + cons_system

            for ii in range(ng+1):
                # computing data
                fx     = np.zeros(ndata)
                xsmpl  = sobol_seq.i4_sobol_generate(ndim, ndata) # xsmpl.shape = (ndat,ndim)

                # computing Xtrain
                for i in range(ndata):
                    xdat        = x_min + xsmpl[i,:]*(x_max-x_min)
                    Xtrain[i,:] = xdat
                for i in range(ndata):
                    fx[i] = funcs_system[ii](np.array(Xtrain[i,:]))
                # not meant for multi-output
                ytrain[ii,:] = fx

        else: # data suplied
            # data arrays
            ndim          = Xtrain.shape[1]
            ndata         = Xtrain.shape[0]
            ytrain        = np.zeros((ng+1, ndata))
            funcs_system  = [obj_system] + cons_system

            for ii in range(ng+1):
                fx     = np.zeros(ndata)

                for i in range(ndata):
                    fx[i] = funcs_system[ii](np.array(Xtrain[i,:]))


                # not meant for multi-output
                ytrain[ii,:] = fx


        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)

        return Xtrain.reshape(ndata, ndim, order='F'), ytrain.reshape(ng+1, ndata, order='F')

    ##############################
    # --- GP as obj function --- #
    ##############################

    def GP_obj_f(self, d, GP, xk, obj=1):
        '''
        define exploration - explotation strategy
        '''

        # internal variable calls
        if obj == 1:
            obj_f = GP.GP_inference_np(xk+d)
            return obj_f[0]

        elif obj == 2:
            obj_f = GP.GP_inference_np(xk+d)
            return obj_f[0] - np.sqrt(obj_f[1])

        else:
            print('error, objective for GP not specified')

    ####################################
    # --- System constraints check --- #
    ####################################

    def System_Cons_check(self, x):
        '''
        This checks if all the constraints of the system are satisfied
        '''
        # internal calls
        cons_system = self.cons_system


        cons = []
        for con_i in range(len(cons_system)):
            cons.append(cons_system[con_i](x))
        cons      = np.array(cons)
        satisfact = cons>0
        satisfact = satisfact.all()

        return satisfact

    ###################################
    # --- Trust Region Constraint --- #
    ###################################

    def TR_con(self, d):
        '''
        TR constraint
        '''
        # traditional TR constraint
        if self.safe_TR == False:
            return self.Delta0**2 - d@d.T
        # have variance influence
        elif self.safe_TR == True:
            TR_gp  = self.GP_TR.GP_inference_np(self.xk+d)
            ratio_ = 10*np.sqrt(TR_gp[1])/TR_gp[0]
            return self.Delta0**2/(ratio_+1) - d@d.T

    ###################################
    # --- Mismatch Constraint --- #
    ##################################

    def GP_conf(self, xk, GP_con_i, d): # change to GP con
        '''
        mistmatch constraint
        '''
        if self.safe_con == True:
            con_fi = GP_con_i.GP_inference_np(xk+d)
            return con_fi[0] - 3*np.sqrt(con_fi[1]) # - because -g
        else:
            return GP_con_i.GP_inference_np((xk+d).flatten())

    #######################
    # --- TR function --- #
    #######################

    def Adjust_TR(self, Delta0, xk, xnew, GP_obj):
        '''
        Adjusts the TR depending on the rho ratio between xk and xnew
        '''
        Delta_max, eta0, eta1          = self.Delta_max, self.eta0, self.eta1
        gamma_red, gamma_incr          = self.gamma_red, self.gamma_incr
        obj_system                     = self.obj_system

        # --- compute rho --- #
        plant_i     = obj_system(np.array(xk).flatten())
        plant_iplus = obj_system(np.array(xnew).flatten())
        rho         = (plant_i - plant_iplus)/(GP_obj.GP_inference_np(np.array(xk).flatten())[0]-
                                               GP_obj.GP_inference_np(np.array(xnew).flatten())[0] )

        # --- Update TR --- #
        if plant_iplus<plant_i:
            if rho>=eta0:
                if rho>=eta1:
                    Delta0 = min(Delta0*gamma_incr, Delta_max)
                elif rho<eta1:
                    Delta0 = Delta0*gamma_red
                # Note: xk = xnew this is done later in the code
            if rho<eta0:
                #print('rho<eta0 -- backtracking')
                self.suboptimal_list.append(copy.deepcopy(xk))
                xnew   = xk
                Delta0 = Delta0*gamma_red
        else:
            self.suboptimal_list.append(copy.deepcopy(xk))
            xnew   = xk
            Delta0 = Delta0*gamma_red
            #print('plant_iplus<plant_i -- backtracking')

        return Delta0, xnew, xk

    ###################################
    # --- Random Sampling in ball --- #
    ###################################

    def Ball_sampling(self, ndim):
        '''
        This function samples randomly withing a ball of radius self.Delta0
        '''
        u      = np.random.normal(0,1,ndim)  # random sampling in a ball
        norm   = np.sum(u**2)**(0.5)
        r      = random.random()**(1.0/ndim)
        d_init = r*u/norm*self.Delta0*2      # random sampling in a ball

        return d_init

    ########################################
    # --- Constrain Violation and step --- #
    ########################################

    def Step_constraint(self, Delta0, xk, xnew, GP_obj):
        '''
        Calls Adjust_TR which adjusts the trust region and decides on the step size
        depending on constraint violation or the objective similarity
        '''
        Adjust_TR = self.Adjust_TR

        if not self.System_Cons_check(np.array(xnew).flatten()):
            xnew   = xk
            #print('Constraint violated -- backtracking')
            self.violation_list.append(copy.deepcopy(xk))
            return Delta0*self.gamma_red, xnew, xk

        else:
            Delta0, xnew, xk = Adjust_TR(Delta0, xk, xnew, GP_obj)
            return Delta0, xnew, xk

    ########################################
    # --- Constraint GP construction --- #
    ########################################

    def GPs_construction(self, xk, Xtrain, ytrain, ndatplus, H_norm=0.):
        '''
        Constructs a GP for every cosntraint
        '''
        # internal calls
        ndat, multi_hyper, ng, ndim    = self.ndat+ndatplus, self.multi_hyper, self.ng, self.ndim
        GP_conf                        = self.GP_conf

        # --- objective function GP --- #
        GP_obj   = GP_model(Xtrain, ytrain[0,:].reshape(ndat,1), 'RBF',
                            multi_hyper=multi_hyper, var_out=True)
        GP_con   = [0]*ng # Gaussian processes that output mistmatch (functools)
        GP_con_2 = [0]*ng # Constraints for the NLP
        GP_con_f = [0]*(ng+1)

        # adding backoff for safe constraints
        if self.safe_con == True:
            con_var_out = True
        else:
            con_var_out = False
        # adding TR
        if self.safe_TR == True:
            self.GP_TR = GP_obj # this is debatable and could be closest constraint
            self.xk    = xk     # this can be problematic if xk is not the centre for next iter

        # constructiong GPs for constrains and objective
        for igp in range(ng):
            GP_con[igp]     = GP_model(Xtrain, ytrain[igp+1,:].reshape(ndat,1), 'RBF',
                                       multi_hyper=multi_hyper, var_out=con_var_out)
            GP_con_2[igp]   = functools.partial(GP_conf, xk, GP_con[igp]) # partially evaluating a function
            GP_con_f[igp]   = {'type': 'ineq', 'fun': GP_con_2[igp]}
        if self.TR_method == True:
            GP_con_f[igp+1]     = {'type': 'ineq', 'fun': self.TR_con}
        else:
            GP_con_f = GP_con_f[:-1]

        return GP_obj, GP_con_f

    ######################################
    # --- Real-Time Optimization alg --- #
    ######################################

    def RTO_routine(self):
        '''
        --- description ---
        '''
        # internal variable calls
        store_data, Xtrain, ytrain      = self.store_data, self.Xtrain, self.ytrain
        multi_hyper, n_iter             = self.multi_hyper, self.n_iter
        ndim, ndat, multi_opt           = self.ndim, self.ndat, self.multi_opt
        GP_obj_f, bounds, ng            = self.GP_obj_f, self.bounds, self.ng
        multi_opt, Delta0, x0           = self.multi_opt, self.Delta0, self.x0
        obj_system, cons_system         = self.obj_system, self.cons_system
        Adjust_TR, Step_constraint      = self.Adjust_TR, self.Step_constraint
        GP_conf                         = self.GP_conf
        TR_scaling                      = self.TR_scaling
        Ball_sampling                   = self.Ball_sampling
        GPs_construction                = self.GPs_construction

        # variable definitions
        funcs_system  = [obj_system]+ cons_system

        # --- building GP models from existing data --- #
        # evaluating initial point
        xnew    = x0
        Xtrain  = np.vstack((Xtrain,xnew))
        ynew    = np.zeros((1,ng+1))
        for ii in range(ng+1):
                ynew[0,ii] = funcs_system[ii](np.array(xnew[:]).flatten())
        ytrain  = np.hstack((ytrain,ynew.T))
        # --- building GP models for first time --- #
        GP_obj, GP_con_f = GPs_construction(xnew, Xtrain, ytrain, 1)

        # renaming data
        X_opt    = np.copy(Xtrain)
        y_opt    = np.copy(ytrain)

        # --- TR lists --- #
        TR_l    = ['error']*(n_iter+1)
        TR_l[0] = self.Delta0

        # --- rto -- iterations --- #
        options          = {'disp':False,'maxiter':10000}  # solver options
        lb, ub           = bounds[:,0], bounds[:,1]

        for i_rto in tqdm(range(n_iter)):
            # --- optimization -- multistart --- #
            localsol = [0.]*multi_opt           # values for multistart
            localval = np.zeros((multi_opt))    # variables for multistart
            xk       = xnew
            TRb      = (bounds-xk.reshape(ndim,1, order='F'))
            for j in range(multi_opt):
                d_init = Ball_sampling(ndim) # random sampling in a ball
                # GP optimization
                res = minimize(GP_obj_f, d_init, args=(GP_obj, xk), method='SLSQP',
                               jac='3-point',
                               options=options, bounds=(TRb), constraints=GP_con_f)
                localsol[j] = res.x
                #print('res = ',res.status)
                if (res.success == True):
                    localval[j] = res.fun
                else:
                    localval[j] = np.inf
            if np.min(localval) == np.inf:
                print('warming, no feasible solution found')
            # selecting best point
            minindex    = np.argmin(localval) # choosing best solution
            xnew        = localsol[minindex] + xk  # selecting best solution

            # re-evaluate best point (could be done more efficiently - no re-evaluation)
            xnew = np.array([xnew]).reshape(1,ndim)
            ynew = np.zeros((1,ng+1))
            for ii in range(ng+1):
                ynew[0,ii] = funcs_system[ii](np.array(xnew[:]).flatten())

            # adding new point to sample
            X_opt  = np.vstack((X_opt,xnew))
            y_opt  = np.hstack((y_opt,ynew.T))

            # --- Update TR --- #
            self.Delta0, xnew, xk = Step_constraint(self.Delta0, xk, xnew, GP_obj)      # adjust TR

            # --- re-training GP --- #
            GP_obj, GP_con_f = GPs_construction(xnew, X_opt, y_opt, 2+i_rto)

            # --- TR listing --- #
            TR_l[i_rto+1] = self.Delta0

        # --- output data --- #
        return X_opt, y_opt, TR_l, xnew, self.suboptimal_list
    
##########################
#### Constrained BO ######
##########################

# this is just with Trust region switched OFF

def CBO_opt(
        f,
        x_dim, #TODO here the functionality of multiple input dimensions has to be implemented
        bounds,  
        f_eval_, # length of trajectory (objective function evaluation budget)
        i_rep
        ): 

    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''
# comment 1233
    test_fun = f.fun_test
    n = 9 # no of initial points to build model besides starting point.

    if f.func_type == 'WO_f': 
        test_con = [f.WO_con1_test, f.WO_con2_test]
        radius = 0.1
    else: 
        test_con = [f.con_test]
        radius = 0.5

    # Xtrain = f.init_points[i_rep]

    x0 = f.x0[i_rep].flatten()
    Xtrain = random_points_in_circle(n, radius=radius, center=f.x0[i_rep].transpose())
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2
    budget = f_eval_ - samples_number -1 # the initial samples in CBO_TR need to be subtracted

    GP_opt = BO(
        test_fun, 
        test_con,
        x0,
        Delta0,
        Delta_max,
        eta0,
        eta1,
        gamma_red,
        gamma_incr,
        budget, 
        data,
        bounds = np.array(bounds), # re-constructed from google colab
        multi_opt= 10, # TODO: check, whether the other algorithms have these as well. If so, put it outside
        multi_hyper=10, 
        store_data=True,
        TR_method=False, #if False: This is then only constrained BO - not Safe BO
        )

    X_opt, y_opt, TR_l, xnew, backtrck_l = GP_opt.RTO_routine()

    # storing outputs for plotting
    X_opt_plot = X_opt
    i_best       = np.argmin(y_opt[0]) # y_opt[0] are the values of the objective function, the others are from the constraints
    x_best       = X_opt[i_best]
    y_opt       = y_opt[0][i_best]

    team_names = ['20','21']
    cids = ['01234567']
    print('------------------------------------FINISH-----------------------------------------')
    return x_best, y_opt, team_names, cids, X_opt_plot, TR_l, xnew, backtrck_l, samples_number


###################
#### Safe BO ######
###################

# this is just with Trust region switched on

def CBO_TR_opt(
        f,
        x_dim, #TODO here the functionality of multiple input dimensions has to be implemented
        bounds,  
        f_eval_, # length of trajectory (objective function evaluation budget)
        i_rep
        ): 

    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    test_fun = f.fun_test
    n = 9 # no of initial points to build model besides starting point.
    
    if f.func_type == 'WO_f': 
        test_con = [f.WO_con1_test, f.WO_con2_test]
        radius = 0.1
    else: 
        test_con = [f.con_test]
        radius = 0.5

    # Xtrain = f.init_points[i_rep]
    x0 = f.x0[i_rep].flatten()
    Xtrain = random_points_in_circle(n, radius=radius, center=f.x0[i_rep].transpose())
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2
    f_eval_ = f_eval_ - samples_number # the initial samples in CBO_TR need to be subtracted

    GP_opt = BO(  # TODO: Check whether the SafeOpt algorithm has similar hyperparameters and if these must then be packed outside the routine
        test_fun, 
        test_con,
        x0,
        Delta0,
        Delta_max,
        eta0,
        eta1,
        gamma_red,
        gamma_incr,
        f_eval_, 
        data,
        bounds = np.array(bounds), # re-constructed from google colab
        multi_opt= 10, # TODO: check, whether the other algorithms have these as well. If so, put it outside
        multi_hyper=10, 
        store_data=True,
        TR_method=True, #if False: This is then only constrained BO - not Safe BO
        )

    X_opt, y_opt, TR_l, xnew, backtrck_l = GP_opt.RTO_routine()

    # storing outputs for plotting
    X_opt_plot = X_opt
    i_best       = np.argmin(y_opt[0]) # y_opt[0] are the values of the objective function, the others are from the constraints
    x_best       = X_opt[i_best]
    y_opt       = y_opt[0][i_best]

    team_names = ['20','21']
    cids = ['01234567']
    print('------------------------------------FINISH-----------------------------------------')
    return x_best, y_opt, team_names, cids, X_opt_plot, TR_l, xnew, backtrck_l, samples_number