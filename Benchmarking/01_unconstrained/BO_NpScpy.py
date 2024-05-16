# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 21:14:06 2023

@author: adelrioc
"""

# import libraries
import numpy as np
import time
import numpy.random as rnd
from scipy.spatial.distance import cdist
import sobol_seq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm

class GP_model:
    
    ###########################
    # --- initializing GP --- #
    ###########################    
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True):
        
        # print('GP_model: __init__: Receiving and standardizing X, Y')
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
        
        # print('GP_model: __init__: Determining Hyperparameters in a loop that repeatedly calls negative_loglikelihood which calls Cov_mat')
        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()
        # print('GP_model: __init__: Hyperparameters determined')
    
    #############################
    # --- Covariance Matrix --- #
    #############################    
    
    def Cov_mat(self, kernel, X_norm, W, sf2):

        # print('Using Cov_mat to return covariance matrix from GP_model')
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

        # print('GP_model: calc_cov_sample to return covariance matrix')

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

        # print('Using negative_loglikelihood from GP_model to return NLL')
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

        # print('GP_model: determine_hyperparameters to return hypopt, invKopt. The multistart for hyperparameter determination is set to 2. NEEDS TO BE PUT BACK TO 10')
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

        # print('GP_model: GP_inference_np to return mean_sample, var_sample')
        '''
        --- description ---
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
            mean[i] = np.matmul(np.matmul(k.T,invK),Ysample[:,i])
            var[i]  = max(0, sf2opt - np.matmul(np.matmul(k.T,invK),k)) # numerical error
            #var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

        # --- compute un-normalized mean --- #    
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2
        
        if var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]
        
#################################
# --- Bayesian Optimization --- #
#################################

class GP_optimizer:
    
    ###########################
    # --- initializing GP --- #
    ###########################  
    
    def __init__(
            self, 
            obj_f, 
            data, 
            iter_opt, 
            bounds, 
            # multi_opt=10,
            multi_opt=2, 
            # multi_hyper=10,
            multi_hyper=2, 
            store_data=False, 
            EI_bool=True
            ):
        
        # print('GP_optimizer: __init__')
        '''
        multi_hyper: number of multi-starts for model fitting (hyperparameter estimation)
        data = ['int', bound_list=[[0,10],[-5,5],[3,8]], samples_number] <=> d = ['int', np.array([[-12, 8]]), 3]
        data = ['data0', data=[Xtrain,ytrain]]
        Note 1: in UCB_obj_f we define the objective as only the mean or with the variance
        '''
        # print('- Note: GP_optimizer works for unconstrained one output GP optimization')
        
        # print('GP_optimizer: __init__: data: ' + str(data) + ' store_data: '+str(store_data))

        # GP variable definitions
        self.obj_f, self.data, self.store_data = obj_f, data, store_data
        self.multi_hyper, self.multi_opt       = multi_hyper, multi_opt
        self.iter_opt, self.bounds             = iter_opt, bounds
        self.EI_bool                           = EI_bool
        # data creating
        self.Xtrain, self.ytrain               = self.data_handling()
        self.ndim, self.ndat                   = self.Xtrain.shape[1], self.Xtrain.shape[0]        
     
    #########################
    # --- training data --- #
    #########################
        
    def data_handling(self):

        # print('GP_Optimizer: data_handling')

        '''
        --- description ---
        '''
        data = self.data
        
        # Training data
        if data[0]=='int':
            print('- No preliminar data supplied, computing data by sobol sequence')
            Xtrain, ytrain = self.compute_data(data)
            return Xtrain, ytrain

        elif data[0]=='data0':
            print('- Training data has been suplied')
            # Xtrain = data[1][0]
            # ytrain = data[1][1]
            Xtrain = data[1]
            ytrain = np.array([self.obj_f(point) for point in Xtrain]).reshape((Xtrain.shape[0],1)) #!only for one-dimensional output
            return Xtrain, ytrain

        else:
            print('- error, data ragument ',data,' is of wrong type; can be int or ')
            return None 
    
    ##########################
    # --- computing data --- #
    ##########################
    
    def compute_data(self, data):

        # print('GP_optimizer: compute_data')
        '''
        --- description ---
        '''
        # internal variable calls
        obj_f, ndata = self.obj_f, data[2]
        ndim         = data[1].shape[0]
        x_max, x_min = data[1][:,1], data[1][:,0]
        # print(x_max, x_min)
        
        # computing data
        fx     = np.zeros(ndata) # objective function values at position x are initialized as zeros
        xsmpl  = sobol_seq.i4_sobol_generate(ndim, ndata) # xsmpl.shape = (ndat,ndim)
        
        Xtrain = np.zeros((ndata, ndim))
        # computing Xtrain
        for i in range(ndata):
            xdat        = x_min + xsmpl[i,:]*(x_max-x_min)
            Xtrain[i,:] = xdat

        for i in range(ndata):
            fx[i] = obj_f(Xtrain[i,:]) # here the argument x in the test_functions is handled. It is called as Xtrain[i,:]

        # not meant for multi-output
        ytrain = fx.reshape(ndata,1)
           
        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)

        print('this is how the computed data looks like: ')
        print(Xtrain.reshape(ndata,ndim, order='F'), ytrain.reshape(ndata,1))
        raise
        
        return Xtrain.reshape(ndata,ndim, order='F'), ytrain.reshape(ndata,1)

    ##############################
    # --- GP as obj function --- #
    ##############################
    
    def UCB_obj_f(self, x, GP, obj=2):

        # print('GP_optimizer: UCB_obj_f')

        '''
        define exploration - exploitation strategy
        '''
        # internal variable calls
        if obj == 1:
            obj_f = GP.GP_inference_np(x)
            return obj_f[0]
        
        elif obj == 2:
            obj_f = GP.GP_inference_np(x)
            return obj_f[0] - 3*np.sqrt(obj_f[1]) 
        
        else:
            print('error, objective for GP not specified')

    #############################################
    # --- Expected improvement obj function --- #
    #############################################
    
    def EI_obj_f(self, x, GP, obj=2):

        # print('GP_optimizer: EI_obj_f')

        '''
        define exploration - explotation strategy
        (mu(x)-f(x)-xi)*Phi(z)+sigma(x)phi(z)
        Z = (mu(x)-f(x)-xi)/sigma(x)
        '''
        # function definitions
        xi     = 0.1
        obj_f  = GP.GP_inference_np(x)   # GP evaluation
        if obj_f[1] <= 1e-7: # variance check if I am evaulating too close to a known
            return 0.0
        else:
            f_plus = self.f_plus_EI          # best function value so far
            temp   = f_plus - obj_f[0] + xi
            Z_x    = temp/obj_f[1]           # standardized value for normal
            EI_f   = temp * norm.cdf(Z_x) + obj_f[1] * norm.pdf(Z_x)        
            return -EI_f               
            
    ########################################
    # --- Creating storing data arrays --- #
    ########################################   
    
    def create_data_arrays(self): 

        # print('GP_optimizer: create_data_arrays')

        '''
        --- description ---
        '''
        # internal variable calls
        store_data, iter_opt = self.store_data, self.iter_opt
        ndim, bounds         = self.ndim, self.bounds
        
        if store_data == True:
            print('- remember: store data only available for 1-dimension')
            ntest = 100 # number of samples for trajectory
            Xtest_l = np.zeros((ntest,ndim))                # only 1D supported
            Xtest_l = np.linspace(bounds[0][0], bounds[0][1], num=ntest).reshape(ntest,ndim, order='F')
            ymean_l = np.zeros((iter_opt+1,ntest))
            ystd_l  = np.zeros((iter_opt+1,ntest))
            return Xtest_l, ymean_l, ystd_l
        
        elif store_data == False:
            return 'NA', 'NA', 'NA'
        
    ########################
    # --- Storing data --- #
    ########################   
    
    def add_data(self, Xtest_l, ymean_l, ystd_l, i_opt, GP_m):

        # print('GP_optimizer: add_data')

        '''
        --- description ---
        '''     
        store_data =  self.store_data
        
        if store_data == True:
            # internal variable calls
            n_test               = Xtest_l.shape[0]
            EI_obj_f             = self.EI_obj_f

            if self.EI_bool==True: # if expected improvement
                # -- loop to compute GP prediction -- #
                for ii in range(n_test):
                    m_ii              = EI_obj_f(Xtest_l[ii,:], GP_m)
                    ymean_l[i_opt,ii] = m_ii       
                return ymean_l, 'NA'                

            else:                  # if UCB
                # -- loop to compute GP prediction -- #
                for ii in range(n_test):
                    m_ii, std_ii      = GP_m.GP_inference_np(Xtest_l[ii,:])
                    ymean_l[i_opt,ii] = m_ii 
                    ystd_l[i_opt,ii]  = std_ii        
                return ymean_l, ystd_l
        
        elif store_data == False:
            return 'NA', 'NA'
    
    ############################
    # --- Optimization alg --- #
    ############################   
    
    def optimization_routine(self):

        # print('GP_optimizer: optimisation routine -------------------------------------------------------------------------')
        '''
        --- description ---
        store_data as boolean
        add_data dependen on store_data. If store_data == False (as initialized), add_data does nothing
        iter_opt is the number of overal steps (= length of trajectory)
        multi_hyper = number of multi-starts for GP fitting
        mutli_opt = number of multi-start for acquisition function optimization
        '''
        # internal variable calls
        store_data, Xtrain, ytrain   = self.store_data, self.Xtrain, self.ytrain
        ndim, ndat, obj_f, multi_opt = self.ndim, self.ndat, self.obj_f, self.multi_opt
        multi_hyper, iter_opt        = self.multi_hyper, self.iter_opt
        add_data, bounds   = self.add_data, self.bounds
        create_data_arrays           = self.create_data_arrays
        if self.EI_bool==True:
            GP_obj_f       = self.EI_obj_f
            self.f_plus_EI = np.min(ytrain) # !!np.max(ytrain)
        else:
            GP_obj_f = self.UCB_obj_f
        
        # building the first GP model
        # print('GP_optimizer: optimisation routine: building the first GP model')
        GP_m   = GP_model(Xtrain, ytrain, 'RBF', multi_hyper=multi_hyper, var_out=True)
        # print('GP_optimizer: optimisation routine: After building first GP model we set X_opt to Xtrain and y_opt to ytrain')
        X_opt  = Xtrain
        y_opt  = ytrain
        
        # --- storing data --- #
        # print('GP_optimizer: optimisation routine: now we create data arrays for Xtest_l, ymean_l and ystd_l')
        Xtest_l, ymean_l, ystd_l = create_data_arrays()        
        
        # --- optimization loop --- #
        # print('GP_optimizer: optimisation routine: now we start the optimisation loop')
        options          = {'disp':False,'maxiter':10000}  # solver options
        lb, ub           = bounds[:,0], bounds[:,1]
        # print('GP_optimizer: optimisation routine: the startvectors for multistart are created using sobol sampling')
        multi_startvec   = sobol_seq.i4_sobol_generate(ndim,multi_opt)
        
        # optimization -- iterations
        for i_opt in range(iter_opt-ndat):

            # --- storing data --- #
            ymean_l, ystd_l = add_data(Xtest_l, ymean_l, ystd_l, i_opt, GP_m)
            
            localsol = [0.]*multi_opt           # x-value entries for multistart are re-set to 0
            localval = np.zeros((multi_opt))    # GP(x) entries for multistart are re-set to 0
            
            # optimization -- multistart
            for j in range(multi_opt):
                # print('GP_optimizer: optimisation routine: optimisation iteration ' + str(i_opt+1) + ' and multistart nr.: ' + str(j+1) + '/' + str(multi_opt) + ' NEEDS TO BE SET BACK TO 10')
                x_init    = lb + (ub-lb)*multi_startvec[j,:]
                
                # print('GP_optimizer: optimisation routine: optimisation iteration ' + str(i_opt+1) + ' and multistart nr.: ' + str(j+1) + ': objective function gets minimized')
                # print('GP_optimizer: optimisation routine: optimisation iteration ' + str(i_opt+1) + ' and multistart nr.: ' + str(j+1) + ': This minimization is a loop that ends when the tolerance is reached. Therefore, the length of this loop varies for each start.')
                # GP optimization
                res = minimize(GP_obj_f, x_init, args=(GP_m), method='SLSQP',
                               options=options, bounds=bounds, tol=1e-12)
                # print('GP_optimizer: optimisation routine: optimisation iteration ' + str(i_opt+1) + ' and multistart nr.: ' + str(j+1) + ': done with minimization of objective function')
                localsol[j] = res.x
                if res.success == True:
                    localval[j] = res.fun
                else:
                    localval[j] = np.inf

            # print('GP_optimizer: optimisation routine: now all multistarts for iteration ' + str(i_opt+1) + ' are done and the results are stored in localval')
            if np.min(localval) == np.inf:
                print('warning, no feasible solution found')

            # print('GP_optimizer: optimisation routine: xnew is the x belonging to the lowest value found in localval for iteration ' + str(i_opt+1))
            minindex    = np.argmin(localval) # choosing best solution
            xnew        = localsol[minindex]  # selecting best solution

            xnew   = np.array([xnew]).reshape(1,ndim)
            ynew   = obj_f(xnew)
            # adding new point to sample
            # print('GP_optimizer: optimisation routine: xnew and ynew are added to X_opt and y_opt')
            X_opt  = np.vstack((X_opt,xnew))
            y_opt  = np.vstack((y_opt,ynew))
            # re-training GP
            # print('GP_optimizer: optimisation routine: after iteration ' + str(i_opt+1) + ' the GP is now re-trained based on the updated X_opt and y_opt')
            GP_m   = GP_model(X_opt, y_opt, 'RBF', multi_hyper=10, var_out=True)

            # print('GP_optimizer: optimisation routine: checking for expected improvement in the end of iteration ' + str(i_opt+1))
            if self.EI_bool==True:
                self.f_plus_EI = np.min(ytrain) # !!np.max(ytrain)


        # print('GP_optimizer: after all optimisation iterations are done, add_data is called again ???')    
        # --- storing data --- #
        ymean_l, ystd_l = add_data(Xtest_l, ymean_l, ystd_l, i_opt+1, GP_m)
        
        if store_data == False:
            return X_opt, y_opt
        elif store_data == True:
            return X_opt, y_opt, Xtest_l, ymean_l, ystd_l
        
#################################   
# --- Bayesian Optimization --- #
#################################

def BO_np_scipy(f, x_dim, bounds, iter_tot, has_x0=False):

    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    if has_x0 == True:

        d_     = ['data0', f.init_points[0], 10] #TODO this needs to be automatized to allow x0 for more starting points

    else:
        # n_rs   = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point # old
        n_rs = int(max(x_dim+1,iter_tot*.05))
        d_     = ['int', bounds, n_rs]

    GP_opt = GP_optimizer(
        f.fun_test, 
        d_, 
        iter_tot, 
        bounds,
        store_data=False, 
        EI_bool=False
        )

    X_opt, y_opt = GP_opt.optimization_routine()
    i_best       = np.argmin(y_opt)
    x_best       = X_opt[i_best]
    y_opt       = y_opt[i_best]

    team_names = ['20','21']
    cids = ['01234567']
    print('------------------------------------FINISH-----------------------------------------')
    return x_best, y_opt, team_names, cids







    