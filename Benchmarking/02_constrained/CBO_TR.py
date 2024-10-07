import time
import random
import numpy as np
import numpy.random as rnd
from scipy.spatial.distance import cdist
import sobol_seq
from scipy.optimize import minimize
from scipy.optimize import broyden1
from scipy import linalg
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Ellipse

class GP_model:

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


        lb               = np.array([-5.]*(nx_dim+1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub               = np.array([5.]*(nx_dim+1) + [ -4.])   # lb on parameters (this is inside the exponential)
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
        this returns the GP inference (prediction of covariance and mean) at position x
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
            var[i]  = max(0., sf2opt - np.matmul(np.matmul(k.T,invK),k)) # numerical error
            #var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

        # --- compute un-normalized mean --- #
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2

        if var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]

class ITR_GP_RTO:

    ###################################
    # --- initializing ITR_GP_RTO --- #
    ###################################

    def __init__(self, obj_system, cons_system, x0,
                 Delta0, Delta_max, eta0, eta1, gamma_red, gamma_incr,
                 n_iter, data, bounds, multi_opt, multi_hyper, store_data=True):
        '''
        n_iter is the number of objective function evaluations, therefore it is the total number of iterations 
        (meaning the length of the trajectory in terms of the benchmarking)
        multi_opt is the number of multistarts to minimize the acquisition function per n_iter
        multi_hyper is the number of miltistarts to fit the GPs to data provided
        data = ['int', bound_list=[[0,10],[-5,5],[3,8]], samples_number] <=> d = ['int', np.array([[-12, 8]]), 3]
        data = ['data0', Xtrain]

        Note 1: remember the data collected

        '''


        # internal variable definitions
        self.n_iter                                   = n_iter
        self.multi_opt, self.multi_hyper              = multi_opt, multi_hyper
        self.store_data, self.data, self.bounds       = store_data, data, bounds
        self.x0, self.obj_system                      = x0, obj_system
        # self.obj_system_init                          = obj_system.init_points
        # self.cons_system                              = cons_system # original     
        self.cons_system                              = [cons_system]
        # TR adjustment variables
        self.Delta_max, self.eta0, self.eta1          = Delta_max, eta0, eta1
        self.gamma_red, self.gamma_incr               = gamma_red, gamma_incr

        # other definitions
        self.TRmat                = np.linalg.inv(np.diag(bounds[:,1]-bounds[:,0]))
        # self.ng                   = len(cons_system)
        self.ng                   = 1 #TODO automatize
        self.Delta0               = Delta0
        # data creating
        self.Xtrain, self.ytrain  = self.data_handling()
        self.ndim, self.ndat      = self.Xtrain.shape[1], self.Xtrain.shape[0]
        print('note: remember constraints are set as positive, so they should be set as -g(x)')


    #########################
    # --- training data --- #
    #########################

    def data_handling(self):
        '''
        --- description ---
            
        self.data is provided as d_ = ['int', bounds, samples_number] 
        samples_number is the number of samples the algorithm is allowed to do to build the initial GPs
        '''

        # print('ITR_GP_RTO: data_handling')
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
        self.data is provided as d_ = ['int', bounds, samples_number] 
        samples_number is the number of samples the algorithm is allowed to do to build the initial GPs
        '''

        # print('ITR_GP_RTO: compute_data')
        # internal variable calls
        data[1], cons_system  = np.array(data[1]), self.cons_system
        ng, obj_system        = self.ng, self.obj_system
        # obj_system_init       = self.obj_system_init


        if Xtrain.shape == (0,): # no data suplied
            print('ITR_GP_RTO: compute_data: no data supplied, sobol sampling')
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
                    # fx[i] = funcs_system[ii](np.array(Xtrain[i]))
                # not meant for multi-output
                ytrain[ii,:] = fx

        else: # data suplied
            # print('ITR_GP_RTO: compute_data: data supplied')
            # data arrays
            ndim          = Xtrain.shape[1]
            ndata         = Xtrain.shape[0]
            ytrain        = np.zeros((ng+1, ndata))
            funcs_system  = [obj_system]+ cons_system

            for ii in range(ng+1):
                fx     = np.zeros(ndata)

                for i in range(ndata):
                    fx[i] = funcs_system[ii](np.array(Xtrain[i,:]))
                # not meant for multi-output
                ytrain[ii,:] = fx

        # print('ITR_GP_RTO: compute_data: saving Xtrain and ytrain')

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
            #print('ITR_GP_RTO: GP_obj_f: obj==1: returning objective function value for xk+d')
            return obj_f[0]

        elif obj == 2:
            obj_f = GP.GP_inference_np(xk+d)
            #print('ITR_GP_RTO: GP_obj_f: obj==2: returning objective function value for xk+d')
            return obj_f[0] - 2*np.sqrt(obj_f[1])

        else:
            print('error, objective for GP not specified')

    ####################################
    # --- System constraints check --- #
    ####################################

    def System_Cons_check(self, x):
        '''
        This checks if all the constraints of the system are satisfied
        '''
        # print('ITR_GP_RTO: System_Cons_check')

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
        it must be Delta0**2 <= d**2, which gets re-formulated to Delta0**2 - d**2 <= 0
        '''
        #print('ITR_GP_RTO: TR_con: This checkes whether d is within (or outside?) the trust region radius')
        #return self.Delta0**2 - d@self.TRmat@d.T
        return self.Delta0**2 - d@d.T

    ###################################
    # --- Mismatch Constraint --- #
    ##################################

    def mistmatch_con(self, xk, GP_con_i, d):
        '''
        mistmatch constraint
        '''
        #print('ITR_GP_RTO: mistmatch_con: This takes the point d xk and the GP for constraint i and returns the inference for constraint i at position (xk+d)')
        return GP_con_i.GP_inference_np((xk+d).flatten())


    #######################
    # --- TR function --- #
    #######################

    def Adjust_TR(self, Delta0, xk, xnew, GP_obj, i_rto):
        '''
        Adjusts the TR depending on the rho ratio between xk and xnew
        '''
        #print('ITR_GP_RTO: Adjust_TR: Adjusts the TR radius Delta0 depending on the rho ratio between xk and xnew')
        Delta_max, eta0, eta1          = self.Delta_max, self.eta0, self.eta1
        gamma_red, gamma_incr          = self.gamma_red, self.gamma_incr
        obj_system                     = self.obj_system

        # --- compute rho --- #
        plant_i     = obj_system(np.array(xk).flatten())
        plant_iplus = obj_system(np.array(xnew).flatten())
        rho         = (plant_i - plant_iplus)/( (GP_obj.GP_inference_np(np.array(xk).flatten())[0]) -
                                               (GP_obj.GP_inference_np(np.array(xnew).flatten())[0]) )

        # --- Update TR --- #
        if plant_iplus<plant_i: # this is what we want, since we want to minimize the objective system
            if rho>=eta0:
                if rho>=eta1:
                    Delta0 = min(Delta0*gamma_incr, Delta_max) # here we should add gradient !!
                elif rho<eta1:
                    Delta0 = Delta0*gamma_red
                # Note: xk = xnew this is done later in the code
            if rho<eta0:
                #xnew                    = xk
                #self.backtrack_l[i_rto] = True
                #print('rho<eta0 -- backtracking')
                Delta0 = Delta0*gamma_red
        else:
            xnew                    = xk
            Delta0                  = Delta0*gamma_red
            self.backtrack_l[i_rto] = True
            print('plant_iplus<plant_i -- backtracking')

        return Delta0, xnew, xk

    ###################################
    # --- Random Sampling in ball --- #
    ###################################

    def Ball_seeding(self, ndim):
        '''
        This function provides positions d_init for sampling randomly within a ball of radius self.Delta0
        '''
        # print('ITR_GP_RTO: Ball_seeding: providing random positions d_init in radius in a circle with radiuis of the trust region in each input dimension')
        u      = np.random.normal(0,1,ndim)  # random sampling in a ball
        norm   = np.sum(u**2)**(0.5)
        r      = random.random()**(1.0/ndim)
        d_init = r*u/norm*self.Delta0*2      # random sampling in a ball

        return d_init

    ########################################
    # --- Constrain Violation and step --- #
    ########################################

    def Step_constraint(self, Delta0, xk, xnew, GP_obj, i_rto):
        '''
        Calls Adjust_TR which adjusts the trust region and decides on the step size
        depending on constraint violation or the objective similarity
        '''
        # print('ITR_GP_RTO: Step_constraint')
        Adjust_TR = self.Adjust_TR

        if not self.System_Cons_check(np.array(xnew).flatten()):
            xnew                    = xk
            self.backtrack_l[i_rto] = True
            # print('Constraint violated -- backtracking')
            return Delta0*self.gamma_red, xnew, xk

        else:
            Delta0, xnew, xk = Adjust_TR(Delta0, xk, xnew, GP_obj, i_rto)
            return Delta0, xnew, xk

    ########################################
    # --- Constraint GP construction --- #
    ########################################

    def GPs_construction(self, xk, Xtrain, ytrain, ndatplus, H_norm=0.):
        '''
        Constructs a GP for every constraint and the objective function
        ng is the number of constraints^
        ytrain is a vector that contains the y values that correspond to the objective function on position 0 and the values that belong to the constraints on the other entries
        '''
        # print('ITR_GP_RTO: GPs_construction')

        # internal calls
        ndat, multi_hyper, ng, ndim    = self.ndat+ndatplus, self.multi_hyper, self.ng, self.ndim
        mistmatch_con                  = self.mistmatch_con

        # --- objective function GP --- #
        #print('ITR_GP_RTO: GPs_construction: Building GP_obj (Gaussian Process of the objective function fitted to the data (x,y_obj) provided without xk)')
        GP_obj     = GP_model(Xtrain, ytrain[0,:].reshape(ndat,1), 'RBF',
                              multi_hyper=multi_hyper, var_out=True)

        GP_con      = [0]*ng # Gaussian processes that output mistmatch (functools)
        GP_con_2    = [0]*ng # Constraints for the NLP

        GP_con_f    = [0]*(ng+1)

        for igp in range(ng):
            #print('ITR_GP_RTO: GPs_construction: For constraint ' + str(igp+1) + '/ ' + str(ng) + ': GP_con is build based on the data (x,y_constraint) without xk')
            GP_con[igp]     = GP_model(Xtrain, ytrain[igp+1,:].reshape(ndat,1), 'RBF',
                                       multi_hyper=multi_hyper, var_out=False)
            #print('ITR_GP_RTO: GPs_construction: For constraint ' + str(igp+1) + '/ ' + str(ng) + ': GP_con_f[igp] is build as inference (= prediction) wrapper of GP_con dependend on d given xk')
            GP_con_2[igp]   = functools.partial(mistmatch_con, xk, GP_con[igp]) # partially evaluating a function. This creates a callable on position igp in vector GP_con_2 that applies the function mistmatch_con, but with xk pre-set for xk and GP_con[igp] pre-set for GP_con_i. Thereby, the only remaining argument in the callable is d. This means, whenever the callable gets called with d, it calculates the probabilty of the violating the constraint at position xk+d.
            GP_con_f[igp]   = {'type': 'ineq', 'fun': GP_con_2[igp]}

        #print('ITR_GP_RTO: GPs_construction: TR_con is build as the trust region constraint (trust region is a circle that must have an area of Delta0^2 <= d^2')
        GP_con_f[igp+1] = {'type': 'ineq', 'fun': self.TR_con} #in the final position of vector GP_con_f there is the trust-region constraint
        #print('ITR_GP_RTO: GPs_construction: returning GPs for objective function (GP_obj), GPs of constraints for inference (GP_con_f), and the constrain GPs themselves (GP_con)')
        #print('ITR_GP_RTO: GPs_construction: GP_con_f has the GP_con, GP_con_2(= GP_con given d to predict the probabilty of GP_con being violated at xk+d, which calls the mismatch constraint), and TR_con on positions 0, 1, and 2')
        return GP_obj, GP_con_f, GP_con

    ######################################
    # --- Real-Time Optimization alg --- #
    ######################################

    def RTO_routine(self):
        '''
        --- description ---
        returns: X_opt, y_opt, TR_l, xnew, self.backtrack_l
        X_opt as all data_points evaluated
        y_opt as the objective function values corresponding to X_opt
        TR_l as the list of trust-regions
        x_new as the optimal point proposed by the final evaluation
        self.backtrack_l as the list of backtracking booleans (?)
        '''
        # print('ITR_GP_RTO: RTO_routine: internal variable calls')
        # internal variable calls
        store_data, Xtrain, ytrain      = self.store_data, self.Xtrain, self.ytrain
        multi_hyper, n_iter             = self.multi_hyper, self.n_iter
        ndim, ndat, multi_opt           = self.ndim, self.ndat, self.multi_opt
        GP_obj_f, bounds, ng            = self.GP_obj_f, self.bounds, self.ng
        multi_opt, Delta0, x0           = self.multi_opt, self.Delta0, self.x0
        obj_system, cons_system         = self.obj_system, self.cons_system
        Adjust_TR, Step_constraint      = self.Adjust_TR, self.Step_constraint
        mistmatch_con                   = self.mistmatch_con
        Ball_seeding                   = self.Ball_seeding
        GPs_construction                = self.GPs_construction
        self.n_outputs                  = len(cons_system) + 1

        # variable definitions
        # print('ITR_GP_RTO: RTO_routine: variable definitions')
        funcs_system       = [obj_system]+ cons_system
        self.backtrack_l   = [False]*n_iter #!!

        # --- building GP models from existing data --- #
        # evaluating initial point
        #print('ITR_GP_RTO: RTO_routine: initial point: setting xnew = x0')
        xnew    = x0
        #print('ITR_GP_RTO: RTO_routine: initial point: stacking Xtrain = xnew and Xtrain')
        Xtrain  = np.vstack((Xtrain,xnew))
        ynew    = np.zeros((1,ng+1))
        #print('ITR_GP_RTO: RTO_routine: initial point: evaluation objective function at Xtrain')
        for ii in range(ng+1):
                ynew[0,ii] = funcs_system[ii](np.array(xnew[:]).flatten())
        ytrain  = np.hstack((ytrain,ynew.T))
        #print('ITR_GP_RTO: RTO_routine: initial point: I now have ytrain')
        # --- building GP models for first time --- #
        #print('ITR_GP_RTO: RTO_routine: building GP models for GP_obj, GP_con_f, and GP_con first time based on xnew, Xtrain, ytrain. xnew appears double, in Xtrain and seperately as ')
        GP_obj, GP_con_f, GP_con = GPs_construction(xnew, Xtrain, ytrain, 1)

        # renaming data
        #print('ITR_GP_RTO: RTO_routine: renaming data, the initial Xtrain and ytrain are from now on in X_opt and y_opt')
        X_opt    = np.copy(Xtrain)
        y_opt    = np.copy(ytrain)

        # --- TR lists --- #
        TR_l    = ['error']*(n_iter+1)
        TR_l[0] = self.Delta0

        # --- rto -- iterations --- #
        # print('ITR_GP_RTO: RTO_routine: Iterations start now -----------------------------------------------------------------------------------------------------')
        options          = {'disp':False,'maxiter':10000}  # solver options
        lb, ub           = bounds[:,0], bounds[:,1]

        for i_rto in range(n_iter):
            # print('============================ RTO iteration ',i_rto)

            # --- optimization -- multistart --- #
            localsol = [0.]*multi_opt           # values for multistart
            localval = np.zeros((multi_opt))    # variables for multistart

            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': xk = xnew. xnew from previous iteration.')
            xk       = xnew
            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': re-centering the trust region on xk')
            TRb      = (bounds-xk.reshape(ndim,1, order='F'))
            # TODO: sampling on ellipse, not only Ball

            for j in range(multi_opt):
                #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ', multi_start: ' + str(j+1) + ': Ball_seeding')
                d_init = Ball_seeding(ndim) # random sampling in a ball
                # GP optimization
                #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ', multi_start: ' + str(j+1) + ': find d that minimizes mean-2sqrt(variance) based on xk and GP, bounds = TRb, constraints=GP_con_f. The next entry is from within the minimization loop.')
                res = minimize(GP_obj_f, d_init, args=(GP_obj, xk), method='SLSQP',
                               options=options, bounds=(TRb), constraints=GP_con_f, tol=1e-12)
                #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ', multi_start: ' + str(j+1) + ': based on the given position xk, distance d to promising xk+d found')
                localsol[j] = res.x
                if res.success == True:
                    localval[j] = res.fun # this stores the minimized function value (meaning the minimal mean and variance found for d at xk(fixed)+d)
                    #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ', multi_start: ' + str(j+1) + ': minimization successful, storing d as localsol[j] and objective function value as localval[j]')
                else:
                    localval[j] = np.inf
                    #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ', multi_start: ' + str(j+1) + ': minimization unsuccessful, storing d as localsol[j] and objective function value as infinity')
                #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ', multi_start: ' + str(j+1) + ': finished')
            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': all multistarts done, d resulting in lowest minimum for objective function found')
            if np.min(localval) == np.inf:
                print('warning, no feasible solution found')
            # selecting best point
            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': choosing best solution')
            minindex    = np.argmin(localval) # choosing best solution
            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': xnew = xk+d_best')
            xnew        = localsol[minindex] + xk  # selecting best solution

            # re-evaluate best point (could be done more efficiently - no re-evaluation)
            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': evaluate at xnew to obtain y')
            xnew = np.array([xnew]).reshape(1,ndim)
            ynew = np.zeros((1,ng+1))
            for ii in range(ng+1):
                #ynew[0,ii] = funcs[ii](np.array(xnew[:]).flatten())
                ynew[0,ii] = funcs_system[ii](np.array(xnew[:]).flatten())

            # adding new point to sample
            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': adding xnew and y point to X_opt and y_opt (=data storage)')
            X_opt  = np.vstack((X_opt,xnew))
            y_opt  = np.hstack((y_opt,ynew.T))

            # --- Update TR --- #
            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': Update Ball Sampling Radius Delta0, xnew and xk')
            self.Delta0, xnew, xk = Step_constraint(self.Delta0, xk, xnew, GP_obj, i_rto)      # adjust TR
            GP_obj, GP_con_f, GP_con = GPs_construction(xnew, X_opt, y_opt, 2+i_rto)

            # --- TR listing --- #
            #print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ': Store TR radius')
            TR_l[i_rto+1] = self.Delta0

            # print('ITR_GP_RTO: RTO_routine: rto -- iterations: iteration: ' + str(i_rto+1) + ' complete')

        # --- output data --- #
        return X_opt, y_opt, TR_l, xnew, self.backtrack_l
    

def CBO_TR(
        f,
        x_dim, #TODO here the functionality of multiple input dimensions has to be implemented
        bounds,  
        f_eval_, # length of trajectory (objective function evaluation budget)
        i_rep
        ): 

    '''
    function called in the benchmarking algorithm
    TODO this is supposed to become the function that is called from the benchmarking routine
    this is where we want to end up with: a, b, team_names, cids = i_algorithm(t_.fun_test, N_x_, bounds_, f_eval_)
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''
# comment 1233
    test_fun = f.fun_test
    test_con = f.con_test
    Xtrain = f.init_points[i_rep]
    x0 = f.x0[i_rep].flatten()
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2
    f_eval_ = f_eval_ - samples_number # the initial samples in CBO_TR need to be subtracted

    GP_opt = ITR_GP_RTO(  # TODO: Check whether the SafeOpt algorithm has similar hyperparameters and if these must then be packed outside the routine
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