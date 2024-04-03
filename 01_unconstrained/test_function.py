# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:00:56 2023

@author: adelrioc
"""

import numpy as np
import copy

class Test_function:
    
    #################################
    # --- initializing function --- #
    #################################    
    def __init__(self, func_type, n_x, track_x, x_shift):

        '''
        x_shift: this is the starting point that is set for each repitition (5 per algorithm and function)
        it was randomly chosen for each dimension between -3 and 3
        '''
        
        # initialize lists
        self.f_list = []; self.x_list = []; 
        # function specifications
        self.func_type = func_type
        self.n_x       = n_x
        self.track_x   = track_x
        self.x_shift   = x_shift
    
    
    ################    
    # run function #
    ################
    
    def fun_test(self, x):
        
        if self.func_type == 'Rosenbrock_f':
            '''
            Test function: Rosenbrock function (http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html)
            Note: Website does not work anymore (checked Feb_24_2024), this one does: https://www.sfu.ca/~ssurjano/rosen.html
            Note:
                works only for vectors

            
            x: this is the argument that the function gets during the evaluation.
            The way this benchmarking is coded is, that there is a coordinate shift for each repitition to x_shift. Meaning all 
            optimisation algorithms provide solutions based on x_shift. In order to receive the true value of the objective
            function during evaluation, the candidate x has to be added to x_shift. 
            '''

            x = np.array(x)
            x = x.reshape((self.n_x,1))
            x = x + self.x_shift
            z = (1.-x).T@(1.-x) + 100*(x[1:]-x[:-1]**2).T@(x[1:]-x[:-1]**2)
            
            # track f 
            self.f_list.append(z[0,0]) 
            # track x
            if self.track_x:
                self.x_list.append(x) 
                
            # return objective        
            return z[0,0]
        
        elif self.func_type == 'Levy_f':
            '''
            Test function: Levy function (https://www.sfu.ca/~ssurjano/levy.html)
            Note:
                works only for vectors
            '''
            x = np.array(x)
            x = x.reshape((self.n_x,1))
            x = x + self.x_shift
            w   = 1. + (x-1.)/4.
            z   = np.sin(np.pi*w[0])**2 + np.sum((w[:-1]-1.)**2*(1. + 10.*np.sin(np.pi*w[:-1]+1.)**2)) \
            + (w[-1]-1.)**2*(1.+np.sin(2.*np.pi*w[-1]))
            
            # track f 
            self.f_list.append(z[0]) 
            # track x
            if self.track_x:
                self.x_list.append(x) 
            
            # return objective
            return z[0]
        
        elif self.func_type == 'Rastringin_f':
            '''
            Test function: Rastringin function (https://www.sfu.ca/~ssurjano/rastr.html)
            Note:
                works only for vectors
            '''
            x = np.array(x)
            x = x.reshape((self.n_x,1))
            x = x + self.x_shift
            z   = 10.*self.n_x +  np.sum(x**2 - 10.*np.cos(2.*np.pi*x))
            
            # track f 
            self.f_list.append(z) 
            # track x
            if self.track_x:
                self.x_list.append(x) 
            
            # return objective
            return z
        
        elif self.func_type == 'Ackely_f':
            '''
            Test function: Ackely function (https://www.sfu.ca/~ssurjano/ackley.html)
            Note:
                works only for vectors
            '''
            x = np.array(x)
            x = x.reshape((self.n_x,1))
            x = x + self.x_shift


            a = 20.; b=0.2; c=2.*np.pi
        
            x = x.reshape((self.n_x,1))
            z = ( -a * np.exp(-b*np.sqrt(1./self.n_x*np.sum(x**2))) - 
                 np.exp(1./self.n_x*np.sum(np.cos(c*x))) + a + np.exp(1.) )
            
            # track f 
            self.f_list.append(z) 
            # track x
            if self.track_x:
                self.x_list.append(x) 
            
            # return objective
            return z
            
        
    ####################    
    # re-arrange lists #
    ####################
    
    def best_f_list(self):
        '''
        Returns
        -------
        List of best points so far
        '''
        self.best_f = [min(self.f_list[:i]) for i in range(1,len(self.f_list))]
        
    #############    
    # cut lists #
    #############
    
    def pad_or_truncate(self, n_p):
        '''
        n_p: number of desired elements on list
        -------
        Truncate or pad list 
        '''
        # get last element
        b_last = copy.deepcopy(self.best_f[:n_p])[-1]
        l_last = copy.deepcopy(self.f_list[:n_p])[-1]   
        
        self.best_f_c = copy.deepcopy(self.best_f[:n_p]) + [b_last]*(n_p - len(self.best_f[:n_p]))
        self.f_list_c = copy.deepcopy(self.f_list[:n_p]) + [l_last]*(n_p - len(self.f_list[:n_p]))
        
        
# TODO -> plot every algorithm trajectory with best, meadian, 25% and 75%
# computer best trajectory inside class algorithm
# compute all statistics later. 