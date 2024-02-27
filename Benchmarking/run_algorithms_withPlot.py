# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:44:12 2023

@author: adelrioc
"""
###########
# Imports #
###########

# importing libraries
import matplotlib.pyplot as plt
import copy
from Stochastic_local_search import* 

# importing test functions
from test_function import* 
# importing utilities
from plot_utils import*

import sys
sys.path.append('safeopt_local')
from safeopt_f import*
from safeopt_alg import*

##########################
# Algorithms & Functions #
##########################

algorithms_test = [SS_alg, safeopt_alg]

functions_test  = ['Ackely_f','Rosenbrock_f', 'Levy_f', 'Rastringin_f']

###########################
# Optimization parameters #
###########################

N_x_l        = [2, 2, 2]    # dimensions
start_       = [5, 10, 15]   # start counting
reps         = 3
f_eval_l     = [5, 10, 20] # function evaluations
test_results = {}

#############
# Dimension #
#############
for i_run in range(len(N_x_l)):
    N_x_                = N_x_l[i_run]
    f_eval_             = f_eval_l[i_run]
    dim_S               = 'D'+str(N_x_)
    test_results[dim_S] = {}
    # TODO change to -7,7
    bounds_             = np.array([[-8,8] for i in range(N_x_)])
    
    #############
    # Functions #
    #############
    for i_function in functions_test:
        print('===================== ',i_function,'D'+str(N_x_))
        test_results[dim_S][i_function]              = {}
        test_results[dim_S][i_function]['all means'] = {}
        test_results[dim_S][i_function]['all 95']    = {}
        test_results[dim_S][i_function]['all 05']    = {}
        all_f_                                       = []
        randShift_l                                  = np.random.uniform(-3,3, (reps,N_x_))
        
        
        ##############
        # Algorithms #
        ##############
        for i_algorithm in algorithms_test:
            print('== ',str(i_algorithm.__name__))
            test_results[dim_S][i_function][str(i_algorithm.__name__)] = []
            
            ###############
            # Repetitions #
            ###############
            for i_rep in range(reps):
                # random shift
                x_shift_ = randShift_l[i_rep,:].reshape((N_x_,1))
                # test function
                t_       = Test_function(i_function, N_x_, False, x_shift_)
                # algorithm
                i_algorithm(t_.fun_test, N_x_, bounds_, f_eval_)
                # post-processing
                t_.best_f_list()
                t_.pad_or_truncate(f_eval_)
                # store result
                test_results[dim_S][i_function][str(i_algorithm.__name__)].append(copy.deepcopy(t_.best_f_c))     
            # statistics from each algorithm for a function    
            l_   = np.array(test_results[dim_S][i_function][str(i_algorithm.__name__)])
            m_   = np.mean(l_, axis=0)
            q05_ = np.quantile(l_, 0.05, axis=0)
            q95_ = np.quantile(l_, 0.95, axis=0)
            test_results[dim_S][i_function]['all means'][str(i_algorithm.__name__)] = copy.deepcopy(m_)
            test_results[dim_S][i_function]['all 95'][str(i_algorithm.__name__)]    = copy.deepcopy(q05_)
            test_results[dim_S][i_function]['all 05'][str(i_algorithm.__name__)]    = copy.deepcopy(q95_)
            all_f_.append(copy.deepcopy(l_))
        # statistics for all algorithms for a function       
        test_results[dim_S][i_function]['mean']   = np.mean(all_f_, axis=(0,1))
        test_results[dim_S][i_function]['median'] = np.median(all_f_, axis=(0,1))
        test_results[dim_S][i_function]['q 0']    = np.max(all_f_, axis=(0,1))
        test_results[dim_S][i_function]['q 100']  = np.min(all_f_, axis=(0,1))
    
import pickle
#import _pickle as pickle 

#with open('results_algorithms_Plots.py', 'wb') as file:
#    file.write(pickle.dumps(test_results))

# plot results
folderplt_ = 'Benchmarking/plots/Convergence_plots/'
plot_performance(test_results, algorithms_test, functions_test, folderplt_, N_x_l)
            
