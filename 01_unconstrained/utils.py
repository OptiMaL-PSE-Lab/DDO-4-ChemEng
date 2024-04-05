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
# importing algorithms
from BFGS_multistart import* 
from Stochastic_local_search import* 
from Cuadratic_opt import*
from Scipy_opt_algs import*
from BO_NpScpy import*
# importing test functions
from test_function import* 

def ML4CE_eval_algs_uncon(
        N_x_l, 
        f_eval_l,
        functions_test,
        algorithms_test,
        reps,
        ):

    trajectories = {}

    #############
    # Dimension #
    #############
    for i_run in range(len(N_x_l)):
        N_x_                = N_x_l[i_run]
        f_eval_             = f_eval_l[i_run]
        dim_S               = 'D'+str(N_x_)
        trajectories[dim_S] = {}
        bounds_             = np.array([[-5,5] for i in range(N_x_)])

        #############
        # Functions #
        #############
        for i_function in functions_test:
            print('===================== ',i_function,'D'+str(N_x_))
            trajectories[dim_S][i_function]              = {}
            trajectories[dim_S][i_function]['all means'] = {}
            all_f_                                       = []
            randShift_l                                  = np.random.uniform(-3,3, (reps,N_x_))
            
            
            ##############
            # Algorithms #
            ##############
            info = []

            for i_algorithm in algorithms_test:
                print('== ',str(i_algorithm.__name__))
                trajectories[dim_S][i_function][str(i_algorithm.__name__)] = []
                
                ###############
                # Repetitions #
                ###############
                for i_rep in range(reps):
                    # random shift
                    x_shift_ = randShift_l[i_rep,:].reshape((N_x_,1))
                    # test function
                    t_       = Test_function(i_function, N_x_, False, x_shift_)
                    # algorithm
                    a, b, team_names, cids = i_algorithm(t_.fun_test, N_x_, bounds_, f_eval_)
                    # post-processing
                    t_.best_f_list()
                    t_.pad_or_truncate(f_eval_)
                    # store result
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)].append(copy.deepcopy(t_.best_f_c))     
                # statistics from each algorithm for a function    
                l_   = np.array(trajectories[dim_S][i_function][str(i_algorithm.__name__)])
                m_   = np.mean(l_, axis=0)
                trajectories[dim_S][i_function]['all means'][str(i_algorithm.__name__)] = copy.deepcopy(m_)
                all_f_.append(copy.deepcopy(m_))
                info.append({'alg_name': str(i_algorithm.__name__), 'team names': team_names, 'CIDs': cids})
            # statistics for all algorithms for a function       
            trajectories[dim_S][i_function]['mean']   = np.mean(all_f_, axis=0)
            trajectories[dim_S][i_function]['median'] = np.median(all_f_, axis=0)
            trajectories[dim_S][i_function]['q 0']    = np.max(all_f_, axis=0)
            trajectories[dim_S][i_function]['q 100']  = np.min(all_f_, axis=0)

    return info, trajectories
            

def ML4CE_table_uncon(
        trajectories, 
        algs_test,
        funcs_test,
        multim,
        N_x_l,
        start_,
        ):
    
    '''
    This function calculates the test results based on the trajectories and puts them in a format ready to be plotted in tables
    Furthermore, it normalizes the results among all algorithms for a given dimension and test function. By this, the height profile
    of the functions gets vanished, leading to an increased comparability among the performance on unimodal and multimodal functions.
    Reason: Unimodal tend to have higher function values in general which seems for the algorithms to perform worse
    
        multim: it gets the list of multi-modal functions, the rest are assumed to be uni-modal
        N_x_l: Input dimensions to be benchmarked
        start_: No of starting points per dimension
    '''

    # computing table of results
    alg_perf   = {}
    n_f        = len(funcs_test)


    # for every algorithm
    for i_dim in range(len(N_x_l)): 
        dim_S           = 'D'+str(N_x_l[i_dim])
        alg_perf[dim_S] = {}
        for i_alg in algs_test:
            # print('==  ',str(i_alg.__name__),' ==')
            alg_perf[dim_S][str(i_alg.__name__)] = {}
            # for every function
            for i_fun in range(n_f):      
                # retrive performance
                medall_ = trajectories[dim_S][funcs_test[i_fun]]['mean']
                trial_  = trajectories[dim_S][funcs_test[i_fun]]['all means'][str(i_alg.__name__)]
                lowall_ = trajectories[dim_S][funcs_test[i_fun]]['q 100']
                higall_ = trajectories[dim_S][funcs_test[i_fun]]['q 0']
                # score performance
                perf_ = ( (higall_[start_[i_dim]:] - trial_[start_[i_dim]:])
                        /(higall_[start_[i_dim]:] - lowall_[start_[i_dim]:]) )
                alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i_fun]] = copy.deepcopy(np.sum(perf_)/len(perf_))

    cell_text_global = []

    for i_alg in algs_test:
        # Plot bars and create text labels for the table
        # for different dimensions
        for row in  range(len(N_x_l)):
            dim_S = 'D'+str(N_x_l[row])
            r_    = [alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i]] for i in range(n_f)]
            cell_text_global.append(r_)
    
    # return array
    cell_text_global_array = np.array(cell_text_global)

    matrix_store_l = []
    
    # iteratively select all the performances for changing dimensions 
    # (each column is a test function and each row is an algorihm)
    for i in range(len(N_x_l)):

        # select matrix for dimension i
        matrix = cell_text_global_array[i::len(N_x_l)]
        # obtain column-wise the min-max values 
        matrix_min = matrix.min(axis=0)
        matrix_max = matrix.max(axis=0)
        # normalize column-wise
        matrix_norm = (matrix-matrix_min)/(matrix_max-matrix_min)
        matrix_store_l.append(matrix_norm)


    ### This is super blocky and needs to be simplyfied ###

    # each matrix in here represents an input dimension. Each row is an algorithm, each column is a test function
    combined_matrix_norm = np.stack(matrix_store_l, axis=0)

    # have each matrix be an algorithm, each row be a dimension and each column a test function
    combined_matrix_norm_reshape = np.transpose(combined_matrix_norm,axes=(1, 0, 2))

    ### start building averages for the multimodal functions ###

    # Calculate row-wise averages of the first two original columns
    row_wise_averages_first = np.mean(combined_matrix_norm_reshape[:, :, :len(multim)], axis=2)

    # Expand dimensions to allow broadcasting
    row_wise_averages_first_expanded = row_wise_averages_first[:, :, np.newaxis]

    # Add the new column to each matrix
    result1 = np.concatenate((combined_matrix_norm_reshape, row_wise_averages_first_expanded), axis=2)

    # Calculate row-wise averages of the second two original columns
    row_wise_averages_second = np.mean(combined_matrix_norm_reshape[:, :, len(multim):], axis=2)

    # Expand dimensions to allow broadcasting
    row_wise_averages_second_expanded = row_wise_averages_second[:, :, np.newaxis]

    # Add the new column to each matrix
    result2 = np.concatenate((result1, row_wise_averages_second_expanded), axis=2)

    # Calculate row-wise averages of the two new columns
    row_wise_averages_new = np.mean(result2[:, :, :6], axis=2)

    # Expand dimensions to allow broadcasting
    row_wise_averages_new_expanded = row_wise_averages_new[:, :, np.newaxis]

    # Add the new column to each matrix
    result3 = np.concatenate((result2, row_wise_averages_new_expanded), axis=2)

    ### calculate the final row ###
    # Calculate column-wise averages across each matrix
    column_wise_averages = np.mean(result3, axis=1)

    # Stack the column-wise averages as an additional row to each matrix
    arr_with_avg_row = np.concatenate((result3, column_wise_averages[:, np.newaxis, :]), axis=1)

    return np.around(arr_with_avg_row, decimals=2)