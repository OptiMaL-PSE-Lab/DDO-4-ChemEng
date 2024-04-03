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

    test_results = {}

    #############
    # Dimension #
    #############
    for i_run in range(len(N_x_l)):
        N_x_                = N_x_l[i_run]
        f_eval_             = f_eval_l[i_run]
        dim_S               = 'D'+str(N_x_)
        test_results[dim_S] = {}
        bounds_             = np.array([[-5,5] for i in range(N_x_)])

        #############
        # Functions #
        #############
        for i_function in functions_test:
            print('===================== ',i_function,'D'+str(N_x_))
            test_results[dim_S][i_function]              = {}
            test_results[dim_S][i_function]['all means'] = {}
            all_f_                                       = []
            randShift_l                                  = np.random.uniform(-3,3, (reps,N_x_))
            
            
            ##############
            # Algorithms #
            ##############
            info = []

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
                    a, b, team_names, cids = i_algorithm(t_.fun_test, N_x_, bounds_, f_eval_)
                    # post-processing
                    t_.best_f_list()
                    t_.pad_or_truncate(f_eval_)
                    # store result
                    test_results[dim_S][i_function][str(i_algorithm.__name__)].append(copy.deepcopy(t_.best_f_c))     
                # statistics from each algorithm for a function    
                l_   = np.array(test_results[dim_S][i_function][str(i_algorithm.__name__)])
                m_   = np.mean(l_, axis=0)
                test_results[dim_S][i_function]['all means'][str(i_algorithm.__name__)] = copy.deepcopy(m_)
                all_f_.append(copy.deepcopy(m_))
                info.append({'alg_name': str(i_algorithm.__name__), 'team names': team_names, 'CIDs': cids})
            # statistics for all algorithms for a function       
            test_results[dim_S][i_function]['mean']   = np.mean(all_f_, axis=0)
            test_results[dim_S][i_function]['median'] = np.median(all_f_, axis=0)
            test_results[dim_S][i_function]['q 0']    = np.max(all_f_, axis=0)
            test_results[dim_S][i_function]['q 100']  = np.min(all_f_, axis=0)

    return info, test_results
            

def ML4CE_table_uncon(
        test_results,
        algorithms_test,
        functions_test,
        N_x_l,
        start_,
        info
        ):

    # computing table of results
    alg_perf   = {}
    test_res   = test_results
    algs_test  = algorithms_test
    funcs_test = functions_test
    n_f        = len(funcs_test)
    # use for performance table
    columns = funcs_test + ['Multimodal','Unimodal','All'] # !!! This needs to be in the same order as m_ and u_ (see below) !!!
    multim  = ['Rastringin_f', 'Ackely_f']
    unim    = ['Rosenbrock_f', 'Levy_f']
    rows    = ['D2', 'D5', 'D10', 'All']

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
                medall_ = test_res[dim_S][funcs_test[i_fun]]['mean']
                trial_  = test_res[dim_S][funcs_test[i_fun]]['all means'][str(i_alg.__name__)]
                lowall_ = test_res[dim_S][funcs_test[i_fun]]['q 100']
                higall_ = test_res[dim_S][funcs_test[i_fun]]['q 0']
                # score performance
                perf_ = ( (higall_[start_[i_dim]:] - trial_[start_[i_dim]:])
                        /(higall_[start_[i_dim]:] - lowall_[start_[i_dim]:]) )
                alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i_fun]] = copy.deepcopy(np.sum(perf_)/len(perf_))

    n_f   = len(funcs_test)
    n_p   = len(funcs_test)*len(N_x_l)
    nrow  = int(n_p/2) if n_p%2==0 else int(n_p/2)+1

    for i_alg in algs_test:
        #plt.figure(figsize=(16, 25))        
        # Plot bars and create text labels for the table
        cell_text = []
        n_rows    = len(rows)
        # for different dimensions
        for row in  range(len(N_x_l)):
            dim_S = 'D'+str(N_x_l[row])
            r_    = [alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i]] for i in range(n_f)]
            m_    = [alg_perf[dim_S][str(i_alg.__name__)][j] for j in multim] # !!! This needs to be in order (see above) !!!
            u_    = [alg_perf[dim_S][str(i_alg.__name__)][k] for k in unim]
            c_    = r_ + [sum(m_)/len(m_)] + [sum(u_)/len(u_)] + [sum(r_)/len(r_)]
            c_    = [round(c_i,2) for c_i in c_]
            cell_text.append(c_)
        # for all dimensions the below need fixing
        r_    = [sum([alg_perf['D'+str(N_x_l[j])][str(i_alg.__name__)][funcs_test[i]] 
                    for j in range(len(N_x_l))])/len(N_x_l) 
                for i in range(n_f)]
        m_    = [sum([alg_perf['D'+str(N_x_l[j])][str(i_alg.__name__)][jj] 
                    for j in range(len(N_x_l))])/len(N_x_l) 
                for jj in multim]
        u_    = [sum([alg_perf['D'+str(N_x_l[j])][str(i_alg.__name__)][k] 
                    for j in range(len(N_x_l))])/len(N_x_l) 
                for k in unim]
        c_    = r_ + [sum(m_)/len(m_)] + [sum(u_)/len(u_)] + [sum(r_)/len(r_)]
        c_    = [round(c_i,2) for c_i in c_]
        cell_text.append(c_)
        
        # plot table
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=cell_text,
                            rowLabels=rows,
                            colLabels=columns,
                            loc='center',
                            label = 'test'
                            )
        ax.set_title(i_alg.__name__)
        fig.tight_layout()
