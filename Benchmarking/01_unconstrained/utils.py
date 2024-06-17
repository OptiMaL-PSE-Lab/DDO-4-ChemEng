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
from Scipy_opt_algs import*
from BO_NpScpy import*
from COBYQA import *
# importing test functions
from test_function import* 
import os
from datetime import datetime
from pylab import grid
import pickle

'''
how do the algorithms deal with start-data?

I SHOULD MANUALLY CHECK THE f_eval

For now: Just provide a global starting point for all methods.

LS_QM:
- initiates numpy array f_list of shape (trajectory length) and x_list of shape (trajectory_length, dimensions)
- f_list and x_list are filled with 1e-13 (practically zero)
- then n_rs = int(max(x_dim+1,iter_tot*.05)) is determined
- then n_rs points are generated with a custom Random_search_Qopt and inserted into the first n_rs positions in x_list
- THIS UPPER-PART IS AFFECTED BY THE LIST PROVIDED. BASICALLY, n_rs has to be taken from somewhere (either in the algorithm itself
or provided within the algorithm itself. Best would be to have it as an argument for the generation of points)
- then the first n_rs positions in f_list are filled by evaluating x_list
- in the very end, f_list and x_list are completely filled (however, the points achieved are taken from t_.best_f so f_list and x_list are not returned
- TODO: The first quadratic model is built using the n_rs points provided. This is the same as the BO case his could then be comparable with the BO-case

- here I found out that the length of the f_eval (which creates the length of the list for the starting points) is 
dependend on the length of this list of N_x_l. By this, when I comment-out several dimensions, f_eval has not the correct length

COBYLA:
- n_rs = int(max(x_dim+1,iter_tot*.05)) is used to determine number of starting points
- Random search is used to determine the best point
- best point is then given to the scipy implementation
- the list for x and f is managed in the objective function
- TODO: check how the first model is built here

COBYQA:
- TODO: It is not accessable which points are used to build the first model. Only x_best is provided to the algorithm. 
        COBLYQA then probably builds a model in a certain radius around this, but the points it uses are not accessible
-       Therefore this is different to BO and LS_QM, who could technically be build based on the same datapoints.
- same as COBYLA


BO:
- TODO: needs to be re-done with manual insertion of dimensions and iterations (see LS_QM last points)
- produces 5 starting points per dimension to build the initial models. they are in the shape (n_points,n_dimension)
- compute_data then takes the points and evaulates the points in the objective function and builds the first GP


The algorithms should be adjusted in a way where they receive the list of starting points, evaluate the starting points and 
then pick the best one. So this functionality should remain within the algorithms.

def ML4CE_uncon_gen_start(dim, reps, bounds, function, seed)
    think about here: at which point is this function to be called?
    should be for each function the same starting points for each dimension for each algorithm for each rep. 
    this means depending on the repetition
    there should be a list of starting points generated in the beginning of the benchmarking and depending on the 
    so we need a dictionary, that has a list of dimensions, per dimension a list of all functions and per function a list for all repetitions
    then, the algorithms take a list of starting points depending on the dimension, the function and the repetition. 
    by this, all algorithm get the same set of starting points per repetition, dimension and function
    depending on how the algorithms deal with these initial points I have to check. probably then just the 
    evaluation of f_best and x_best anstelle der random search
    since BO needs 10 starting points to create the first model, it is probably the easiest, to 

'''

def save_dict(target_folder, data_dict, timestamp):
    # Create a folder with current date and time
    folder_path = os.path.join(target_folder, timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # Save dictionary to file within the created folder
    file_path = os.path.join(folder_path, 'trajectories.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)


def ML4CE_uncon_eval(
        N_x_l, 
        f_eval_l,
        functions_test,
        algorithms_test,
        reps,
        home_dir,
        SafeData = False,
        ):

    trajectories = {}
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    #############
    # Dimension #
    #############
    for i_run in range(len(N_x_l)):
        N_x_                = N_x_l[i_run]
        f_eval_             = f_eval_l[i_run]
        dim_S               = 'D'+str(N_x_)
        trajectories[dim_S] = {}
        bounds_             = np.array([[-7,7] for i in range(N_x_)])

        #############
        # Functions #
        #############
        for i_function in functions_test:
            print('===================== ',i_function,'D'+str(N_x_))
            trajectories[dim_S][i_function]              = {}
            trajectories[dim_S][i_function]['all means'] = {}
            trajectories[dim_S][i_function]['all 90']    = {}
            trajectories[dim_S][i_function]['all 10']    = {}
            trajectories[dim_S][i_function]['f_list']    = {}
            trajectories[dim_S][i_function]['x_list']    = {}
            all_f_                                       = []
            randShift_l                                  = np.random.uniform(-3,3, (reps,N_x_))

            # save random shift
            trajectories[dim_S][i_function]['rand_shift'] = randShift_l
            
            ##############
            # Algorithms #
            ##############
            info = []

            for i_algorithm in algorithms_test:
                print('== ',str(i_algorithm.__name__))
                trajectories[dim_S][i_function][str(i_algorithm.__name__)] = []
                trajectories[dim_S][i_function]['f_list'][str(i_algorithm.__name__)] = []
                trajectories[dim_S][i_function]['x_list'][str(i_algorithm.__name__)] = []
                
                ###############
                # Repetitions #
                ###############
                for i_rep in range(reps):
                    # random shift
                    x_shift_ = randShift_l[i_rep,:].reshape((N_x_,1))
                    # test function
                    t_       = Test_function(i_function, N_x_, True, x_shift_, bounds_)
                    # algorithm
                    a, b, team_names, cids = i_algorithm(t_, N_x_, bounds_, f_eval_)
                    # post-processing
                    t_.best_f_list()                    # List of best points so far
                    t_.pad_or_truncate(f_eval_)         
                    # store result
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)].append(copy.deepcopy(t_.best_f_c))
                    trajectories[dim_S][i_function]['f_list'][str(i_algorithm.__name__)].append(copy.deepcopy(t_.f_list))
                    trajectories[dim_S][i_function]['x_list'][str(i_algorithm.__name__)].append(copy.deepcopy(t_.x_list))
                    # safe data in an overwriting fashion
                    if SafeData == True: save_dict(home_dir, trajectories, timestamp)
                # statistics from each algorithm for a function    
                l_   = np.array(trajectories[dim_S][i_function][str(i_algorithm.__name__)])
                m_   = np.mean(l_, axis=0)
                q10_ = np.quantile(l_, 0.10, axis=0)
                q90_ = np.quantile(l_, 0.90, axis=0)
                trajectories[dim_S][i_function]['all means'][str(i_algorithm.__name__)] = copy.deepcopy(m_)
                trajectories[dim_S][i_function]['all 90'][str(i_algorithm.__name__)]    = copy.deepcopy(q10_)
                trajectories[dim_S][i_function]['all 10'][str(i_algorithm.__name__)]    = copy.deepcopy(q90_)
                all_f_.append(copy.deepcopy(l_))
                info.append({'alg_name': str(i_algorithm.__name__), 'team names': team_names, 'CIDs': cids})
                # safe data in an overwriting fashion
                if SafeData == True: save_dict(home_dir, trajectories, timestamp)
            # statistics for all algorithms for a function       
            # trajectories[dim_S][i_function]['mean']   = np.mean(all_f_, axis=0)     # old
            # trajectories[dim_S][i_function]['median'] = np.median(all_f_, axis=0)   # old
            # trajectories[dim_S][i_function]['q 0']    = np.max(all_f_, axis=0)      # old
            # trajectories[dim_S][i_function]['q 100']  = np.min(all_f_, axis=0)      # old
            trajectories[dim_S][i_function]['mean']   = np.mean(all_f_, axis=(0,1))
            trajectories[dim_S][i_function]['median'] = np.median(all_f_, axis=(0,1))
            trajectories[dim_S][i_function]['q 0']    = np.max(all_f_, axis=(0,1))
            trajectories[dim_S][i_function]['q 100']  = np.min(all_f_, axis=(0,1))
            # safe data in an overwriting fashion
            if SafeData == True: save_dict(home_dir, trajectories, timestamp)

    # over-write one last time
    if SafeData == True:
        
        save_dict(home_dir, trajectories, timestamp)

        return info, trajectories, timestamp
    
    else:

        return info, trajectories, None
    
    
                

def ML4CE_uncon_table(
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
            print('==  ',str(i_alg.__name__),' ==')
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



def ML4CE_uncon_table_plot(array, functions_test, algorithms_test, N_x_l, home_dir, timestamp, SafeFig = False):
    
    columns = functions_test + ['Multimodal','Unimodal','All'] # !!! Do not change this order !!!
    rows_names_l = ['D' + str(num) for num in N_x_l]
    rows_names_l.append('All')
    num_tables = array.shape[0] # num of tables is the number of algorithms



    for i in range(num_tables):
        matrix = array[i]
        plt.figure(figsize=(15,15))
        plt.axis('off')
        plt.table(
            cellText=matrix, 
            loc='center', 
            cellLoc='center',
            colLabels=columns,
            rowLabels=rows_names_l
            )
        # plt.title(f'Table {algorithms_test[i].__name__}')

        plt.tight_layout()  # Adjust the layout to prevent overlapping
        

        if SafeFig == True:

            def directory_exists(directory_name):
                root_directory = os.getcwd()  # Root directory on Unix-like systems
                directory_path = os.path.join(root_directory, directory_name)
                return os.path.isdir(directory_path)

            directory_name = os.path.join(home_dir, timestamp, 'tables')

            if directory_exists(directory_name):

                plt.savefig(directory_name + '/{}.png'.format(algorithms_test[i].__name__))
            else:
                print(f"The directory '{directory_name}' does not exist in the root directory. Creating directory.")
                os.mkdir(directory_name)
                plt.savefig(directory_name + '/{}.png'.format(algorithms_test[i].__name__))
        
            plt.close()

        else:
            plt.title(algorithms_test[i].__name__)
            plt.show()
                

def ML4CE_uncon_graph_abs(test_res, algs_test, funcs_test, N_x_l, home_dir, timestamp, SafeFig=False):

    colors = plt.cm.tab10(np.linspace(0, 1, len(algs_test)))
    line_styles = ['-', '--', '-.', ':']  
    alg_indices = {alg: i for i, alg in enumerate(algs_test)}

    n_f = len(funcs_test)

    for i_dim in range(len(N_x_l)):
        dim_S = 'D' + str(N_x_l[i_dim])
        for i_fun in range(n_f):
            plt.figure(figsize=(15, 15))
            for i_alg in algs_test:
                trial_ = test_res[dim_S][funcs_test[i_fun]]['all means'][str(i_alg.__name__)]
                up_     = test_res[dim_S][funcs_test[i_fun]]['all 90'][str(i_alg.__name__)]
                down_   = test_res[dim_S][funcs_test[i_fun]]['all 10'][str(i_alg.__name__)]
                alg_index = alg_indices[i_alg]  # Get the index of the algorithm
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]  # Use modulo to cycle through line styles
                plt.plot(trial_, color=color, linestyle=line_style, lw=3, label=str(i_alg.__name__))
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=False)
                plt.gca().fill_between(x_ax,down_, up_, color=color, alpha=0.2)

                # Calculate the position of the vertical line based on the length of the trajectory
                length = len(down_)
                if length == 20:
                    vline_pos = 5
                elif length == 50:
                    vline_pos = 10
                elif length == 100:
                    vline_pos = 15
                else:
                    vline_pos = None  # Or set a default value if needed

                # Add the vertical line if a valid position is calculated
                if vline_pos is not None:
                    plt.axvline(x=vline_pos, color='red', linestyle='--', linewidth=2)

                # Setting x-axis ticks to integer values starting from 0 and showing every 5th tick
                tick_positions = np.arange(0, len(down_), 5)
                if (len(down_) - 1) % 5 != 0:  # If the last position is not already included
                    tick_positions = np.append(tick_positions, len(down_) - 1)
                tick_labels = np.arange(0, len(down_), 5)
                if len(tick_labels) < len(tick_positions):
                    tick_labels = np.append(tick_labels, len(down_) - 1)

                plt.xticks(tick_positions, tick_labels, fontsize=24, fontname='Times New Roman')

            plt.ylabel('obj value', fontsize = '28', fontname='Times New Roman')
            plt.xlabel('iterations', fontsize = '28', fontname='Times New Roman')
            # plt.yscale('log')
            plt.legend(loc='best', prop={'family':'Times New Roman', 'size': 24})
            plt.tick_params(axis='x', labelsize=24, labelcolor='black', labelfontfamily='Times New Roman')  # Set size and font name of x ticks
            plt.tick_params(axis='y', labelsize=24, labelcolor='black', labelfontfamily='Times New Roman')  # Set size and font name of y ticks
            # plt.title(funcs_test[i_fun] + ' ' + dim_S + ' convergence plot')
            # grid(True)
        
            if SafeFig == True:

                def directory_exists(directory_name):
                    root_directory = os.getcwd()  # Root directory on Unix-like systems
                    directory_path = os.path.join(root_directory, directory_name)
                    return os.path.isdir(directory_path)

                directory_name = os.path.join(home_dir, timestamp, 'trajectory_plots_1D')
                if directory_exists(directory_name):

                    plt.savefig(directory_name + '/{}_{}_1D.png'.format(dim_S, funcs_test[i_fun]))
                else:
                    print(f"The directory '{directory_name}' does not exist in the root directory.")
                    os.mkdir(directory_name)
                    plt.savefig(directory_name + '/{}_{}_1D.png'.format(dim_S, funcs_test[i_fun]))



import numpy as np
import matplotlib.pyplot as plt

def ML4CE_uncon_contour_allin1(functions_test, algorithms_test, N_x_, x_shift_origin, bounds_, bounds_plot, SafeFig=False):
    f_eval_ = 40  # trajectory length (= evaluation budget) --> Keep in mind that BO uses 10 datapoints to build the model when plotting
    track_x = True
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms_test)))
    alg_indices = {alg: i for i, alg in enumerate(algorithms_test)}

    for fun_ in functions_test:
        t_ = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

        # evaluate grid with vmap
        n_points = 100
        x1lb = bounds_plot[0][0]
        x1ub = bounds_plot[0][1]
        x1 = np.linspace(start=x1lb, stop=x1ub, num=n_points)
        x2lb = bounds_plot[1][0]
        x2ub = bounds_plot[1][1]
        x2 = np.linspace(start=x2lb, stop=x2ub, num=n_points)
        X1, X2 = np.meshgrid(x1, x2)

        # define plot
        plt.figure(figsize=(15, 15))
        ax3 = plt.subplot()

        # Initialize an empty array to store results
        y = np.empty((n_points, n_points))

        # Iterate over each element and apply the function
        for i in range(n_points):
            for j in range(n_points):
                y[i, j] = t_.fun_test(np.array([[X1[i, j], X2[i, j]]]))

        # add objective contour
        ax3.contour(X1, X2, y, 50)

        for alg_ in algorithms_test:
            print(alg_)

            # color setting
            alg_index = alg_indices[alg_]
            color = colors[alg_index]

            # initiate test function
            f_ = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

            # perform optimization
            a, b, team_names, cids = alg_(f_, N_x_, bounds_, f_eval_, has_x0=True)
            X_opt = np.array(f_.x_list[:f_eval_+1])  # needs to be capped because of COBYLA not respecting the budget

            # connect best-so-far
            best_points = []
            best_value = float('inf')  # Initialize with a high value

            for point in X_opt:
                y = f_.fun_test(point)
                if y < best_value:
                    best_value = y
                    best_points.append(point)

            best_values_x1 = [point[0] for point in best_points]
            best_values_x2 = [point[1] for point in best_points]

            ax3.plot(best_values_x1, best_values_x2, marker='o', linestyle='-', color=color, label=alg_.__name__)

            # Add starting point to the trajectory
            ax3.plot(X_opt[0,0,0], X_opt[0,1,0], marker='s', color='black', markersize=10)

            ax3.axis([x1lb, x1ub, x2lb, x2ub])
            plt.ylabel('X2', fontsize='28', fontname='Times New Roman')
            plt.xlabel('X1', fontsize='28', fontname='Times New Roman')
            plt.tick_params(axis='x', labelsize=24, labelcolor='black', labelfontfamily='Times New Roman')  # Set size and font name of x ticks
            plt.tick_params(axis='y', labelsize=24, labelcolor='black', labelfontfamily='Times New Roman')  # Set size and font name of y ticks

        # Add legend
        plt.legend(fontsize=24)

        if SafeFig == True:

            def directory_exists(directory_name):
                root_directory = os.getcwd()  # Root directory on Unix-like systems
                directory_path = os.path.join(root_directory, directory_name)
                return os.path.isdir(directory_path)

            directory_name = 'images/trajectory_plots_2D'
            if directory_exists(directory_name):

                plt.savefig(directory_name + '/{}_allin1.png'.format(f_.func_type))
            else:
                print(f"The directory '{directory_name}' does not exist in the root directory.")

            plt.close

        else:
            plt.show()


###################################################### UNUSED FUNCTIONS #############################################################
def ML4CE_uncon_graph_rel(test_res, algs_test, funcs_test, N_x_l):

    '''
    relative performance
    '''

    n_f   = len(funcs_test)
    n_p   = len(funcs_test)*len(N_x_l)
    nrow  = int(n_p/2) if n_p%2==0 else int(n_p/2)+1


    for i_alg in algs_test:
        plt.figure(figsize=(16, 25))
        for i_dim in range(len(N_x_l)):
            dim_S = 'D'+str(N_x_l[i_dim])
            for i_fun in range(n_f):
                plt.subplot(nrow,2, n_f*i_dim+i_fun+1)
                trial_  = test_res[dim_S][funcs_test[i_fun]]['all means'][str(i_alg.__name__)]
                up_     = test_res[dim_S][funcs_test[i_fun]]['all 90'][str(i_alg.__name__)]
                down_   = test_res[dim_S][funcs_test[i_fun]]['all 10'][str(i_alg.__name__)]
                medall_ = test_res[dim_S][funcs_test[i_fun]]['mean']
                lowall_ = test_res[dim_S][funcs_test[i_fun]]['q 100']
                higall_ = test_res[dim_S][funcs_test[i_fun]]['q 0']
                plt.plot(trial_, color='C'+str(i_fun), lw=3, label=str(i_alg.__name__))
                plt.plot(medall_, '--', lw=2, label='median alg')
                plt.plot(lowall_, '--', lw=2, label='best alg')
                plt.plot(higall_, '--', lw=2, label='worst alg')
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=True)
                plt.gca().fill_between(x_ax,down_, up_, color='C'+str(i_fun), alpha=0.2)
                plt.ylabel('obj value')
                plt.xlabel('iterations')
                plt.legend(loc='best')
                plt.title(funcs_test[i_fun]+' '+dim_S+' convergence plot')
                grid(True)



def ML4CE_uncon_contours(
        obj_func, 
        i_algorithm,
        bounds_plot, 
        X_opt, 
        xnew, 
        samples_number,
        func_type,
        Cons = False,
        TR_plot=False, 
        TR_l=False,
        PlotArrows=False,
        Zoom = False,
        SafeFig = False,
        ):
    
    '''
    cons_plot_l -> list of functions for the constraints for plotting
    cons_eval_l -> list of functions for the constraints for evaluation
    bounds -> list of bounds as [[x1lb,x1ub],[x2lb,x2ub]]
    '''
    # evaluate grid with vmap
    n_points = 100
    x1lb      = bounds_plot[0][0]; x1ub = bounds_plot[0][1]
    x1       = np.linspace(start=x1lb,stop=x1ub,num=n_points)
    x2lb      = bounds_plot[1][0]; x2ub = bounds_plot[1][1]
    x2       = np.linspace(start=x2lb,stop=x2ub,num=n_points)
    X1,X2      = np.meshgrid(x1,x2)

    # define plot
    plt.figure(figsize=(15, 15))
    ax3 = plt.subplot()

    # Initialize an empty array to store results
    y = np.empty((n_points, n_points))
    # Iterate over each element and apply the function
    for i in range(n_points):
        for j in range(n_points):
            y[i, j] = obj_func.fun_test(np.array([[X1[i, j], X2[i, j]]]))

    # add objective contour
    ax3.contour(X1, X2, y, 50)

    if Cons == True: 

        # evaluate list of points for constraint plot
        con_list = [obj_func.con_plot(x_i) for x_i in x2]

        # add constraints to plot
        if func_type == 'Rosenbrock_f':
            ax3.plot(con_list,x2, 'black', linewidth=3) #!!! careful here where to put x1 and x2
        else:
            ax3.plot(x1,con_list, 'black', linewidth=3) #!!! careful here where to put x1 and x2
    
    # add algorithm evaluations to plot
    # for CBO_TR we want to plot the initial samples to build the first GPs as *    
    if i_algorithm.__name__ == 'CBO_TR':
        # Calculate the length of the array
        array_length = len(X_opt[samples_number:, 0])
        # Define a color gradient from light red to dark red
        color_gradient = [(0, 0, 1 - i/array_length) for i in range(array_length)]
        # Plot each point with a color from the gradient
        for i in range(array_length):
            ax3.plot(X_opt[samples_number + i, 0], X_opt[samples_number + i, 1], marker='o', color=color_gradient[i])
        # add initial samples to plot
        ax3.plot(X_opt[:samples_number,0], X_opt[:samples_number,1], '*')

    else: 
        # Calculate the length of the array
        array_length = len(X_opt)
        # Define a color gradient from light red to dark red
        color_gradient = [(0, 0, 1 - i/array_length) for i in range(array_length)]
        # Plot each point with a color from the gradient
        for i in range(array_length):
            # ax3.plot(X_opt[i, 0], X_opt[i, 1], marker='o', color=color_gradient[i])
            ax3.plot(X_opt[i, 0], X_opt[i, 1], marker='o', color='grey')

    threshold = 2  # Define threshold here


    # connect best-so-far
    best_points = []
    best_value = float('inf')  # Initialize with a high value

    for point in X_opt:
        y = obj_func.fun_test(point)
        if y < best_value:
            best_value = y
            best_points.append(point)

    best_values_x1 = [point[0] for point in best_points]
    best_values_x2 = [point[1] for point in best_points]

    ax3.plot(best_values_x1, best_values_x2, marker='o', linestyle='-', color ='black')


    if PlotArrows == True: 

        # Extract x and y coordinates
        x_coords = X_opt[:, 0, 0]
        y_coords = X_opt[:, 1, 0]

        # Connect the points in order and add arrows
        for i in range(len(X_opt) - 1):
            plt.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], color='blue')
            # Calculate distance between consecutive points
            distance = np.sqrt((x_coords[i+1] - x_coords[i])**2 + (y_coords[i+1] - y_coords[i])**2)
            if distance > threshold:
                # Calculate midpoint
                midpoint = ((x_coords[i] + x_coords[i+1]) / 2, (y_coords[i] + y_coords[i+1]) / 2)
                # Calculate direction
                direction = (x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i])
                # Normalize direction
                direction /= np.linalg.norm(direction)
                # Add arrow
                plt.arrow(midpoint[0], midpoint[1], direction[0], direction[1], head_width=0.2, head_length=0.2, fc='blue', ec='blue')

    # add trust regions to samples in plot
    if TR_plot == True:
        for i in range(X_opt[samples_number:,:].shape[0]):
            x_pos = X_opt[samples_number+i,0]
            y_pos = X_opt[samples_number+i,1]
            # plt.text(x_pos, y_pos, str(i))
            circle1 = plt.Circle((x_pos, y_pos), radius=TR_l[i], color='black', fill=False, linestyle='--')
            ax3.add_artist(circle1)

    # Add starting point to the trajectory
    ax3.plot(X_opt[0,0,0], X_opt[0,1,0], marker = 's', color = 'black', markersize=10)

    # add final candidate to plot
    xnew = xnew.flatten()
    ax3.plot(xnew[0], xnew[1], marker = '^', color = 'black', markersize=10)

    if Zoom == True:

        # Extract x and y coordinates
        x_coords = X_opt[:, 0, 0]
        y_coords = X_opt[:, 1, 0]

        # zoom-in 
        x1lb_zoom = min(x_coords)-1
        x1ub_zoom = max(x_coords)+1
        x2lb_zoom = min(y_coords)-1
        x2ub_zoom = max(y_coords)+1

        ax3.axis([x1lb_zoom,x1ub_zoom,x2lb_zoom,x2ub_zoom])

    else:
        ax3.axis([x1lb,x1ub,x2lb,x2ub])


    if SafeFig == True:

        def directory_exists(directory_name):
            root_directory = os.getcwd()  # Root directory on Unix-like systems
            directory_path = os.path.join(root_directory, directory_name)
            return os.path.isdir(directory_path)

        directory_name = 'images/trajectory_plots_2D'
        if directory_exists(directory_name):

            plt.savefig(directory_name + '/{}_{}.png'.format(i_algorithm.__name__, obj_func.func_type))
        else:
            print(f"The directory '{directory_name}' does not exist in the root directory.")

        plt.close

    else:
        plt.show()