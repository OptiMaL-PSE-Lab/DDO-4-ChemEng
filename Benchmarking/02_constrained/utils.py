import numpy as np
import matplotlib.pyplot as plt
from test_function_constr import *
import os   
import pickle
from datetime import datetime
import statistics

def save_dict(target_folder, data_dict, timestamp):
    # Create a folder with current date and time
    folder_path = os.path.join(target_folder, timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # Save dictionary to file within the created folder
    file_path = os.path.join(folder_path, 'trajectories.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)


def ML4CE_con_eval(
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
        bounds_             = np.array([[-5,5] for i in range(N_x_)])

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
            trajectories[dim_S][i_function]['vio_g']     = {}
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
                trajectories[dim_S][i_function]['vio_g'][str(i_algorithm.__name__)] = []

                ###############
                # Repetitions #
                ###############
                for i_rep in range(reps):
                    
                    # random shift
                    x_shift_ = randShift_l[i_rep,:].reshape((N_x_,1))
                    # test function
                    t_       = Test_function(i_function, N_x_, True)
                    # algorithm
                    a, b, team_names, cids, X_opt_plot, TR_l, xnew, backtrck_l, samples_number = i_algorithm(t_, N_x_, bounds_, f_eval_, i_rep) #X_opt_plot, TR_l, xnew, backtrck_l, samples_number are for plotting
                    # post-processing
                    t_.best_f_list()                    # List of best points so far
                    t_.pad_or_truncate(f_eval_)
                    # store result
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)].append(copy.deepcopy(t_.best_f_c))
                    trajectories[dim_S][i_function]['f_list'][str(i_algorithm.__name__)].append(copy.deepcopy(t_.f_list))
                    trajectories[dim_S][i_function]['x_list'][str(i_algorithm.__name__)].append(copy.deepcopy(t_.x_list))
                    trajectories[dim_S][i_function]['vio_g'][str(i_algorithm.__name__)].extend(copy.deepcopy(t_.g_list))
                    # safe data in an overwriting fashion
                    if SafeData == True: save_dict(home_dir, trajectories, timestamp)
                # statistics from each algorithm for a function    
                l_   = np.array(trajectories[dim_S][i_function][str(i_algorithm.__name__)])
                m_   = np.mean(l_, axis=0)
                q10_ = np.quantile(l_, 0.05, axis=0)
                q90_ = np.quantile(l_, 0.95, axis=0)
                trajectories[dim_S][i_function]['all means'][str(i_algorithm.__name__)] = copy.deepcopy(m_)
                trajectories[dim_S][i_function]['all 90'][str(i_algorithm.__name__)]    = copy.deepcopy(q10_)
                trajectories[dim_S][i_function]['all 10'][str(i_algorithm.__name__)]    = copy.deepcopy(q90_)
                all_f_.append(copy.deepcopy(l_))
                info.append({'alg_name': str(i_algorithm.__name__), 'team names': team_names, 'CIDs': cids})
                # safe data in an overwriting fashion
                if SafeData == True: save_dict(home_dir, trajectories, timestamp)
            # statistics for all algorithms for a function       
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
    


def ML4CE_con_table(
        trajectories, 
        algs_test,
        funcs_test,
        N_x_l,
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


    ### DIMENSION ###
    for i_dim in range(len(N_x_l)): 
        dim_S           = 'D'+str(N_x_l[i_dim])
        alg_perf[dim_S] = {}

        ### ALGORITHM ###
        for i_alg in algs_test:
            print('==  ',str(i_alg.__name__),' ==')
            alg_perf[dim_S][str(i_alg.__name__)] = {}

            ### FUNCTION ###
            for i_fun in range(n_f):
                # retrive performance
                medall_ = trajectories[dim_S][funcs_test[i_fun]]['mean']
                trial_  = trajectories[dim_S][funcs_test[i_fun]]['all means'][str(i_alg.__name__)]
                lowall_ = trajectories[dim_S][funcs_test[i_fun]]['q 100']
                higall_ = trajectories[dim_S][funcs_test[i_fun]]['q 0']
                # score performance
                # since we're providing starting points to the algorithms we can use the entire trajectory 
                # instead of cutting the inital evaluations out as we do in the unconstrained benchmarking
                perf_ = ( (higall_[:] - trial_[:])
                        /(higall_[:] - lowall_[:]) ) 
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

    # Calculate row-wise averages of all columns
    row_wise_averages_first = np.mean(combined_matrix_norm_reshape, axis=2)

    # Expand dimensions to allow broadcasting
    row_wise_averages_first_expanded = row_wise_averages_first[:, :, np.newaxis]

    # Add the new column to each matrix
    result1 = np.concatenate((combined_matrix_norm_reshape, row_wise_averages_first_expanded), axis=2)

    ######## The below are taken out since we're only operating in D2 #########

    # # Calculate row-wise averages of the second two original columns
    # row_wise_averages_second = np.mean(combined_matrix_norm_reshape[:, :, len(multim):], axis=2)

    # # Expand dimensions to allow broadcasting
    # row_wise_averages_second_expanded = row_wise_averages_second[:, :, np.newaxis]

    # # Add the new column to each matrix
    # result2 = np.concatenate((result1, row_wise_averages_second_expanded), axis=2)

    # # Calculate row-wise averages of the two new columns
    # row_wise_averages_new = np.mean(result2[:, :, :6], axis=2)

    # # Expand dimensions to allow broadcasting
    # row_wise_averages_new_expanded = row_wise_averages_new[:, :, np.newaxis]

    # # Add the new column to each matrix
    # result3 = np.concatenate((result2, row_wise_averages_new_expanded), axis=2)

    ### calculate the final row ###
    # Calculate column-wise averages across each matrix
    column_wise_averages = np.mean(result1, axis=1)

    # Stack the column-wise averages as an additional row to each matrix
    arr_with_avg_row = np.concatenate((result1, column_wise_averages[:, np.newaxis, :]), axis=1)

    # pure test results # shape (num_alg, num_dim + 1, num_fun + 1)
    test_res = np.around(arr_with_avg_row, decimals=2)



    ### Now attach the constraint violation ###
    # computing table of results
    n_f       = len(funcs_test)
    vio_dict = {}

    ### DIMENSION ###
    for i_dim in range(len(N_x_l)): 
        dim_S           = 'D'+str(N_x_l[i_dim])

        vio_dict[dim_S] = {}

        ### ALGORITHM ###
        for i_alg in algs_test:

            vio_dict[dim_S][str(i_alg.__name__)] = {}

            ### FUNCTION ###
            for i_fun in range(n_f):

                vio_list = trajectories[dim_S][funcs_test[i_fun]]['vio_g'][str(i_alg.__name__)]
                vio_dict[dim_S][str(i_alg.__name__)][funcs_test[i_fun]] = {}
                vio_dict[dim_S][str(i_alg.__name__)][funcs_test[i_fun]]['vio_g'] = {}

                # mean violation
                valid_values = [value for value in vio_list if value is not None]

                if valid_values:
                    mean_vio = statistics.mean(valid_values)
                else:
                    mean_vio = 0 

                vio_dict[dim_S][str(i_alg.__name__)][funcs_test[i_fun]]['vio_g']['mean_vio'] = mean_vio
                
                # mean no of constrain satisfaction
                vio_dict[dim_S][str(i_alg.__name__)][funcs_test[i_fun]]['vio_g']['perc_nonvio']=(vio_list.count(None) / len(vio_list)) * 100



    return test_res, vio_dict


def ML4CE_con_table_plot(array, functions_test, algorithms_test, N_x_l, SafeFig = False):
    
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

            directory_name = 'images/tables'
            if directory_exists(directory_name):

                plt.savefig(directory_name + '/{}.png'.format(algorithms_test[i].__name__))
            else:
                print(f"The directory '{directory_name}' does not exist in the root directory.")
        
            plt.close()

        else:
            plt.show()


def ML4CE_con_contours(
        obj_func, 
        i_algorithm,
        bounds_plot, 
        X_opt, 
        xnew, 
        samples_number,
        func_type,
        TR_plot=False, 
        TR_l=False,
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
            ax3.plot(X_opt[i, 0], X_opt[i, 1], marker='o', color=color_gradient[i])

    # add trust regions to samples in plot
    if TR_plot == True:
        for i in range(X_opt[samples_number:,:].shape[0]):
            x_pos = X_opt[samples_number+i,0]
            y_pos = X_opt[samples_number+i,1]
            # plt.text(x_pos, y_pos, str(i))
            circle1 = plt.Circle((x_pos, y_pos), radius=TR_l[i], color='black', fill=False, linestyle='--')
            ax3.add_artist(circle1)

    # add final candidate to plot
    xnew = xnew.flatten()
    ax3.plot(xnew[0], xnew[1], 'yo')
    ax3.set_title('Contour plot')
    ax3.axis([x1lb,x1ub,x2lb,x2ub])

    if SafeFig == True:

        def directory_exists(directory_name):
            root_directory = os.getcwd()  # Root directory on Unix-like systems
            print('root_directory: ' + root_directory)
            directory_path = os.path.join(root_directory, directory_name)
            return os.path.isdir(directory_path)

        directory_name = 'images/trajectory_plots_2D'
        if directory_exists(directory_name):

            plt.savefig(directory_name + '/{}_{}.png'.format(i_algorithm.__name__, obj_func.func_type))
        else:
            print(f"The directory '{directory_name}' does not exist in the root directory.")
            os.mkdir(directory_name)
            plt.savefig(directory_name + '/{}_{}.png'.format(i_algorithm.__name__, obj_func.func_type))

        plt.close()