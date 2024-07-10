import numpy as np
import matplotlib.pyplot as plt
from test_function_constr import *
import os   
import pickle
from datetime import datetime
import statistics

#########################################
####### Starting point generator ########
#########################################
import numpy as np

def random_points_in_circle(n, radius, center, seed=None):
    """
    Generates n random points within a circle of given radius around a starting point.
    
    Parameters:
    - n: int, number of random points to generate
    - radius: float, radius of the circle
    - center: numpy array of shape (1,2), center of the circle
    - seed: int or None, random seed for reproducibility (default is None)
    
    Returns:
    - points: numpy array of shape (n,2), containing the generated points
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random angles between 0 and 2*pi
    angles = np.random.uniform(0, 2 * np.pi, n)
    
    # Generate random radii with uniform distribution
    radii = radius * np.sqrt(np.random.uniform(0, 1, n))
    
    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # Combine x and y into a (n, 2) array
    points = np.vstack((x, y)).T
    
    # Translate points to be around the given center

    points += center
    
    return points



###########################################
################ DATA SAVING ##############
###########################################
def lab_journal(folder_path, timestamp, file_name, SafeData = False):
    if SafeData: 
        user_input = input("Enter the text you want to save: ").strip()

        if user_input:
            if not os.path.exists(folder_path+'/'+timestamp):
                os.makedirs(folder_path+'/'+timestamp)

            file_path = os.path.join(folder_path, timestamp, file_name)

            with open(file_path, 'w') as file:
                file.write(user_input)

            print(f"Text saved to '{file_name}' in folder '{folder_path}'.")
        else:
            print("Please enter some text before saving.")
    else:return

def directory_exists(directory_name):
    root_directory = os.getcwd()  # Root directory on Unix-like systems
    directory_path = os.path.join(root_directory, directory_name)
    return os.path.isdir(directory_path)

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
        bounds = None,
        SafeData = False,
        ):

    trajectories = {}
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save Info
    lab_journal(home_dir, timestamp, file_name='INFO', SafeData=SafeData)

    #############
    # Dimension #
    #############
    for i_run in range(len(N_x_l)):
        N_x_                = N_x_l[i_run]
        f_eval_             = f_eval_l[i_run]
        dim_S               = 'D'+str(N_x_)
        trajectories[dim_S] = {}

        if bounds is None:
            bounds_             = np.array([[-5,5] for i in range(N_x_)])

            # bounds  = np.array([[4., 7.], [70., 100.]])
        
        else:
            print('im here')
            bounds_ = bounds

        #############
        # Functions #
        #############
        for i_function in functions_test:
            print('===================== ',i_function,'D'+str(N_x_))
            trajectories[dim_S][i_function]              = {}
            trajectories[dim_S][i_function]['all means'] = {}
            trajectories[dim_S][i_function]['all means g'] = {}
            trajectories[dim_S][i_function]['all 90']    = {}
            trajectories[dim_S][i_function]['all 10']    = {}
            trajectories[dim_S][i_function]['all 90 g']    = {}
            trajectories[dim_S][i_function]['all 10 g']    = {}
            trajectories[dim_S][i_function]['f_list']    = {}
            trajectories[dim_S][i_function]['x_list']    = {}
            trajectories[dim_S][i_function]['vio_g']     = {}
            trajectories[dim_S][i_function]['vio_g_values']     = {}
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
                trajectories[dim_S][i_function]['vio_g_values'][str(i_algorithm.__name__)] = []

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
                    X_all_pre = t_.x_list
                    X_all = []
                    # make sure to even-out the trajectory lengths
                    if len(X_all_pre)<f_eval_: 
                        x_last = X_all_pre[-1]
                        X_all = X_all_pre + [x_last]*(f_eval_ - len(X_all_pre))
                    elif len(X_all_pre)>f_eval_:
                        x_last = X_all_pre[:f_eval_][-1]
                        X_all = X_all_pre[:f_eval_] + [x_last]*(f_eval_ - len(X_all_pre[:f_eval_]))
                    else: X_all = X_all_pre
                    X_all = np.array(X_all)
                    # capture constraint handling
                    trajectories[dim_S][i_function]['vio_g_values'][str(i_algorithm.__name__)].append([-t_.con_test(x) for x in X_all])
                    # re-test for feasibility
                    f_feas = Test_function(i_function, N_x_, True)
                    if i_function == 'WO_f':
                        X_feas = [x for x in X_all if f_feas.WO_con1_test(x) > 0 and f_feas.WO_con2_test(x) > 0]
                    else:
                        X_feas = [x for x in X_all if f_feas.con_test(x) > 0]
                    # produce function values for ranking
                    for x in X_feas: f_feas.fun_test(x)
                    # get best evaluations
                    f_feas.best_f_list()
                    f_feas.pad_or_truncate(f_eval_)
                    # store result
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)].append(copy.deepcopy(f_feas.best_f_c))
                    trajectories[dim_S][i_function]['f_list'][str(i_algorithm.__name__)].append(copy.deepcopy(f_feas.f_list))
                    trajectories[dim_S][i_function]['x_list'][str(i_algorithm.__name__)].append(copy.deepcopy(f_feas.x_list))
                    trajectories[dim_S][i_function]['vio_g'][str(i_algorithm.__name__)].extend(copy.deepcopy(t_.g_list))
                    # safe data in an overwriting fashion
                    if SafeData == True: save_dict(home_dir, trajectories, timestamp)
                # statistics from each algorithm for a function    
                l_   = np.array(trajectories[dim_S][i_function][str(i_algorithm.__name__)])
                l_g_ = np.array(trajectories[dim_S][i_function]['vio_g_values'][str(i_algorithm.__name__)]).reshape(l_.shape)
                m_   = np.mean(l_, axis=0)
                m_g_  = np.mean(l_g_, axis=0)
                q10_ = np.quantile(l_, 0.10, axis=0)
                q10_g_ = np.quantile(l_g_, 0.10, axis=0)
                q90_ = np.quantile(l_, 0.90, axis=0)
                q90_g_ = np.quantile(l_g_, 0.90, axis=0)
                trajectories[dim_S][i_function]['all means'][str(i_algorithm.__name__)] = copy.deepcopy(m_)
                trajectories[dim_S][i_function]['all means g'][str(i_algorithm.__name__)] = copy.deepcopy(m_g_)
                trajectories[dim_S][i_function]['all 90'][str(i_algorithm.__name__)]    = copy.deepcopy(q10_)
                trajectories[dim_S][i_function]['all 10'][str(i_algorithm.__name__)]    = copy.deepcopy(q90_)
                trajectories[dim_S][i_function]['all 90 g'][str(i_algorithm.__name__)]    = copy.deepcopy(q10_g_)
                trajectories[dim_S][i_function]['all 10 g'][str(i_algorithm.__name__)]    = copy.deepcopy(q90_g_)
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
        
        # Safe Data
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


def ML4CE_con_table_plot(array, functions_test, algorithms_test, N_x_l, home_dir, timestamp, SafeFig = False):
    
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

def ML4CE_con_graph_abs(test_res, algs_test, funcs_test, N_x_l, home_dir, timestamp, SafeFig=False):

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
                plt.plot(trial_, '-o', color=color, linestyle=line_style, lw=3, label=str(i_alg.__name__), markersize = 10)
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=False)
                plt.gca().fill_between(x_ax,down_, up_, color=color, alpha=0.2)

                #### NO VERTICAL LINE IN CONSTRAINT SINCE WE MEASURE ALL REPETITIONS ################
                # # Calculate the position of the vertical line based on the length of the trajectory
                # length = len(down_)
                # if length == 20:
                #     vline_pos = 5
                # elif length == 50:
                #     vline_pos = 10
                # elif length == 100:
                #     vline_pos = 15
                # else:
                #     vline_pos = None  # Or set a default value if needed

                # # Add the vertical line if a valid position is calculated
                # if vline_pos is not None:
                #     plt.axvline(x=vline_pos, color='black', linestyle='--', linewidth=2)

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


def ML4CE_con_graph_abs_g(test_res, algs_test, funcs_test, N_x_l, home_dir, timestamp, SafeFig=False):

    colors = plt.cm.tab10(np.linspace(0, 1, len(algs_test)))
    line_styles = ['-', '--', '-.', ':']  
    alg_indices = {alg: i for i, alg in enumerate(algs_test)}

    n_f = len(funcs_test)

    for i_dim in range(len(N_x_l)):
        dim_S = 'D' + str(N_x_l[i_dim])
        for i_fun in range(n_f):
            plt.figure(figsize=(15, 15))
            for i_alg in algs_test:
                trial_ = test_res[dim_S][funcs_test[i_fun]]['all means g'][str(i_alg.__name__)]
                up_     = test_res[dim_S][funcs_test[i_fun]]['all 90 g'][str(i_alg.__name__)]
                down_   = test_res[dim_S][funcs_test[i_fun]]['all 10 g'][str(i_alg.__name__)]
                alg_index = alg_indices[i_alg]  # Get the index of the algorithm
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]  # Use modulo to cycle through line styles
                plt.plot(trial_, '-o', color=color, linestyle=line_style,  lw=3, label=str(i_alg.__name__), markersize = 10)
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=False)
                plt.gca().fill_between(x_ax,down_, up_, color=color, alpha=0.2)

                # Add vertical line at x=0
                plt.axhline(y=0, color='black', linestyle='--', linewidth=2)

                #### NO VERTICAL LINE IN CONSTRAINT SINCE WE MEASURE ALL REPETITIONS ################
                # # Calculate the position of the vertical line based on the length of the trajectory
                # length = len(down_)
                # if length == 20:
                #     vline_pos = 5
                # elif length == 50:
                #     vline_pos = 10
                # elif length == 100:
                #     vline_pos = 15
                # else:
                #     vline_pos = None  # Or set a default value if needed

                # # Add the vertical line if a valid position is calculated
                # if vline_pos is not None:
                #     plt.axvline(x=vline_pos, color='black', linestyle='--', linewidth=2)

                # # Setting x-axis ticks to integer values starting from 0 and showing every 5th tick
                tick_positions = np.arange(0, len(down_), 5)
                if (len(down_) - 1) % 5 != 0:  # If the last position is not already included
                    tick_positions = np.append(tick_positions, len(down_) - 1)
                tick_labels = np.arange(0, len(down_), 5)
                if len(tick_labels) < len(tick_positions):
                    tick_labels = np.append(tick_labels, len(down_) - 1)

                plt.xticks(tick_positions, tick_labels, fontsize=24, fontname='Times New Roman')

            plt.ylabel('constraint value', fontsize = '28', fontname='Times New Roman')
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

                    plt.savefig(directory_name + '/{}_{}_g_1D.png'.format(dim_S, funcs_test[i_fun]))
                else:
                    print(f"The directory '{directory_name}' does not exist in the root directory.")
                    os.mkdir(directory_name)
                    plt.savefig(directory_name + '/{}_{}_g_1D.png'.format(dim_S, funcs_test[i_fun]))

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
            y[i, j] = obj_func.fun_test(np.array([[X1[i, j], X2[i, j]]]).flatten())

    # add objective contour
    ax3.contour(X1, X2, y, 50)

    if obj_func.func_type == 'WO_f':
        con_list1 = [obj_func.WO_con1_plot(x_i) for x_i in x1]
        con_list2 = [obj_func.WO_con2_plot(x_i) for x_i in x1]

    else:
        # evaluate list of points for constraint plot
        con_list = [obj_func.con_plot(x_i) for x_i in x2]

    

    # add constraints to plot
    if func_type == 'Rosenbrock_f':
        ax3.plot(con_list,x2, 'black', linewidth=3) #!!! careful here where to put x1 and x2
    elif func_type == 'WO_f':
        ax3.plot(x1,con_list1, 'black', linewidth=3)
        ax3.plot(x1,con_list2, 'black', linewidth=3)
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


import numpy as np
import matplotlib.pyplot as plt

def ML4CE_con_contour_allin1(functions_test, algorithms_test, N_x_, x_shift_origin, bounds_, i_rep, bounds_plot, directory_name, SafeFig=False):
    
    # track time
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # create lab_journal
    lab_journal(directory_name, timestamp,file_name='INFO', SafeData=SafeFig)

    # data inputs
    f_eval_ = 40 # trajectory length (= evaluation budget) --> Keep in mind that BO uses 10 datapoints to build the model when plotting
    track_x = True
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms_test)))
    alg_indices = {alg: i for i, alg in enumerate(algorithms_test)}

    # perform plotting
    for fun_ in functions_test:
        f_contour = Test_function(fun_, N_x_, track_x)

        ####################################
        ### CONTOUR AND CONSTRAINT LINES ###
        ####################################
        # evaluate grid with vmap
        n_points = 50
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
                y[i, j] = f_contour.fun_test(np.array([[X1[i, j], X2[i, j]]]).flatten())

        # add objective contour
        ax3.contour(X1, X2, y, 50)

        if fun_ == 'WO_f':
            con_list1 = [f_contour.WO_con1_plot(x_i) for x_i in x1]
            con_list2 = [f_contour.WO_con2_plot(x_i) for x_i in x1]

        else:
            # evaluate list of points for constraint plot
            con_list = [f_contour.con_plot(x_i) for x_i in x2]

        # add constraints to plot
        if fun_ == 'Rosenbrock_f':
            ax3.plot(con_list,x2, 'black', linewidth=3) #!!! careful here where to put x1 and x2
        elif fun_ == 'WO_f':
            ax3.plot(x1,con_list1, 'black', linewidth=3)
            ax3.plot(x1,con_list2, 'black', linewidth=3)
        else:
            ax3.plot(x1,con_list, 'black', linewidth=3) #!!! careful here where to put x1 and x2
        
        ax3.axis([x1lb,x1ub,x2lb,x2ub])

        ############################
        ### PERFORM OPTIMIZATION ###
        ############################
        for alg_ in algorithms_test:
            print(alg_)

            # color setting
            alg_index = alg_indices[alg_]
            color = colors[alg_index]

            # initiate test function
            f_plot = Test_function(fun_, N_x_, track_x)

            # perform optimization
            a, b, team_names, cids, c, d, e, f, g = alg_(f_plot, N_x_, bounds_, f_eval_, i_rep)

            # get and plot all evaluations
            X_all = np.array(f_plot.x_list)
            ax3.scatter(X_all[:,0], X_all[:,1], marker='o', color=color)

            # re-test for feasibility
            f_feas = Test_function(fun_, N_x_, track_x)

            if fun_ == 'WO_f':

                X_feas = [x for x in X_all if f_feas.WO_con1_test(x) > 0 and f_feas.WO_con2_test(x) > 0]
            else:

                X_feas = [x for x in X_all if f_feas.con_test(x) > 0]

            # produce function values for ranking
            for x in X_feas: f_feas.fun_test(x)

            # get best evaluations
            f_feas.best_f_list()

            # plot
            best_values_x1 = [point[0] for point in f_feas.best_x]
            best_values_x2 = [point[1] for point in f_feas.best_x]
            ax3.plot(best_values_x1, best_values_x2, marker='o', linestyle='-', color=color, label=alg_.__name__)

            # Add starting point to the trajectory
            try: ax3.plot(X_all[0,0], X_all[0,1], marker='s', color='black', markersize=10)
            except: ax3.plot(X_all[0,0,0], X_all[0,1,0], marker='s', color='black', markersize=10)

            ax3.axis([x1lb, x1ub, x2lb, x2ub])
            plt.ylabel('X2', fontsize='28', fontname='Times New Roman')
            plt.xlabel('X1', fontsize='28', fontname='Times New Roman')
            plt.tick_params(axis='x', labelsize=24, labelcolor='black', labelfontfamily='Times New Roman')  # Set size and font name of x ticks
            plt.tick_params(axis='y', labelsize=24, labelcolor='black', labelfontfamily='Times New Roman')  # Set size and font name of y ticks

        # Add legend
        plt.legend(fontsize=24)

        # Data Saving
        if SafeFig == True:
            if directory_exists(directory_name + '/' + timestamp):
                plt.savefig(directory_name + '/' + timestamp + '/{}_allin1.png'.format(fun_))
            else:
                print(f"The directory '{directory_name}' does not exist in the root directory. Creating directory.")
                os.mkdir(directory_name+ '/' + timestamp)
                plt.savefig(directory_name + '/' + timestamp + '/{}_allin1.png'.format(fun_))
            plt.close

        else:
            plt.show()
    
