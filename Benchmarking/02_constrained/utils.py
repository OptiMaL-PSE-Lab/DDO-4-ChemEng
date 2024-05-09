import numpy as np
import matplotlib.pyplot as plt
from test_function_constr import *
import os   
import pickle

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


def ML4CE_con_eval(
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
            trajectories[dim_S][i_function]['all 95']    = {}
            trajectories[dim_S][i_function]['all 05']    = {}
            trajectories[dim_S][i_function]['vio_g']     = {}
            all_f_                                       = []
            randShift_l                                  = np.random.uniform(-3,3, (reps,N_x_))
            
            
            ##############
            # Algorithms #
            ##############
            info = []

            for i_algorithm in algorithms_test:
                print('== ',str(i_algorithm.__name__))
                trajectories[dim_S][i_function][str(i_algorithm.__name__)] = []
                trajectories[dim_S][i_function]['vio_g'][str(i_algorithm.__name__)] = []
                ###############
                # Repetitions #
                ###############
                for i_rep in range(reps):
                    
                    # random shift
                    x_shift_ = randShift_l[i_rep,:].reshape((N_x_,1))
                    # test function
                    t_       = Test_function(i_function, N_x_, False)
                    # algorithm
                    a, b, team_names, cids, X_opt_plot, TR_l, xnew, backtrck_l, samples_number = i_algorithm(t_, N_x_, bounds_, f_eval_, i_rep) #X_opt_plot, TR_l, xnew, backtrck_l, samples_number are for plotting
                    # post-processing
                    t_.best_f_list()                    # List of best points so far
                    t_.pad_or_truncate(f_eval_)
                    trajectories[dim_S][i_function]['vio_g'][str(i_algorithm.__name__)].extend(copy.deepcopy(t_.g_list))
                    # store result
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)].append(copy.deepcopy(t_.best_f_c))
                # statistics from each algorithm for a function    
                l_   = np.array(trajectories[dim_S][i_function][str(i_algorithm.__name__)])
                m_   = np.mean(l_, axis=0)
                q05_ = np.quantile(l_, 0.05, axis=0)
                q95_ = np.quantile(l_, 0.95, axis=0)
                trajectories[dim_S][i_function]['all means'][str(i_algorithm.__name__)] = copy.deepcopy(m_)
                trajectories[dim_S][i_function]['all 95'][str(i_algorithm.__name__)]    = copy.deepcopy(q05_)
                trajectories[dim_S][i_function]['all 05'][str(i_algorithm.__name__)]    = copy.deepcopy(q95_)
                all_f_.append(copy.deepcopy(m_))
                info.append({'alg_name': str(i_algorithm.__name__), 'team names': team_names, 'CIDs': cids})
            # statistics for all algorithms for a function       
            trajectories[dim_S][i_function]['mean']   = np.mean(all_f_, axis=0)
            trajectories[dim_S][i_function]['median'] = np.median(all_f_, axis=0)
            trajectories[dim_S][i_function]['q 0']    = np.max(all_f_, axis=0)
            trajectories[dim_S][i_function]['q 100']  = np.min(all_f_, axis=0)

    return info, trajectories