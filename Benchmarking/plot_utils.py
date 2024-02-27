# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:30:18 2023

@author: adelrioc
"""

#########################
# import plot libraries #
#########################
 
import matplotlib.pyplot as plt
from pylab import grid
import pylab
import numpy as np
import copy

#################
# plot function #
#################

def plot_performance(test_res, algs_test, funcs_test, folderplt, N_x_l):
    '''
    Parameters
    ----------
    list_of_results : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    # plot directory:
        
    alg_perf = {}
    
    n_f = len(funcs_test)
    # usedn in plots
    columns = funcs_test + ['Unimodal','Multimodal','All']
    multim  = ['Rastringin_f', 'Ackely_f']
    unim    = ['Rosenbrock_f', 'Levy_f']
    rows    = ['D2', 'D5', 'D10', 'All']

    
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
                medall_ = test_res[dim_S][funcs_test[i_fun]]['mean']
                trial_  = test_res[dim_S][funcs_test[i_fun]]['all means'][str(i_alg.__name__)]
                lowall_ = test_res[dim_S][funcs_test[i_fun]]['q 100']
                higall_ = test_res[dim_S][funcs_test[i_fun]]['q 0']
                # score performance
                perf_                                = (higall_ - trial_)/(higall_ - lowall_)
                alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i_fun]] = copy.deepcopy(np.sum(perf_)/len(perf_))
                #print(funcs_test[i_fun], '  ',np.sum(perf_)/len(perf_))     

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
                up_     = test_res[dim_S][funcs_test[i_fun]]['all 95'][str(i_alg.__name__)]
                down_   = test_res[dim_S][funcs_test[i_fun]]['all 05'][str(i_alg.__name__)]
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
            
        # Plot bars and create text labels for the table
        cell_text = []
        n_rows    = len(rows)
        # for different dimensions
        for row in  range(len(N_x_l)):
            dim_S = 'D'+str(N_x_l[row])
            r_    = [alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i]] for i in range(n_f)]
            m_    = [alg_perf[dim_S][str(i_alg.__name__)][j] for j in multim]
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
             
        #plt.figure(figsize=(16, 10))    
        the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      #rowColours=colors,
                      colLabels=columns,
                      loc='bottom', bbox=[0.0,-0.45,1,.28])  
        #plt.subplots_adjust(bottom=0.8)
        plt.tight_layout()
        # save file    
        if folderplt !=False:
            plt.savefig(folderplt+str(i_alg.__name__)+'_ConvergencePlot'+'.png', dpi=150,
                        bbox_inches='tight')
        plt.close()

    
###########
# Summary #
###########

def summarize_performance(test_res, algs_test, funcs_test, N_x_l):
    
    alg_perf = {}
    
    n_f = len(funcs_test)
    # usedn in plots
    columns = funcs_test + ['Unimodal','Multimodal','All']
    multim  = ['Rastringin_f', 'Ackely_f']
    unim    = ['Rosenbrock_f', 'Levy_f']
    rows    = ['D2', 'D5', 'D10', 'All']
    
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
                medall_ = test_res[dim_S][funcs_test[i_fun]]['mean']
                trial_  = test_res[dim_S][funcs_test[i_fun]]['all means'][str(i_alg.__name__)]
                lowall_ = test_res[dim_S][funcs_test[i_fun]]['q 0']
                higall_ = test_res[dim_S][funcs_test[i_fun]]['q 100']
                # score performance
                perf_                                = (higall_ - trial_)/(higall_ - lowall_)
                alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i_fun]] = copy.deepcopy(np.sum(perf_)/len(perf_))
                #print(funcs_test[i_fun], '  ',np.sum(perf_)/len(perf_))                                         
            
        
    # Plot bars and create text labels for the table
    cell_text = []
    n_rows    = len(rows)
    # for different dimensions
    for row in  range(len(N_x_l)):
        dim_S = 'D'+str(N_x_l[i_dim])
        r_    = [alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i]] for i in range(n_f)]
        m_    = [alg_perf[dim_S][str(i_alg.__name__)][j] for j in multim]
        u_    = [alg_perf[dim_S][str(i_alg.__name__)][k] for k in unim]
        c_    = r_ + [sum(m_)/len(m_)] + [sum(u_)/len(u_)] + [sum(r_)/len(r_)]
        c_    = [round(c_i,2) for c_i in c_]
        cell_text.append(c_)
    # for all dimensions the below need fixing
    r_    = [sum([alg_perf['D'+str(N_x_l[j])][str(i_alg.__name__)][funcs_test[i]] 
                  for j in range(len(N_x_l))]) 
             for i in range(n_f)]
    m_    = [sum([alg_perf['D'+str(N_x_l[j])][str(i_alg.__name__)][jj] 
                  for j in range(len(N_x_l))]) 
             for jj in multim]
    u_    = [sum([alg_perf['D'+str(N_x_l[j])][str(i_alg.__name__)][k] 
                  for j in range(len(N_x_l))]) 
             for k in unim]
    c_    = r_ + [sum(m_)/len(m_)] + [sum(u_)/len(u_)] + [sum(r_)/len(r_)]
    c_    = [round(c_i,2) for c_i in c_]
    cell_text.append(c_)
         
    plt.figure(figsize=(16, 10))    
    the_table = plt.table(cellText=cell_text,
                  rowLabels=rows,
                  #rowColours=colors,
                  colLabels=columns,
                  loc='bottom')
    plt.show()
            
def contour_plot_f(f, many_points, few_points, 
                   x_min=-15,x_max=15,y_min=-15,y_max=15,
                   cmap_='viridis'):
    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    xy     = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
    
    # Compute function values at each point on the grid
    # TODO create for loop here to evaluate all funcs
    zz = []
    for z_i in range(100*100):
        zz.append(f(xy[z_i,:]))
    zz = np.array(zz).reshape(100,100)
    xy = xy.reshape(100,100,2)
    
    # Plot the contour lines
    plt.figure()
    plt.contourf(xy[:,:,0], xy[:,:,1], zz, cmap=cmap_)  # Use contourf for filled contours
    plt.colorbar()  # Add a colorbar for reference
    plt.scatter(many_points[:, 0], many_points[:, 1], marker='*', label='GP samples')  # Plot data points
    plt.scatter(few_points[:, 0], few_points[:, 1], color='red', label='f samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour Plot of Function f(x, y)')
    plt.legend()
    plt.show()  
    
    
    
    
    
    
    
    