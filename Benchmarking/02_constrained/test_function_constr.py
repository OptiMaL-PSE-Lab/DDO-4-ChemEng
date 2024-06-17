# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:00:56 2023

@author: adelrioc
"""

import numpy as np
import copy
from scipy.optimize import minimize_scalar

##########################################
### threshold for constraint violation ###
##########################################

vio = 0.1

#######################
# calculate distances #
#######################

def eucl_dist(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def closest_point_on_constraint(x_vio, con):
    min_distance = float('inf')
    closest_point = None
    for x1_g in np.linspace(-5, 5, 10000):               # Adjust the range and granularity as needed #TODO this needs to be updated to bounds_ 
        x2_g = con(x1_g)                                 # calculate the constraint value
        distance = eucl_dist(x_vio, (x1_g, x2_g))
        if distance < min_distance:
            min_distance = distance
            closest_point = (x1_g, x2_g)
    return closest_point


class Test_function:

    '''
    This is the test function class, that contains
    - the function itself
    - the constraints per function
    - the starting points per function
    '''
    
    #################################
    # --- initializing function --- #
    #################################    
    def __init__(
            self, 
            func_type, 
            n_x, 
            track_x,
            ):
        
        # initialize lists
        self.f_list = []; 
        self.x_list = []; 
        self.g_list = []; 

        # function specifications
        self.func_type = func_type
        self.n_x       = n_x
        self.track_x   = track_x

        if self.func_type == 'Rosenbrock_f':
            self.x0 = np.array(
                ([[-3.5],[4]],
                [[-4.7],[0]],
                [[-1],[1]]),
                )

            self.init_points = np.array([[ #TODO these are manually copied random points and need to be automated
                [-3.3530743413133854, 3.73574969572994],
                [-3.695999175468267, 4.298600636679204],
                [-3.324485345153229, 4.34397331429392],
                [-3.6551956617560606, 4.445719043701324],
                [-3.2894861583353707, 4.372417455324243],
                [-3.1358366982310306, 3.7318957927275074],
                [-3.3750653613706514, 4.031628585876453],
                [-3.6560822903968844, 4.2695814211741565],
                [-3.402599951172502, 4.290894965399081],
                [-3.58554000313634, 3.8960851498181146]
                ],
                [
                [-4.383496912801828, -0.0013634667911379683],
                [-4.872829217553934, 0.07939168757488668],
                [-4.596064847385586, -0.00678504534014468],
                [-4.3628806545158065, 0.13337499704552347],
                [-4.900695899797675, 0.32541746346155087],
                [-4.6986158824203, 0.42788750770802386],
                [-4.855153739318994, 0.07256667216710011],
                [-4.371620789116707, -0.2625845865624066],
                [-4.417138114611893, 0.1138350903311236],
                [-4.968454704380232, -0.011770250298933527],
                ],
                [[-1.3859377125730643, 0.8502460012604153],
                [-0.9094882325785648, 1.0522645964081918],
                [-0.8257802160551404, 1.2511523667547042],
                [-1.073769078982123, 1.437282168432578],
                [-1.0035681771364549, 0.7201507013205384],
                [-1.0956277155456018, 1.2812525332988598],
                [-0.7949204439192279, 0.5982623762449268],
                [-0.6452272945520198, 1.2229489570716696],
                [-0.8448267653830063, 0.875666652279921],
                [-0.5734729013792419, 1.2194629193024646]],

                ])
                            
            
        if self.func_type == 'Antonio_f':
            self.x0 = np.array(
                ([[-4],[-4]],
                [[-3],[1]],
                [[1],[4]]),
            )

            # needs to be updated for higher dimensions # !!!!!

            self.init_points = np.array([[
                [-4.186356190440544, -3.8345781614272596],
                [-4.133735443252803, -3.9070889874067998],
                [-4.2554621696650266, -3.839167539044557],
                [-3.6991499017924774, -4.067343702881785],
                [-4.2204588351029315, -3.9322351439533074],
                [-3.8665994738407723, -4.294931903565754],
                [-3.5352717073354922, -3.93764772917438],
                [-4.388286226588492, -3.971974085341454],
                [-3.8820488343028465, -4.3689041148478545],
                [-4.2178130348809955, -3.603861244697474]
                ],
                [
                [-3.020792358572854, 1.093699653066452],
                [-3.434556216336288, 1.2060580959472942],
                [-3.1610557040339065, 1.219792127067329],
                [-4.3628806545158065, 1.4144437341072003],
                [-3.1908791866381243, 0.6233448076862496],
                [-3.3527141888662424, 1.1464401407819036],
                [-2.648462121006375, 0.9002924215843959],
                [-2.961973468399645, 0.8925521968287324],
                [-3.3935600071374625, 0.8615225940317944],
                [-3.364708378971091, 1.2354171389730733],
                ],
                [
                [0.5881213509550786, 4.068118384332777],
                [1.0347934616792243, 4.345938241368554],
                [1.351421611414327, 3.925567990718291],
                [1.4884654332087441, 4.092294655773182],
                [0.982431117780894, 3.565937812134032],
                [0.6798258164392602, 4.189425695691275],
                [0.8496647453266832, 3.550606273479979],
                [1.1510961542739127, 4.1176517876307726],
                [1.261537220618603, 3.755053800587087],
                [1.045742654831773, 3.52680512392919]
                ],
                ])



        if self.func_type == 'Matyas_f':
            self.x0 = np.array(
                ([[-4],[-4]],
                [[-4],[4]],
                [[-1],[2]]),
            )
            self.init_points = np.array([[
                [-4.186356190440544, -3.8345781614272596],
                [-4.133735443252803, -3.9070889874067998],
                [-4.2554621696650266, -3.839167539044557],
                [-3.6991499017924774, -4.067343702881785],
                [-4.2204588351029315, -3.9322351439533074],
                [-3.8665994738407723, -4.294931903565754],
                [-3.5352717073354922, -3.93764772917438],
                [-4.388286226588492, -3.971974085341454],
                [-3.8820488343028465, -4.3689041148478545],
                [-4.2178130348809955, -3.603861244697474]
                ],
                [
                [-4.054352730639594, 4.182246644507763],
                [-4.276605526658412, 3.981178432248542],
                [-4.243212994014822, 3.823595983075548],
                [-3.6794393435722803, 3.786178023555208],
                [-3.8572039094958086, 4.339100095213703],
                [-4.090340379322962, 4.2346685302075535],
                [-3.9468489911109974, 4.364129841768841],
                [-3.7862484083265913, 3.68260975746392],
                [-3.8830869545215307, 3.891982741956224],
                [-3.7004486478680696, 4.233162414547007],
                ],
                [
                [-1.321628422969357, 1.9201187760779104],
                [-1.2714112905017754, 2.233366868151361],
                [-0.5741901541251668, 1.837363750589947],
                [-1.3255530396389932, 1.9700450143758137],
                [-0.7220027564531106, 1.7740214800327365],
                [-1.167031434416019, 1.7708098701921013],
                [-0.9117065742157653, 2.398603898563745],
                [-0.9660050845056297, 1.9757924790694936],
                [-0.79155403650756, 1.5855211177502666],
                [-1.4855989434252028, 1.9307039921013924]
                ],
                ])

            
    ################    
    # run function #
    ################
    
    def fun_test(self, x):

        #######################
        # Rosenbrock_function #
        #######################

        if self.func_type == 'Rosenbrock_f':
            '''
            Test function: Rosenbrock function (http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html)
            Note: Website does not work anymore (checked Feb_24_2024), this one does: https://www.sfu.ca/~ssurjano/rosen.html
            Note:
                works only for vectors
            '''

            x = np.array(x)
            x = x.reshape((self.n_x,1))
            z = (1.-x).T@(1.-x) + 100*(x[1:]-x[:-1]**2).T@(x[1:]-x[:-1]**2)
            
            # track f
            self.f_list.append(z[0,0]) 
            if self.track_x:
                self.x_list.append(x)
                
            # return objective        
            return z[0,0]
        
        ####################
        # Antonio_function #
        ####################

        if self.func_type == 'Antonio_f':

            ### OLD ###
            # x = np.array(x)
            # x = x.reshape((self.n_x,1))

            # z = x[0]**2 + 3*x[1]**2 + 0.9*x[0]*x[1]


            ### NEW ###
            a = 1.9
            d = self.n_x
            x = np.array(x)
            x = x.reshape((self.n_x,1))
            i = np.arange(1, d + 1)
            
            z = np.sum((i * x) ** 2 + (a*i / d) * x * x[-1])

            # track f 
            self.f_list.append(z) 
            if self.track_x:
                self.x_list.append(x) 
                
            # return objective        
            return z
        
        ###################
        # Matyas_function #
        ###################

        if self.func_type == 'Matyas_f':

            x = np.array(x)
            x = x.reshape((self.n_x,1))

            z = 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

            # track f 
            self.f_list.append(z) 
            if self.track_x:
                self.x_list.append(x) 
                
            # return objective        
            return z
        

    ##################  
    # run constraint #
    ##################

    def con_test(self,x):

        
        ###################################
        # Constraints Rosenbrock_function #
        ###################################

        if self.func_type == 'Rosenbrock_f':

            '''
            x1 + 1.27 - 2.83x2 + 0.69x2**2 <= 0
            '''

            g1 = x[0] + 1.27 - 2.83*x[1] + 0.69*x[1]**2 


            # if g1 > vio:
            if isinstance(g1, float) and g1 > vio:
                # print('g1 violated')

                def g_func(x):
                    return -1.27 + 2.83*x - 0.69*x**2
            
                # Find the closest point on the function

                # Swapping positions because of Rosenbrock constraint formulation
                x_swapped = x[::-1]
                g_closest = closest_point_on_constraint(x_swapped, g_func)

                # Calculate the Euclidean distance
                distance = eucl_dist(x_swapped, g_closest)

                # track g 
                self.g_list.append(distance)

            else:
                self.g_list.append(None)

            return -g1
        
        ################################
        # Constraints Antonio_function #
        ################################

        if self.func_type == 'Antonio_f':

            '''
            1.5*x1 + 0.6 -x2 <= 0
            '''
            
            g1 = 1.5*x[0] + 0.6 - x[1]


            if isinstance(g1, float) and g1 > vio:
                # print('g1 violated')

                def g_func(x):
                    return 1.5*x+0.6
            
                # Find the closest point on the function
                g_closest = closest_point_on_constraint(x, g_func)

                # Calculate the Euclidean distance
                distance = eucl_dist(x, g_closest)

                # track g 
                self.g_list.append(distance)

            else:
                self.g_list.append(None)
            

            return -g1
        
        ################################
        # Constraints Matyas_function #
        ################################

        if self.func_type == 'Matyas_f':

            '''
            6.31225*x1 + 3.60257 - x2 <= 0
            '''
            
            g1 = 6.31225*x[0]+3.60257 - x[1]

            if isinstance(g1, float) and g1 > vio:
                # print('g1 violated')

                def g_func(x):
                    return 6.31225*x+3.60257
            
                # Find the closest point on the function
                g_closest = closest_point_on_constraint(x, g_func)

                # Calculate the Euclidean distance
                distance = eucl_dist(x, g_closest)

                # track g 
                self.g_list.append(distance)

            else:
                self.g_list.append(None)

            return -g1
        
    ###################  
    # plot constraint #
    ###################

    def con_plot(self,x):

        '''
        These are the same constraints but modified for plotting
        '''

        ###################################
        # Constraints Rosenbrock_function #
        ###################################

        if self.func_type == 'Rosenbrock_f':

            '''
            x1 + 1.27 - 2.83x2 + 0.69x2**2 <= 0
            '''
            x2 = x
            return -1.27 + 2.83*x2 - 0.69*(x2**2)
        
        ################################
        # Constraints Antonio_function #
        ################################

        if self.func_type == 'Antonio_f':

            '''
            1.5*x1 + 0.6 -x2 <= 0
            '''
            x1 = x
            return 1.5*x1 + 0.6
        
        ################################
        # Constraints Matyas_function #
        ################################

        if self.func_type == 'Matyas_f':

            '''
            6.31225*x1 + 3.60257 - x2 <= 0
            '''
            x1 = x
            return 6.31225*x1+3.60257

        
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