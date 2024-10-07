import numpy as np
from cuatro import CUATRO as CUATRO_solv
from cuatro import *

def opt_CUATRO(t_, N_x_, bounds_, f_eval_, i_rep):

    x_start = t_.x0[i_rep].flatten()
    iter_          = f_eval_ # sometimes cuatro evaluates less than it could but function and constraint evaluations match

    def t_CUATRO(x):

        if t_.func_type == 'WO_f':  return t_.fun_test(x), [-t_.WO_con1_test(x), -t_.WO_con2_test(x)]
        else: return t_.fun_test(x), [-t_.con_test(x)]

    solver_instance = CUATRO_solv(
                    init_radius = 0.1, # how much radius should the initial area cover 
                    beta_red = 0.001**(2/iter_), # trust region radius reduction heuristic
                    rescale_radius=True, # scale radii to unit box
                    method = 'local',
                    N_min_samples = 6, # 
                    constr_handling = 'Discrimination', # or 'Regression'
                    sampling = 'base', # maximize closest distance in trust region exploration
                    explore = 'feasible_sampling', 
                    # reject exploration samples that are predicted to violate constraints
                )

    res = solver_instance.run_optimiser(sim=t_CUATRO, x0=x_start, bounds=bounds_, max_f_eval=iter_, )

    team_names = ['9','10']
    cids = ['01234567']

    # print(res['f_best_so_far'], res['x_best_so_far'])
    return None, None, team_names, cids, None, None, None, None, None