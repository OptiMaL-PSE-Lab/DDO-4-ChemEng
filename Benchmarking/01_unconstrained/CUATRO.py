import numpy as np
from cuatro import CUATRO
from cuatro import *

# def sim(x):
#     g1 = lambda x: (x[0] - 1)**3 - x[1] + 1
#     g2 = lambda x: x[0] + x[1] - 1.8
#     f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
#     return f(x), [g1(x), g2(x)]

# x0 = np.array([-2., 2.])

def Random_search(f, n_p, bounds_rs, iter_rs):
    '''
    This function is a naive optimization routine that randomly samples the 
    allowed space and returns the best value.

    n_p: dimensions
    iter_rs: number of points to create
    '''

    # arrays to store sampled points
    localx   = np.zeros((n_p,iter_rs))  # points sampled
    localval = np.zeros((iter_rs))        # function values sampled
    # bounds
    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        x_trial = np.random.uniform(0, 1, n_p)*bounds_range + bounds_bias # sampling
        localx[:,sample_i] = x_trial
        localval[sample_i] = f.fun_test(x_trial)
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return f_b, x_b

# bounds = np.array([(-5., 5.) for _ in range(len(x0))])
# budget = 100

def opt_CUATRO(t_, N_x_, bounds_, f_eval_, has_x0 = False):


    if has_x0 == True: 

        x_best = t_.x0[0].flatten()
        iter_ = f_eval_ - 1

    else:

        n_rs = int(max(N_x_+1,f_eval_*.05))
        f_best, x_best = Random_search(t_, N_x_, bounds_, n_rs)
        iter_          = f_eval_ - n_rs

    def t_CUATRO(x):
        return t_.fun_test(x), []

    solver_instance = CUATRO(
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

    res = solver_instance.run_optimiser(sim=t_CUATRO, x0=x_best, bounds=bounds_, max_f_eval=iter_, )

    # print(res['f_best_so_far'], res['x_best_so_far'])
    return None, None, None, None

def opt_CUATRO_pls(t_, N_x_, bounds_, f_eval_, has_x0 = False):


    if has_x0 == True: 

        x_best = t_.x0[0].flatten()
        iter_ = f_eval_ - 1

    else:

        n_rs = int(max(N_x_+1,f_eval_*.05))
        f_best, x_best = Random_search(t_, N_x_, bounds_, n_rs)
        iter_          = f_eval_ - n_rs

    def t_CUATRO(x):
        return t_.fun_test(x), []

    solver_instance = CUATRO(
                    init_radius = 0.1, # how much radius should the initial area cover 
                    beta_red = 0.001**(2/iter_), # trust region radius reduction heuristic
                    rescale_radius=True, # scale radii to unit box
                    method = 'local',
                    N_min_samples = 6, # 
                    constr_handling = 'Discrimination', # or 'Regression'
                    sampling = 'base', # maximize closest distance in trust region exploration
                    # explore = 'feasible_sampling', 
                    # dim_red='PLS',
                    dim_red='PLS'
                    # reject exploration samples that are predicted to violate constraints
                )

    res = solver_instance.run_optimiser(
        sim=t_CUATRO, 
        x0=x_best, 
        bounds=bounds_, 
        max_f_eval=iter_, 
        n_pls=5, 
        )

    # print(res['f_best_so_far'], res['x_best_so_far'])
    return None, None, None, None