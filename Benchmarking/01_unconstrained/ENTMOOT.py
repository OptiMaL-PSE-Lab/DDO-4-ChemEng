from entmoot import Enting, ProblemConfig, GurobiOptimizer
import numpy as np
from itertools import chain
import gurobipy as gp


def Random_searchENT(f, n_p, bounds_rs, iter_rs):
    '''
    This function is a naive optimization routine that randomly samples the 
    allowed space and returns the best value.
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
        localval[sample_i] = f(x_trial) # f
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return localval,localx



def opt_ENTMOOT(t_, N_x_, bounds_, f_eval_, has_x0 = False):


    if has_x0 == True:

        print('For later if it is decided to keep the algorithm')

    else:

        # define problem
        problem_config = ProblemConfig(rnd_seed=1234)

        # specify dimensions and ranges
        for d_ in range(N_x_):

            bounds_tuple = tuple(np.float64(bounds_[d_]))
            problem_config.add_feature("real", bounds_tuple)

        # specify goal
        problem_config.add_min_objective()

        # define no of initial iterations
        n_rs = int(max(N_x_+1,f_eval_*.05))
        iter_ = f_eval_ - n_rs

        # generate initial samples
        train_y,train_x = Random_searchENT(t_.fun_test, n_p=N_x_, bounds_rs=bounds_, iter_rs=n_rs)

        # get training-dataset in the form required for the fit method of enting
        # x is a list of tuples, which represent one datapoint each. A 2D-problem has 2 entries per tuple
        # y is a numpy array of shape (n_points, 1)
        train_y_cust = train_y.reshape((len(train_y),1))
        train_x_cust = list(zip(*train_x))

        # set hyperparameters
        params = {"unc_params": {"dist_metric": "l1", "acq_sense": "exploration", "beta": 1.5}}
        enting = Enting(problem_config, params=params)
        params_gurobi = {"MIPGap": 0, "OutputFlag": 0}
        opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)

        # Remember the proposals and outcomes in these variables
        opt_trajectory_inputs = []
        opt_trajectory_outputs = np.empty((iter_, 1))

        # optimization loop
        for idx in range(iter_):
            # Put together the initial dataset and any optimization iterations we have done so far
            x = [_ for _ in chain(train_x_cust, opt_trajectory_inputs)]
            y = np.concatenate(
                (train_y_cust, opt_trajectory_outputs[:idx, :]), axis=0
            )

            enting.fit(x, y) 
            res_gur = opt_gur.solve(enting)
            opt_trajectory_inputs.append(tuple(xopt for xopt in res_gur.opt_point)) # opt_point is a list of length n_x (dimensions of x)

            x_opt_eval = np.array(res_gur.opt_point).reshape((len(res_gur.opt_point), 1))

            # in order for t_.fun_test to evaluate the optimal point, it has to be transformed from a list of length n_x 
            # to a numpy array of shape (2,1)
            opt_trajectory_outputs[idx, 0] = t_.fun_test(x_opt_eval)
            # opt_trajectory_outputs[idx, 0] = blackbox_ground_truth(opt_trajectory_inputs)[-1,0]

        return None, None, None, None


