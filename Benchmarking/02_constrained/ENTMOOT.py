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



def opt_ENTMOOT(
        t_, 
        N_x_, 
        bounds_, 
        f_eval_, 
        i_rep
        ):

    # define problem
    # note here: algorithm unstable for some other random seed
    problem_config = ProblemConfig(rnd_seed=1234)

    # specify dimensions and ranges
    for d_ in range(N_x_):

        bounds_tuple = tuple(np.float64(bounds_[d_]))
        problem_config.add_feature("real", bounds_tuple)

    # specify goal
    problem_config.add_min_objective()

    # # define no of initial iterations
    # n_rs = int(max(N_x_+1,f_eval_*.05)) - 1

    # provide starting point
    x_start = t_.x0[i_rep].flatten()
    y_start = t_.fun_test(x_start)

    # read additional init points
    Xtrain = t_.init_points[i_rep]
    Ytrain = np.array([t_.fun_test(i) for i in Xtrain])
    samples_number = Xtrain.shape[0]

    # # generate initial samples in addition to the starting sample
    # train_y_rand,train_x_rand = Random_searchENT(t_.fun_test, n_p=N_x_, bounds_rs=bounds_, iter_rs=n_rs)

    # add starting point and random points
    train_x = np.insert(Xtrain, 0, x_start, axis=0).transpose()
    train_y = np.insert(Ytrain, 0, y_start, axis=0)

    # subtract from iterations
    iter_ = f_eval_ - samples_number

    # get training-dataset in the form required for the fit method of enting
    # x is a list of tuples, which represent one datapoint each. A 2D-problem has 2 entries per tuple
    # y is a numpy array of shape (n_points, 1)
    train_y_cust = train_y.reshape((len(train_y),1))
    train_x_cust = list(zip(*train_x))

    # set hyperparameters
    params = {"unc_params": {"dist_metric": "l1", "acq_sense": "exploration", "beta": 1.5}}
    enting = Enting(problem_config, params=params)
    params_gurobi = {"MIPGap": 0, "OutputFlag": 0}

    # this is where the constraints come in
    model_gur = problem_config.get_gurobi_model_core()

    # x = model_gur._all_feat[0]
    # y = model_gur._all_feat[1]

    # add constraint that all variables should coincide
    # model_gur.addConstr(x == y)

    my_constraint = t_.con_test

    test = my_constraint(model_gur._all_feat)

    # print(model_gur._all_feat)

    model_gur.addConstr(test>=0)
    model_gur.update()
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
        res_gur = opt_gur.solve(enting, model_core=model_gur)
        opt_trajectory_inputs.append(tuple(xopt for xopt in res_gur.opt_point)) # opt_point is a list of length n_x (dimensions of x)

        x_opt_eval = np.array(res_gur.opt_point).reshape((len(res_gur.opt_point), 1))

        # in order for t_.fun_test to evaluate the optimal point, it has to be transformed from a list of length n_x 
        # to a numpy array of shape (2,1)
        opt_trajectory_outputs[idx, 0] = t_.fun_test(x_opt_eval)
        # opt_trajectory_outputs[idx, 0] = blackbox_ground_truth(opt_trajectory_inputs)[-1,0]

    return None, None, None, None, np.array(opt_trajectory_inputs), None, np.array(opt_trajectory_inputs)[-1], None, None


