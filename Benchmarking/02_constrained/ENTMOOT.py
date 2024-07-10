from entmoot import Enting, ProblemConfig, GurobiOptimizer
import numpy as np
from itertools import chain
import gurobipy as gp
from utils import *


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
        i_rep, 
        ):

    # define problem
    # note here: algorithm unstable for some other random seed
    problem_config = ProblemConfig(rnd_seed=1234)
    n = 9 # no of initial points to build model besides starting point.

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
    # Xtrain = t_.init_points[i_rep]
    if t_.func_type == 'WO_f': radius = 0.1
    else: radius = 0.5
    Xtrain = random_points_in_circle(n, radius=radius, center=t_.x0[i_rep].transpose())
    Ytrain = np.array([t_.fun_test(i) for i in Xtrain])
    samples_number = Xtrain.shape[0]

    # add starting point and random points
    train_x = np.insert(Xtrain, 0, x_start, axis=0).transpose()
    train_y = np.insert(Ytrain, 0, y_start, axis=0)

    # subtract from iterations
    iter_ = f_eval_ - samples_number - 1

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

    # if t_.func_type == 'WO_f':
        
    #     my_constraint1 = t_.WO_con1_test
    #     my_constraint2 = t_.WO_con2_test
    #     test1 = my_constraint1(model_gur._all_feat)
    #     test2 = my_constraint2(model_gur._all_feat)
    #     model_gur.addConstr(test1>=0)
    #     model_gur.addConstr(test2>=0)

    #introduce penalty value
    pnlty=5e8

    # my_constraint = t_.con_test
    # test = my_constraint(model_gur._all_feat)
    # model_gur.addConstr(test>=0)

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

        # get constraint values
        if t_.func_type == 'WO_f':
            con1 = t_.WO_con1_test(x_opt_eval)
            con2 = t_.WO_con2_test(x_opt_eval)
            con_val = [con1, con2]
        else:
            con_val = t_.con_test(x_opt_eval)

        # in order for t_.fun_test to evaluate the optimal point, it has to be transformed from a list of length n_x 
        # to a numpy array of shape (2,1)

        # the goal is to have the objective function low. therefore the penalty term should increase the obj fct'''
        '''
        con_val is negative when violated, so I put a - to reverse it and when I take the maximum I access that value
        then I add this to iuncrease the objective funciton value
        '''
        con_val_neg = [-i for i in con_val]
        con_val_neg.append(0)
        opt_trajectory_outputs[idx, 0] = t_.fun_test(x_opt_eval) + pnlty*np.max(con_val_neg)

    return None, None, None, None, np.array(opt_trajectory_inputs), None, np.array(opt_trajectory_inputs)[-1], None, None


