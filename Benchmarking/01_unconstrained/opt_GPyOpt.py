import numpy as np
import GPyOpt

#########################
# --- Random search --- #
#########################


def Random_search(f, n_p, bounds_rs, iter_rs):
    """
    This function is a naive optimization routine that randomly samples the
    allowed space and returns the best value.

    n_p: dimensions
    iter_rs: number of points to create
    """

    # arrays to store sampled points
    localx = np.zeros((n_p, iter_rs))  # points sampled
    localval = np.zeros((iter_rs))  # function values sampled
    # bounds
    bounds_range = bounds_rs[:, 1] - bounds_rs[:, 0]
    bounds_bias = bounds_rs[:, 0]

    for sample_i in range(iter_rs):
        x_trial = np.random.uniform(0, 1, n_p) * bounds_range + bounds_bias  # sampling
        localx[:, sample_i] = x_trial
        localval[sample_i] = f.fun_test(x_trial)
    # choosing the best
    minindex = np.argmin(localval)
    f_b = localval[minindex]
    x_b = localx[:, minindex]

    return f_b, x_b


#########################
# ----- Algorithm ----- #
#########################


def GPyOpt_BO(
    f,
    x_dim,
    bounds,
    f_eval_,  # length of trajectory (objective function evaluation budget)
    has_x0=False,
):
    """
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    take_x_start: boolean to either receive the starting point from f or to randomly sample it
    in case the starting point is randomly sampled, the number of samples is deducted from the evaluation budget. Mind, that
    the number of samples
    in case the starting point is taken from the function, this is treated as us providing the algorithm with a starting point.
    Therefore this counts as a single evaluation from the evaluation budget
    """

    if has_x0 == True:
        iter_ = f_eval_ - 1
        x_best = f.x0[0].flatten()

    else:
        n_rs = int(max(x_dim + 1, f_eval_ * 0.05))
        iter_ = f_eval_ - n_rs
        f_best, x_best = Random_search(f, x_dim, bounds, n_rs)

    # Define the bounds in the format required by GPyOpt
    domain = [
        {"name": "var_" + str(i + 1), "type": "continuous", "domain": bounds[i]}
        for i in range(x_dim)
    ]

    # Initialize Bayesian Optimization
    bo = GPyOpt.methods.BayesianOptimization(
        f=f.fun_test,
        domain=domain,
        initial_design_numdata=n_rs,
        initial_design_type="random",
        exact_feval=True,
    )

    # Run the optimization
    bo.run_optimization(max_iter=iter_)

    team_names = ["9", "10"]
    cids = ["01234567"]

    return bo.x_opt, bo.fx_opt, team_names, cids
