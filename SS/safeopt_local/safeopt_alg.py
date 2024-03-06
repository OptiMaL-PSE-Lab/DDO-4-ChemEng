## Implementing SafeOpt https://github.com/befelix/SafeOpt

from __future__ import print_function, division, absolute_import
import GPy
import numpy as np
import sys
from safeopt_f import*


def minus_f(f, x):
    # Apply the transformation by multiplying the output of the original function by -1
    return -1 * f(x)

def safeopt_alg(f, x_dim, bounds, iter_total):

    # Initialize GP prior
    x_init = np.zeros((1, x_dim))
    y_init = np.array([[minus_f(f,x_init)]])
    gp = GPy.models.GPRegression(x_init, y_init, noise_var=0.01**2)

    # Generate linearly spaced combinations as parameter set
    parameter_set = linearly_spaced_combinations(bounds, num_samples=100)

    # Initialize SafeOpt
    opt = SafeOpt(gp, parameter_set, fmin = -100000)

    # Optimization loop
    for i in range(iter_total):
        # Get next parameters to evaluate
        next_parameters = opt.optimize()

        # Evaluate objective function
        performance = np.array([[minus_f(f,next_parameters)]])

        # Add new data point to the GP
        opt.add_new_data_point(next_parameters, performance)

        x_res, f_res = opt.get_maximum()
        team_names = ['7','8']
        cids = ['01234567']
    return x_res, f_res, team_names, cids


# def objective_function(x):
#     """Rosenbrock function"""
#     return np.sum((100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2))


# # Example usage:
# x_dim = 2  
# bounds = [[-5., 5.]] * x_dim
# iter_total = 20

# x_res, f_res, team_names, cids = safeopt_optimize(objective_function, x_dim, bounds, iter_total)
# print("Min found at:", x_res)
# print("Obj val:", f_res)