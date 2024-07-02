# v2 includes shaping the TR with the curvature of the problem by a broyden update on derivatives
# and a BFGS update on the Hessian, however the TR becomes very small in some parts, so the approach
# does not seem to be too effective.

import time
import random
import numpy as np
import numpy.random as rnd
from scipy.spatial.distance import cdist

# import sobol_seq
from scipy.optimize import minimize
from scipy.optimize import broyden1
from scipy import linalg
import scipy
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Ellipse
from casadi import *

options = {'disp': False, 'maxiter': 10000}  # solver options
# Parameters
Fa = 1.8275
Mt = 2105.2
# kinetic parameters
phi1 = - 3.
psi1 = -17.
phi2 = - 4.
psi2 = -29.
# Reference temperature
Tref = 110. + 273.15  # [=] K.


class WO_system:

    # Parameters
    Fa = 1.8275 # Flowrate Reactand A
    Mt = 2105.2 
    # kinetic parameters
    phi1 = - 3.
    psi1 = -17.
    phi2 = - 4.
    psi2 = -29.
    # Reference temperature
    Tref = 110. + 273.15  # [=] K.

    def __init__(self):
        self.xd, self.xa, self.u, self.ODEeq, self.Aeq, self.states, self.algebraics, self.inputs = self.DAE_system()
        self.eval = self.integrator_system()
        self.f_list = []
        self.g_list = []
        self.x_list = []

    def DAE_system(self):
        # Define vectors with names of states
        states = ['x']
        nd = len(states)
        xd = SX.sym('xd', nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = ['Xa', 'Xb', 'Xc', 'Xe', 'Xp', 'Xg']
        na = len(algebraics)
        xa = SX.sym('xa', na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        inputs = ['Fb', 'Tr']
        nu = len(inputs)
        u = SX.sym("u", nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Reparametrization
        k1 = 1.6599e6 * np.exp(-6666.7 / (Tr + 273.15))
        k2 = 7.2117e8 * np.exp(-8333.3 / (Tr + 273.15))
        k3 = 2.6745e12 * np.exp(-11111. / (Tr + 273.15))

        # reaction rate
        Fr = Fa + Fb
        r1 = k1 * Xa * Xb * Mt
        r2 = k2 * Xb * Xc * Mt
        r3 = k3 * Xc * Xp * Mt

        # residual for x
        x_res = np.zeros((6, 1))
        x_res[0, 0] = (Fa - r1 - Fr * Xa) / Mt
        x_res[1, 0] = (Fb - r1 - r2 - Fr * Xb) / Mt
        x_res[2, 0] = (+ 2 * r1 - 2 * r2 - r3 - Fr * Xc) / Mt
        x_res[3, 0] = (+ 2 * r2 - Fr * Xe) / Mt
        x_res[4, 0] = (+   r2 - 0.5 * r3 - Fr * Xp) / Mt
        x_res[5, 0] = (+ 1.5 * r3 - Fr * Xg) / Mt
        # Define vectors with banes of input variables

        ODEeq = [0 * x]

        # Declare algebraic equations
        Aeq = []

        Aeq += [(Fa - r1 - Fr * Xa) / Mt]
        Aeq += [(Fb - r1 - r2 - Fr * Xb) / Mt]
        Aeq += [(+ 2 * r1 - 2 * r2 - r3 - Fr * Xc) / Mt]
        Aeq += [(+ 2 * r2 - Fr * Xe) / Mt]
        Aeq += [(+   r2 - 0.5 * r3 - Fr * Xp) / Mt]
        Aeq += [(+ 1.5 * r3 - Fr * Xg) / Mt]

        return xd, xa, u, ODEeq, Aeq, states, algebraics, inputs

    def integrator_system(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
        inputs: NaN
        outputs: F: Function([x, u, dt]--> [xf, obj])
        """

        xd, xa, u, ODEeq, Aeq, states, algebraics, inputs = self.DAE_system()
        VV = Function('vfcn', [xa, u], [vertcat(*Aeq)], ['w0', 'u'], ['w'])
        solver = rootfinder('solver', 'newton', VV)

        return solver

    def WO_obj_sys_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        Fb = u[0]
        Tr = u[1]
        Fa = 1.8275
        Fr = Fa + Fb

        obj = -(1043.38 * x[4] * Fr +
                20.92 * x[3] * Fr -
                79.23 * Fa -
                118.34 * Fb) + 0.5 * np.random.normal(0., 1)

        return float(obj)

    def WO_obj_sys_ca_noise_less(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        Fb = u[0]
        Tr = u[1]
        Fa = 1.8275
        Fr = Fa + Fb

        obj = -(1043.38 * x[4] * Fr +
                20.92 * x[3] * Fr -
                79.23 * Fa -
                118.34 * Fb)  # + 0.5*np.random.normal(0., 1)
        
        self.f_list.append(float(obj))
        self.x_list.append(u)

        return float(obj)

    def WO_con1_sys_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        pcon1 = x[0] - 0.12 + 5e-4 * np.random.normal(0., 1)

        return float(pcon1.toarray()[0])

    def WO_con2_sys_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        pcon2 = x[5] - 0.08 + 5e-4 * np.random.normal(0., 1)

        return float(pcon2.toarray()[0])

    def WO_con1_sys_ca_noise_less(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        pcon1 = x[0] - 0.12  # + 5e-4*np.random.normal(0., 1)
        self.g_list.append(float(pcon1.toarray()[0]))

        return float(pcon1.toarray()[0])

    def WO_con2_sys_ca_noise_less(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        pcon2 = x[5] - 0.08  # + 5e-4*np.random.normal(0., 1)
        self.g_list.append(float(pcon2.toarray()[0]))

        return float(pcon2.toarray()[0])
