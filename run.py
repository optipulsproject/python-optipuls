import argparse

import dolfin
import numpy as np

import core
import visualization as vis


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()


problem = core.Problem()
problem.V = dolfin.FunctionSpace(core.mesh, "CG", 1)
problem.theta_init = dolfin.project(core.temp_amb, problem.V)

# Warning: don't change these parameters yet
opts = core.OptimizationParameters(
        beta_welding=core.beta_welding,
        threshold_temp=core.threshold_temp,
        target_point=core.target_point,
        pow_=core.pow_,
        penalty_term_combined=core.penalty_term_combined,
        implicitness=core.implicitness)
problem.opts = opts

core.vhc.problem = problem
core.kappa_rad.problem = problem
core.kappa_ax.problem = problem

time_space = np.linspace(0, core.T, num=core.Nt, endpoint=True)
control = .25 * np.sin(time_space*np.pi / (2*core.T)) + .5

s = core.Simulation(problem, control)

epsilons, deltas_fwd = core.gradient_test(s, iter_max=15)
vis.gradient_test_plot(epsilons, deltas_fwd)
# descent = core.gradient_descent(s, iter_max=50, step_init=2**-25)
