import argparse

import dolfin
import numpy as np

import core
import visualization as vis
from utils import io, laser
from simulation import Simulation
from problem import Problem, OptimizationParameters
from mesh import mesh


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()


# set dolfin parameters
dolfin.set_log_level(40)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 1


# set optimization options
# warning: changing of these values can currently break the optimization
opts = OptimizationParameters()
opts.beta_welding = core.beta_welding
opts.threshold_temp = core.threshold_temp
opts.target_point = core.target_point
opts.pow_ = core.pow_
opts.penalty_term_combined = core.penalty_term_combined
opts.implicitness = core.implicitness


# set up the problem
problem = Problem()
problem.V = dolfin.FunctionSpace(mesh, "CG", 1)
problem.V1 = dolfin.FunctionSpace(mesh, "DG", 0)
problem.theta_init = dolfin.project(core.temp_amb, problem.V)
problem.opts = opts

core.vhc.problem = problem
core.kappa_rad.problem = problem
core.kappa_ax.problem = problem

time_space = np.linspace(0, core.T, num=core.Nt, endpoint=True)
control = laser.linear_rampdown(time_space)
control[:] = 0.5


s = Simulation(problem, control)

epsilons, deltas_fwd = core.gradient_test(s, eps_init=10, iter_max=15)
vis.gradient_test_plot(epsilons, deltas_fwd)
# descent = core.gradient_descent(s, iter_max=200, step_init=2**-25)

# io.save_as_pvd(descent[-1].evo, problem.V, args.scratch+'/paraview/evo.pvd')
# io.save_as_pvd(descent[-1].evo_vel, problem.V1, args.scratch+'/paraview/velocity/evo_vel.pvd')