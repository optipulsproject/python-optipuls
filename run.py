import argparse

import dolfin
from dolfin import Constant
import numpy as np

import core
import visualization as vis
from utils import io, laser
from simulation import Simulation
from problem import Problem
from mesh import mesh, R, R_laser, Z
import coefficients


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()


# set dolfin parameters
dolfin.set_log_level(40)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 1


T, Nt = 0.015, 30
dt = T/Nt

P_YAG = 2000.
absorb = 0.135
R_laser = 0.0002
laser_pd = (absorb * P_YAG) / (np.pi * R_laser**2)

# set up the problem
problem = Problem()

problem.T = T
problem.Nt = Nt
problem.dt = dt

problem.P_YAG = P_YAG
problem.laser_pd = laser_pd

problem.temp_amb = 295.
problem.implicitness = 1.
problem.convection_coeff = 20.
problem.radiation_coeff = 2.26 * 10**-9
problem.liquidus = 923.0
problem.solidus = 858.0

# optimization parameters
problem.control_ref = np.zeros(Nt)
problem.beta_control = 10**2
problem.beta_velocity = 10**18
problem.velocity_max = 0.15
problem.beta_liquidity = 10**12
problem.beta_welding = 10**-2
problem.threshold_temp = 1000.
problem.target_point = dolfin.Point(0, .7*Z)
problem.pow_ = 20

# initialize FEM spaces
problem.V = dolfin.FunctionSpace(mesh, "CG", 1)
problem.V1 = dolfin.FunctionSpace(mesh, "DG", 0)

problem.theta_init = dolfin.project(problem.temp_amb, problem.V)


coefficients.vhc.problem = problem
coefficients.kappa_rad.problem = problem
coefficients.kappa_ax.problem = problem

problem.vhc = coefficients.vhc
problem.kappa = coefficients.kappa


time_space = np.linspace(0, T, num=Nt, endpoint=True)
control = laser.linear_rampdown(time_space)
control[:] = 0.5
# control[:Nt//2] = 1.


s = Simulation(problem, control)

epsilons, deltas_fwd = problem.gradient_test(s, eps_init=10**-5, iter_max=15)
vis.gradient_test_plot(epsilons, deltas_fwd)
# descent = core.gradient_descent(s, iter_max=200, step_init=2**-25)

# io.save_as_pvd(descent[-1].evo, problem.V, args.scratch+'/paraview/evo.pvd')
# io.save_as_pvd(descent[-1].evo_vel, problem.V1, args.scratch+'/paraview/velocity/evo_vel.pvd')