import argparse
import pathlib

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
import optimization


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()


# set dolfin parameters
dolfin.set_log_level(40)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 1


T, Nt = 0.024, 240
dt = T/Nt

P_YAG = 1600.
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
problem.velocity_max = 0.08
problem.beta_liquidity = 10**12
problem.beta_welding = 10**-2
# problem.threshold_temp = 1050.  # for 100 time steps
problem.threshold_temp = 1095.  # for 200+ time steps?
problem.target_point = dolfin.Point(0, .75*Z)
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

# running forward simulation with linear-rampdown pulse shape
for rampdown_time in [0., 0.005, 0.010, 0.015]:
    control = laser.linear_rampdown(time_space, 0.005, 0.005 + rampdown_time)
    simulation = Simulation(problem, control)
    simulation.name = 'linear_rampdown_' + f'{rampdown_time:1.3f}'

    path = args.scratch+'/linear_rampdown_'+ f'{rampdown_time:1.3f}'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


    with open(path + '/report.txt', 'w+') as file:
        file.write(simulation.report)

    io.save_as_pvd(
        simulation.evo, simulation.problem.V, path + '/paraview/evo.pvd')
    io.save_as_pvd(
        simulation.evo_vel, simulation.problem.V1,
        path + '/paraview/evo_vel.pvd')


print('Creating a test simulation.')
test_control = 0.5 + 0.1 * np.sin(0.5 * time_space / np.pi)
test_simulation = Simulation(problem, test_control)

vis.size_inches = (6, 4)
epsilons, deltas_fwd = optimization.gradient_test(
        test_simulation, eps_init=10**-5, iter_max=15)
vis.gradient_test_plot(
        epsilons, deltas_fwd, outfile=args.scratch+'/gradient_test.pdf', transparent=True)
print(f'Gradient test complete. See {args.scratch}/gradient_test.png')

print('Creating an initial guess simulation.')
control = np.zeros(Nt)
simulation = Simulation(problem, control)

descent = optimization.gradient_descent(
        simulation, iter_max=25, step_init=2**-25)

optimal = descent[-1]

path_optimized = args.scratch + '/optimized'
pathlib.Path(path_optimized).mkdir(parents=True, exist_ok=True)
with open(path_optimized + '/report.txt', 'w') as file:
        file.write(optimal.report)

io.save_as_pvd(
        optimal.evo, optimal.problem.V,
        path_optimized + '/paraview/evo.pvd')
io.save_as_pvd(
        optimal.evo_vel, optimal.problem.V1,
        path_optimized + '/paraview/evo_vel.pvd')

vis.optimal_control_plot(
        optimal.control,
        outfile=path_optimized+'/optimal_control.png')

np.save(path_optimized+'/optimal_control.npy', optimal.control)

print(f'Gradient descent complete. See {args.scratch}/optimal_control.png')
