import argparse
import os

import dolfin
import numpy as np

from optipuls.problem import Problem
from optipuls.simulation import Simulation
from optipuls.time import TimeDomain
from optipuls.space import SpaceDomain
from optipuls.material import Material
from optipuls.optimization import gradient_test
from optipuls.visualization import gradient_test_plot


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--outdir',
    default='output',
    help='numerical artifacts output directory',
)
parser.add_argument(
    '--P_YAG',
    type=int,
    default=3000,
    help='maximal allowed laser power [W]',
)
parser.add_argument(
    '-Z',
    '--thickness',
    type=float,
    default=0.001,
    help='cylinder height (sheet thickness) [m]',
)
parser.add_argument(
    '-T',
    '--time_total',
    type=float,
    default=0.018,
    help='maximal pulse duration [s]',
)
parser.add_argument(
    '--time_resolution',
    type=float,
    default=10000,
    help='amount of discrete time step per second',
)

args = parser.parse_args()


# set up dolfin
dolfin.set_log_level(40)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 1

# set up the problem
problem = Problem()
problem.space_domain = SpaceDomain(0.0025, 0.0002, args.thickness)

# initialize FEM spaces
problem.V = dolfin.FunctionSpace(problem.space_domain.mesh, 'CG', 1)
problem.V1 = dolfin.FunctionSpace(problem.space_domain.mesh, 'DG', 0)

# load the material optipuls package resources
problem.material = Material.load('EN_AW-6082_T6.json')

# physical parameters
problem.temp_amb = 295.
problem.implicitness = 1.
problem.convection_coeff = 20.
problem.radiation_coeff = 2.26 * 10**-9
problem.theta_init = dolfin.project(problem.temp_amb, problem.V)
problem.absorb = 0.135

# optimization parameters
problem.beta_control = 10**2
problem.beta_velocity = 10**18
problem.velocity_max = 0.12
problem.beta_liquidity = 10**12
problem.beta_welding = 10**-2
problem.target_point = dolfin.Point(0, .75 * problem.space_domain.Z)
problem.threshold_temp = 1000.
problem.pow_ = 20

# set laser's parameters
laser_pd = (problem.absorb * args.P_YAG) / (np.pi * problem.space_domain.R_laser**2)
problem.P_YAG = args.P_YAG
problem.laser_pd = laser_pd

# initialize time_domain
problem.time_domain = TimeDomain(
        args.time_total,
        int(args.time_total * args.time_resolution),
        )

# initialize testing simulation
timeline = problem.time_domain.timeline
T = problem.time_domain.T
control_test = .25 * np.sin(timeline * np.pi / (2*T)) + .5
simulation_test = Simulation(problem, control_test)

# run gradient test
epsilons, deltas_fwd = gradient_test(simulation_test, iter_max=15)

# plot the results
os.makedirs(f'{args.outdir}', exist_ok=True)
gradient_test_plot(
    epsilons,
    deltas_fwd,
    outfile=f'{args.outdir}/gradient_test.png',
    )
