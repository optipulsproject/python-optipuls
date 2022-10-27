import argparse
import os

import dolfin
import numpy as np

from optipuls.simulation import Simulation
from optipuls.problem import Problem
import optipuls.optimization as optimization
from optipuls.time import TimeDomain
from optipuls.space import SpaceDomain
from optipuls.material import Material
from optipuls.utils.io import save_as_pvd
from optipuls.visualization import control_plot
import optipuls.optimization as optimization


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()

# set dolfin parameters
dolfin.set_log_level(40)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 1

# initialize the problem
problem = Problem()

# initialize the time domain
time_domain = TimeDomain(0.020, 200)
problem.time_domain = time_domain

# load the mesh
mesh = dolfin.Mesh()
with dolfin.XDMFFile("examples/mesh/singlespot-XZ/singlespot_XZ.xdmf") as infile:
    infile.read(mesh)

# initialize the space domain
space_domain = SpaceDomain(0.0025, 0.0002, 0.0005)
space_domain.dim = 2
space_domain._x = dolfin.SpatialCoordinate(mesh)
space_domain._ds = dolfin.Measure('ds', domain=mesh)
problem.space_domain = space_domain

# set simulation parameters
P_YAG = 3000.
absorb = 0.135
laser_pd = (absorb * P_YAG) / (np.pi * space_domain.R_laser**2)

problem.P_YAG = P_YAG
problem.laser_pd = laser_pd

problem.temp_amb = 295.
problem.implicitness = 1.
problem.convection_coeff = 20.
problem.radiation_coeff = 2.26 * 10**-9
problem.liquidus = 923.0
problem.solidus = 858.0

# optimization parameters
problem.beta_control = 10**2
problem.beta_velocity = 10**18
problem.velocity_max = 0.15
problem.beta_liquidity = 10**12
problem.beta_welding = 10**-2
problem.threshold_temp = 1000.
problem.target_point = dolfin.Point(0, 0, -.7 * space_domain.Z)
problem.pow_ = 20

# initialize FEM spaces
problem.V = dolfin.FunctionSpace(mesh, "CG", 1)
problem.V1 = dolfin.FunctionSpace(mesh, "DG", 0)

problem.theta_init = dolfin.project(problem.temp_amb, problem.V)

# read the material properties and initialize equation coefficients
problem.material = Material.load('EN_AW-6082_T6.json')

print('Creating an initial simulation.')
control = np.zeros(time_domain.Nt)
simulation = Simulation(problem, control)

descent = optimization.gradient_descent(
        simulation, iter_max=20, step_init=2**-25)

optimized = descent[-1]

os.makedirs(f'{args.scratch}', exist_ok=True)

# save optimized control
control_plot(
        optimized.control * P_YAG,
        labels=['Optimized Control'],
        timeline=time_domain.timeline,
        y_max=P_YAG,
        outfile=f'{args.scratch}/optimized_control.png')

# save the temperature evolution to visualize in ParaView
save_as_pvd(optimized.evo, problem.V, f'{args.scratch}/paraview/optimized/evo.pvd')

print(f'Gradient descent complete. See {args.scratch}.')