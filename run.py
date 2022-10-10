import argparse

import dolfin
from dolfin import Constant, as_matrix
import numpy as np

import optipuls.visualization as vis
from optipuls.simulation import Simulation
from optipuls.problem import Problem
import optipuls.material as material
import optipuls.optimization as optimization
from optipuls.time import TimeDomain
from optipuls.space import SpaceDomain
from optipuls.material import Material


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()


# set dolfin parameters
dolfin.set_log_level(40)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 1


# set up the problem
problem = Problem()

time_domain = TimeDomain(0.020, 200)
problem.time_domain = time_domain


# experimental

mesh = dolfin.Mesh()
with dolfin.XDMFFile("mesh/singlespot/singlespot_XZ.xdmf") as infile:
    infile.read(mesh)

# mesh_boundaries = dolfin.MeshValueCollection("size_t", mesh, 2)
# with dolfin.XDMFFile("mesh/singlespot/singlespot_XZ_boundaries.xdmf") as infile:
#     infile.read(mesh_boundaries, "ids")

# mesh_subdomains = dolfin.MeshValueCollection("size_t", mesh, 3)
# with dolfin.XDMFFile("mesh/singlespot/singlespot_XZ_subdomains.xdmf") as infile:
#     infile.read(mesh_subdomains, "ids")

space_domain = SpaceDomain(0.0050, 0.0002, 0.0005)
space_domain.dim = 2
# subdomain_data = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mesh_boundaries)
space_domain._x = dolfin.SpatialCoordinate(mesh)
# space_domain._ds = dolfin.Measure('ds', domain=mesh, subdomain_data=subdomain_data)
space_domain._ds = dolfin.Measure('ds', domain=mesh)

problem.space_domain = space_domain

#####

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

########################################


problem.material = Material.load('EN_AW-6082_T6.json')


# R = 0.0025;
# r = 0.0002;
# overlap = 0.6;
# d = 2 * r * (1 - overlap);


from optipuls.utils.laser import linear_rampdown
from optipuls.utils.io import save_as_pvd

print('Creating an initial simulation.')
control = np.zeros(time_domain.Nt)
simulation = Simulation(problem, control)

descent = optimization.gradient_descent(
        simulation, iter_max=20, step_init=2**-25)

vis.control_plot(
        descent[-1].control,
        labels=['Optimal Control'],
        outfile=args.scratch+'/optimal_control.png')
print(f'Gradient descent complete. See {args.scratch}/optimal_control.png')

# optimized = descent[-1]
# save_as_pvd(optimized.evo, problem.V, 'output/optimized.pvd')


# control = linear_rampdown(time_domain.timeline, t1=0.005, t2=0.010)
# simulation = Simulation(problem, control)
# print(simulation.evo.max())
# save_as_pvd(simulation.evo, problem.V, 'output/evo.pvd')
# theta_final = dolfin.Function(problem.V)
# theta_final.set_allow_extrapolation(True)
# theta_final.vector().set_local(simulation.evo[-1])

# from optipuls.utils.shiftedexpression import ShiftedExpression

# theta_shifted = ShiftedExpression(
#     expr_interpolate=(lambda p: theta_final(p)),
#     expr_extrapolate=(lambda _: problem.temp_amb),
#     shift=(d, 0, 0),
#     bounding_box_tree=mesh.bounding_box_tree(),
#     )

# theta_init_new = dolfin.Function(problem.V)
# theta_init_new = dolfin.interpolate(theta_shifted, problem.V)


# print('Creating a test simulation.')
# test_control = 0.5 + 0.1 * np.sin(0.5 * time_domain.timeline / np.pi)
# test_simulation = Simulation(problem, test_control)

# epsilons, deltas_fwd = optimization.gradient_test(
#         test_simulation, eps_init=10**-5, iter_max=15)
# vis.gradient_test_plot(
#         epsilons, deltas_fwd, outfile=args.scratch+'/gradient_test.png')
# print(f'Gradient test complete. See {args.scratch}/gradient_test.png')


# print('A linear rampdown simulation.')
# control = linear_rampdown(time_domain.timeline, t1=0.005, t2=0.010)
# simulation = Simulation(problem, control)
# save_as_pvd(simulation.evo, problem.V, 'output/evo.pvd')

# print('Creating an initial simulation.')
# control = np.zeros(time_domain.Nt)
# simulation = Simulation(problem, control)

# descent = optimization.gradient_descent(
#         simulation, iter_max=20, step_init=2**-25)

# vis.control_plot(
#         descent[-1].control,
#         labels=['Optimal Control'],
#         outfile=args.scratch+'/optimal_control.png')
# print(f'Gradient descent complete. See {args.scratch}/optimal_control.png')

# optimized = descent[-1]
# save_as_pvd(optimized.evo, problem.V, 'output/optimized.pvd')
