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


time_space = np.linspace(0, core.T, num=core.Nt, endpoint=True)

control = .25 * np.sin(time_space*np.pi / (2*core.T)) + .5
control[:] = .5
s = core.Simulation(control)
epsilons, deltas_fwd = core.gradient_test(s, iter_max=15)
vis.gradient_test_plot(epsilons, deltas_fwd)
# descent = core.gradient_descent(s, iter_max=50, step_init=2**-25)
