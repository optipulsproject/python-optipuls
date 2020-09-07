from core import *
import argparse
from visualization import gradient_test_plot

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()


time_space = np.linspace(0, T, num=Nt, endpoint=True)

control = .25 * np.sin(time_space*np.pi / (2*T)) + .5
epsilons, deltas_fwd = gradient_test(control, n=15)
gradient_test_plot(epsilons, deltas_fwd)

