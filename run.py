from core import *
import argparse
from visualization import gradient_test_plot

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls')
args = parser.parse_args()


time_space = np.linspace(0, T, num=Nt, endpoint=True)

control = np.vectorize(u)(time_space, t1=0.005, t2=0.010)

epsilons, deltas_fwd = gradient_test(
    control, n=15, diff_type='forward', eps_init=2**-10)
gradient_test_plot(epsilons, deltas_fwd)
