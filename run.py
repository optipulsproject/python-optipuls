from core import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()

time_space = np.linspace(0, T, num=Nt, endpoint=True)
control = np.vectorize(u)(time_space, t1=0.005, t2=0.005)

epsilons, deltas = gradient_test(control, n=20)
gradient_test_plot(epsilons, deltas)

