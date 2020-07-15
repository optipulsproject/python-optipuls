from core import *
import argparse
from visualization import gradient_test_plot

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()


time_space = np.linspace(0, T, num=Nt, endpoint=True)

control = np.vectorize(u)(time_space, t1=0.005, t2=0.010)

deltas = []
labels = []

implicitness.assign(Constant("1.0"))
eps, delta = gradient_test(
    control, n=12, diff_type='forward', eps_init=2**-12)

deltas.append(delta)
labels.append('impl 1.0, qd none')

for deg in range(1,6):
    parameters["form_compiler"]["quadrature_degree"] = deg
    eps, delta = gradient_test(
        control, n=12, diff_type='forward', eps_init=2**-12)
    deltas.append(delta)
    labels.append('impl 1.0, qd ' + str(deg))

gradient_test_plot(eps, *deltas, labels=labels,
    outfile=args.scratch+'/gradient_test_plot.png', dpi=160)
