from core import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
args = parser.parse_args()

time_space = np.linspace(0, T, num=Nt, endpoint=True)
control = np.vectorize(u)(time_space, t1=0.005, t2=0.010)
evo = solve_forward(control)
save_as_npy(evo, args.output+'/evo.npy')
save_as_pvd(evo, args.output+'/paraview/evo.pvd')
# optionally load pre-calculated evolution
# evo = np.load(args.output+'/evo.npy')
