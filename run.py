from core import *
import argparse
from visualization import gradient_test_plot, objective_plot, control_plot

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='../output')
parser.add_argument('-s', '--scratch', default='/scratch/OptiPuls/current')
args = parser.parse_args()

# preheat the body and save the state (if not done yet)
# control = np.full(Nt, 1., dtype=float)
# evo = solve_forward(control)
# save_as_npy(evo[-1], args.scratch+'/theta_hot.npy')

# load the saved state
# theta_hot_np = np.load(args.scratch+'/theta_hot.npy')
# theta_hot = Function(V)
# theta_hot.vector().set_local(theta_hot_np)


time_space = np.linspace(0, T, num=Nt, endpoint=True)

control = .25 * np.sin(time_space*np.pi / (2*T)) + .5
epsilons, deltas_fwd = gradient_test(control, n=15)
gradient_test_plot(epsilons, deltas_fwd)


# print("Starting optimization process")
# control = np.vectorize(u)(time_space, t1=0, t2=0.005)
# control[:] = 0.
# evo = solve_forward(control, theta_hot)
# J_total_before = J_total(evo, control, coefficients, expressions)
# J_vector_before = J_vector(evo, control, coefficients, expressions)
# print("J_total before optimization:", J_total_before)
# descent = gradient_descent(control, init=theta_hot, iter_max=30, s=10**-10)

# control_plot_outfile = args.scratch+'/control_plot.png'
# control_plot(
#     descent[0], descent[-1],
#        labels=["initial control", "optimized control"],
#        outfile=control_plot_outfile)
# print("See", control_plot_outfile)

# evo_optimized = solve_forward(descent[-1], theta_hot)
# J_total_after = J_total(evo_optimized, descent[-1], coefficients, expressions)
# J_vector_after = J_vector(evo_optimized, descent[-1], coefficients, expressions)
# print("J_total after optimization:", J_total_after)

# objective_plot_outfile = args.scratch+'/objective_plot.png'
# objective_plot(
#     J_vector_before, J_vector_after,
#     labels=["objective before", "objective after"],
#     outfile=objective_plot_outfile)
# print("See", objective_plot_outfile)
