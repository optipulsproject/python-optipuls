import dolfin
import numpy as np


def func_generator(evo, V):
    theta = dolfin.Function(V)
    for coefs in evo:
        theta.vector().set_local(coefs)
        yield theta


def func_pair_generator(evo, V):
    theta_curr = dolfin.Function(V)
    theta_next = dolfin.Function(V)
    theta_curr.vector().set_local(evo[0])

    for coefs in evo[1:]:
        theta_next.vector().set_local(coefs)
        yield theta_curr, theta_next 
        self.theta_curr.assign(theta_next)


def values_generator(evo, V, point):
    for theta in func_generator(evo, V):
        yield theta(point)


def get_func_values(evo, V, point):
    '''Returns consequent values at a given point as a np.array.'''
    return np.fromiter(
        values_generator(evo, V, point),
        dtype=np.float64,
        count=len(evo),
        )
