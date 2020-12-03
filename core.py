from collections import namedtuple
import json

import dolfin
from dolfin import dx, Constant, DOLFIN_EPS
import ufl
from ufl import inner, grad, conditional, ge, gt, lt, le, And
import numpy as np
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt

import splines as spl
from coefficients import vhc, kappa_rad, kappa_ax
from simulation import Simulation


# Space and time discretization parameters
R = 0.0025
R_laser = 0.0002
Z = 0.0005
T, Nt = 0.015, 300
dt = T/Nt

# Model constants
temp_amb = 295.
# enthalpy = Constant("397000")
P_YAG = 2000.
absorb = 0.135
laser_pd = (absorb * P_YAG) / (np.pi * R_laser**2)
implicitness = 1.

convection_coeff = 20.
radiation_coeff = 2.26 * 10**-9


# Optimization parameters
beta_control = 10**2
beta_velocity = 10**18
beta_welding = 10**-2
beta_liquidity = 10**12
velocity_max = 0.15
target_point = dolfin.Point(0, .7*Z)
central_point = dolfin.Point(0, 0)
threshold_temp = 1000.
pow_ = 20.

OptimizationParameters = namedtuple(
        'OptimizationParameters',
        [
            'beta_welding',
            'threshold_temp',
            'target_point',
            'pow_',
            'penalty_term_combined',
            'implicitness',
        ])

control_ref = np.zeros(Nt)

# Aggregate state
liquidus = 923.0
solidus = 858.0


class Domain_2(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.6 * R

class Domain_3(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.4 * R

class Domain_4(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.2 * R

class Domain_5(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.2 * R and x[1] > 0.5 * Z

class LaserBoundary(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > Z-DOLFIN_EPS and x[0] < R_laser
    
class EmptyBoundary(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            ((x[1] > Z-DOLFIN_EPS and x[0] >= R_laser) or x[1] < DOLFIN_EPS)

class SymAxisBoundary(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] < DOLFIN_EPS)
    
# Create and refine mesh
mesh = dolfin.RectangleMesh(dolfin.Point(0,0), dolfin.Point(R,Z), 25, 5)

domain_2 = Domain_2()
domain_3 = Domain_3()
domain_4 = Domain_4()
domain_5 = Domain_5()
# near_laser = NearLaser()

edge_markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim()-1)
domain_2.mark(edge_markers, True)
mesh = dolfin.refine(mesh, edge_markers)

edge_markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim()-1)
domain_3.mark(edge_markers, True)
mesh = dolfin.refine(mesh, edge_markers)

edge_markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim()-1)
domain_4.mark(edge_markers, True)
mesh = dolfin.refine(mesh, edge_markers)

edge_markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim()-1)
domain_5.mark(edge_markers, True)
mesh = dolfin.refine(mesh, edge_markers)

x = dolfin.SpatialCoordinate(mesh)


boundary_markers = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim()-1)

laser_boundary = LaserBoundary()
laser_boundary.mark(boundary_markers, 1)

empty_boundary = EmptyBoundary()
empty_boundary.mark(boundary_markers, 2)


ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundary_markers)


class DescendLoopException(Exception):
    pass


def kappa(theta):
    return dolfin.as_matrix(
            [[kappa_rad(theta), Constant(0)],
             [Constant(0), kappa_ax(theta)]])


def laser_bc(control_k):
    return laser_pd * Constant(control_k)


def cooling_bc(theta):
    return - convection_coeff * (theta - temp_amb)\
           - radiation_coeff * (theta**4 - temp_amb**4)


def norm2(vector):
    return dt * sum(vector**2)


def norm(vector):
    return np.sqrt(norm2(vector))


def avg(u_k, u_kp1, implicitness=implicitness):
    return implicitness * u_kp1 + (1 - implicitness) * u_k


def a(u_k, u_kp1, v, control_k):
    u_avg = avg(u_k, u_kp1)

    a_ = vhc(u_k) * (u_kp1 - u_k) * v * x[0] * dx\
       + dt * inner(kappa(u_k) * grad(u_avg), grad(v)) * x[0] * dx\
       - dt * laser_bc(control_k) * v * x[0] * ds(1)\
       - dt * cooling_bc(u_avg) * v * x[0] * (ds(1) + ds(2))

    return a_


def solve_forward(V, theta_init, control):
    '''Calculates the solution to the forward problem with the given control.

    This is a low level function, which is normally not supposed to be used
    directly by the end user. High level interfaces such as Simulation.evo must
    be used instead.

    For further details, see the indexing diagram.

    Parameters:
        V: dolfin.FunctionSpace
            The FEM space of the problem being solved.
        theta_init: dolfin.Function(V)
            The initial state.
        control: ndarray
            The laser power coefficient for every time step. 

    Returns:
        evolution: ndarray
            The coefficients of the calculated solution in the basis of
            the space V at each time step including the given initial state.
            
    '''

    # initialize state functions
    theta_k = dolfin.Function(V)    # stands for theta[k]
    theta_kp1 = dolfin.Function(V)  # stands for theta[k+1]

    # FEM equation setup
    v = dolfin.TestFunction(V)

    Nt = len(control)
    evolution = np.zeros((Nt+1, len(V.dofmap().dofs())))

    # preparing for the first iteration
    theta_k.assign(theta_init)
    evolution[0] = theta_k.vector().get_local()

    # solve forward, i.e. theta_k -> theta p_kp1, k = 0, 1, 2, ..., Nt-1
    for k in range(Nt):
        F = a(theta_k, theta_kp1, v, control[k])
        dolfin.solve(F == 0, theta_kp1)
        evolution[k+1] = theta_kp1.vector().get_local()

        # preparing for the next iteration
        theta_k.assign(theta_kp1)

    return evolution


def solve_adjoint(V, evo, control, opts):
    '''Calculates the solution to the adjoint problem.

    The solution to the adjoint equation is calculated using the explicitly
    given evolution (solution to the forward problem) and the control.

    For better understanding of the indeces see docs/indexing-diagram.txt.

    Parameters:
        V: dolfin.FunctionSpace
            The FEM space of the problem being solved.
        evo: ndarray
            The coefficients of the solution to the corresponding forward
            problem in the basis of the space V (see solve_forward).
        control: ndarray
            The laser power profile.
        opts: object
            Contains optimization parameters required in the adjoint_equation.

    Returns:
        evo_adj: ndarray
            The coefficients of the calculated adjoint solution in the basis of
            the space V.
            
    '''

    # initialize state functions
    theta_km1 = dolfin.Function(V)  # stands for theta[k-1]
    theta_k = dolfin.Function(V)    # stands for theta[k]
    theta_kp1 = dolfin.Function(V)  # stands for theta[k+1]

    # initialize adjoint state functions
    p_km1 = dolfin.Function(V)      # stands for p[k-1]
    p_k = dolfin.Function(V)        # stands for p[k]

    # FEM equation setup
    p = dolfin.TrialFunction(V)     # stands for unknown p[k-1]
    v = dolfin.TestFunction(V)

    Nt = len(control)
    evo_adj = np.zeros((Nt+1, len(V.dofmap().dofs())))

    # PointSource's magnitute precalculation
    if opts.beta_welding:
        sum_ = 0
        for k in range(1, Nt+1):
            theta_k.vector().set_local(evo[k])
            sum_ += theta_k(opts.target_point) ** opts.pow_
        p_norm = sum_ ** (1 / opts.pow_)
        magnitude_pre = opts.beta_welding * (p_norm - opts.threshold_temp)\
                      * sum_ ** (1/opts.pow_ - 1)

    # preparing for the first iteration
    # p[Nt] is never used so does not need to be initialized
    # the next line is for readability, can be omitted for performance
    theta_k.vector().set_local(evo[Nt])

    # solve backward, i.e. p_k -> p = p_km1, k = Nt, Nt-1, Nt-2, ..., 1
    for k in range(Nt, 0, -1):
        theta_km1.vector().set_local(evo[k-1])

        F = a(theta_km1, theta_k, p, control[k-1])
          # + opts.penalty_term_combined(k-1, theta_km1, theta_k)
        penalty = dt * opts.penalty_term_combined(k-1, theta_km1, theta_k)
        if penalty:
            F += penalty

        if k < Nt:
            F += a(theta_k, theta_kp1, p_k, control[k])
              # + opts.penalty_term_combined(k, theta_k, theta_kp1)
            penalty = dt * opts.penalty_term_combined(k, theta_k, theta_kp1)
            if penalty:
                F += penalty


        dF = dolfin.derivative(F, theta_k, v)
        
        # sometimes rhs(dF) is void which leads to a ValueError
        try:
            A, b = dolfin.assemble_system(dolfin.lhs(dF), dolfin.rhs(dF))
        except ValueError:
            A, b = dolfin.assemble_system(dolfin.lhs(dF), Constant(0)*v*dx)

        if opts.beta_welding:
            # calculate total magnitude and apply PointSource
            magnitude = - magnitude_pre\
                      * theta_k(opts.target_point) ** (opts.pow_ - 1)
            point_source = dolfin.PointSource(V, opts.target_point, magnitude)
            point_source.apply(b)

        dolfin.solve(A, p_km1.vector(), b)
 
        evo_adj[k-1] = p_km1.vector().get_local()

        # preparing for the next iteration
        p_k.assign(p_km1)
        theta_kp1.assign(theta_k)
        theta_k.assign(theta_km1)

    return evo_adj


def Dj(V, evo_adj, control):
    '''Calculates the gradient of the cost functional for the given control.

    For further details, see `indexing diagram`.

    Parameters:
        evo_adj: ndarray
            The evolution in time of the adjoint state.
        control: ndarray
            The laser power coefficient for every time step. 

    Returns:
        Dj: ndarray
            The gradient of the cost functional.
    '''

    p = dolfin.Function(V)
    z = np.zeros(Nt)

    for i in range(Nt):
        p.vector().set_local(evo_adj[i])
        z[i] = dolfin.assemble(p * x[0] * ds(1))
    
    Dj = beta_control * (control-control_ref) - laser_pd*z

    return Dj


def gradient_descent(simulation, iter_max=50, step_init=1, tolerance=10**-9):
    '''Runs the gradient descent procedure.

    Parameters:
        simulation: Simulation
            Simulation object used as the initial guess.
        iter_max: integer
            The maximal allowed number of major iterations.
        step_init: float
            The initial value of the descent step (Dj multiplier).
        tolerance: float
            The gradient descent procedure stops when the gradient norm becomes
            less than tolerance.

    Returns:
        descent: [Simulation]
            List of simulations, corresponding to succesful steps of gradient
            descent, starting with the provided initial guess.

    '''

    try:
        slow_down = False

        print('Starting the gradient descent procedure...')
        descent = [simulation]
        step = step_init
        problem = simulation.problem

        print(f'{"i.j " :>6} {"s":>14} {"J":>14} {"norm(Dj)":>14}')
        print(f'{simulation.J:36.7e} {simulation.Dj_norm:14.7e}')

        for i in range(iter_max):
            if simulation.Dj_norm < tolerance:
                print(f'norm(Dj) = {simulation.Dj_norm:.7e} < tolerance')
                break

            j = 0
            while True:
                control_trial = (
                    simulation.control - step * simulation.Dj).clip(0, 1)
                if np.allclose(control_trial, simulation.control):
                    raise DescendLoopException
                simulation_trial = Simulation(problem, control_trial)

                if simulation_trial.J < simulation.J: break
                print(f'{i:3}.{j:<2} {step:14.7e} '
                      f'{simulation_trial.J:14.7e}  {13*"-"}')
                j += 1
                step /= 2

            simulation = simulation_trial
            print(f'{i:3}.{j:<2} {step:14.7e} '
                  f'{simulation.J:14.7e} {simulation.Dj_norm:14.7e}')
            descent.append(simulation)
            step *= 2

        else:
            print('Maximal number of iterations was reached.')

    except KeyboardInterrupt:
        print('Interrupted by user...')
    except DescendLoopException:
        print('The descend procedure has looped since control reached '
              'the boundary of the feasible set:\n'
              'project(control_next) == control_current.')

    print('Terminating.')

    return descent


def gradient_test(simulation, iter_max=15, eps_init=.1, diff_type='forward'):
    '''Checks the accuracy of the calculated gradient Dj.

    The finite difference for J w.r.t. a random normalized direction is compared
    to the inner product (Dj,direction),  the absolute and the relative errors
    are calculated.

    Every iteration epsilon is divided by two.

    Parameters:
        simulation: Simulation
            Initial simulation used for testing.
        n: integer
            Number of tests.
        diff_type: 'forward' (default) or 'two_sided'
            The exact for of the finite difference expression.
        eps_init: float
            The initial value of epsilon.

    Returns:
        epsilons: [float]
            Epsilons used for testing.
        deltas: [float]
            Relative errors.

    '''

    print('Starting the gradient test...')
    # np.random.seed(0)
    direction = np.random.rand(Nt)
    direction /= norm(direction)
    direction *= .0005 * T

    problem = simulation.problem
    D = simulation.Dj

    epsilons = []
    deltas = []

    inner_product = dt * np.sum(D*direction)
    print(f'    inner(Dj,direction) = {inner_product:.8e}')
    print(f'{"epsilon":>20}{"diff":>20}{"delta_abs":>20}{"delta":>20}')

    try:
        for eps in (eps_init * 2**-k for k in range(iter_max)):
            if diff_type == 'forward':
                control_ = (simulation.control + eps * direction).clip(0, 1)
                simulation_ = Simulation(problem, control_)
                diff = (simulation_.J - simulation.J) / eps

            elif diff_type == 'two_sided':
                control_ = (control - eps * direction).clip(0, 1)
                simulation_ = Simulation(problem, control_)
                control__ = (control + eps * direction).clip(0, 1)
                simulation__ = Simulation(problem, control__)
                diff = (simulation__.J - simulation_.J) / (2 * eps)

            delta_abs = inner_product - diff
            delta = delta_abs / inner_product
            deltas.append(delta)
            epsilons.append(eps)

            print(f'{eps:20.8e}{diff:20.8e}{delta_abs:20.8e}{delta:20.8e}')

        print('Maximal number of iterations was reached.')

    except KeyboardInterrupt:
        print('Interrupted by user...')

    print('Terminating.')

    return epsilons, deltas


def penalty_term_velocity(k, theta_k, theta_kp1, velocity_max=velocity_max):
    theta_avg = avg(theta_k, theta_kp1, implicitness=.5)
    grad_norm = ufl.sqrt(inner(grad(theta_avg), grad(theta_avg)) + DOLFIN_EPS)

    func = (theta_kp1 - theta_k) / dt / grad_norm
    # allow small negative values (velocity_max)
    func += dolfin.Constant(velocity_max)
    func *= conditional(le(func, 0.), 1., 0.)
    # filter to the solidus-liquidus corridor
    func *= conditional(
            And(ge(theta_k, solidus), lt(theta_kp1, liquidus)), 1., 0.)
    # invert the sign
    func *= -1

    form = func**2 * x[0] * dx

    return form


def penalty_term_liquidity(k, theta_k, theta_kp1, implicitness=implicitness):
    '''Returns a UFL form corresponding to the liquidity penalty term.

    Parameters:
        theta_k: Function(V)
            Function representing the state from the current time slice.
        theta_kp1: Function(V)
            Function representing the state from the next time slice.
        implicitness: float
            The weight of theta_kp1 in the expression. The global implicitness
            parameter is used by default.

    Returns:
        form: UFL form
            A valid UFL form for assembling.

    '''
    
    theta_avg = avg(theta_k, theta_kp1, implicitness=1.)
    func = theta_avg - Constant(solidus)
    # filter to liquid parts
    func *= conditional(ge(func, 0.), 1., 0.)

    form = func**2 * x[0] * dx

    return form


def penalty_term_combined(k, theta_k, theta_kp1):
    '''Provides a linear combination of different penalty terms as a UFL form
    for solve_adjoint.'''
    if k == Nt:
        form = Constant(beta_liquidity) *\
           penalty_term_liquidity(k, theta_k, theta_kp1, implicitness=1.)
    else:
        form = 0
    form += Constant(beta_velocity) *\
            penalty_term_velocity(k, theta_k, theta_kp1)

    return form


def penalty_welding(V, evo, control):
    '''Penalty due to the maximal temperature at the target point.'''

    sum_ = 0
    theta = dolfin.Function(V)
    for k in range(1, Nt+1):
        theta.vector().set_local(evo[k])
        sum_ += np.float_power(theta(target_point), pow_)
    norm = np.float_power(sum_, 1/pow_)
    result = .5 * beta_welding * (norm - threshold_temp)**2

    return result


def vectorize_penalty_term(V, evo, penalty_term):
    '''Takes a penalty_term and provides a vector of penalties at time steps.

    Parameters:
        V: dolfin.FunctionSpace
            The FEM space of the problem being solved.
        evo: ndarray
            The coefficients of the solution to the corresponding forward
            problem in the basis of the space V (see solve_forward).

        penalty_term: (k, theta_k, theta_kp1, *args) -> UFL form
            Penalty term as a UFL form to be assembled and vectorized.

    Returns:
        penalty_vector: ndarray
            Contains penalty values due to penalty_term at every time step.

    '''

    theta_k = dolfin.Function(V)
    theta_kp1 = dolfin.Function(V)

    Nt = len(evo) - 1
    penalty_vector = np.zeros(Nt)

    theta_k.vector().set_local(evo[0])
    for k in range(Nt):
        theta_kp1.vector().set_local(evo[k+1])
        penalty = penalty_term(k, theta_k, theta_kp1)
        if penalty:
            penalty_vector[k] = dolfin.assemble(penalty)
        theta_k.assign(theta_kp1)

    return penalty_vector


def cost_total(V, evo, control):
    '''Calculates the total cost of a given simulation.

    This function puts together all the penalties from different terms.

    '''

    # penalties from integral temrs
    cost = dt * vectorize_penalty_term(V, evo, penalty_term_combined).sum()
    # control penalty
    cost += .5 * beta_control * norm2(control - control_ref)
    # welding penalty
    if beta_welding:
        cost += penalty_welding(V, evo, control)

    return cost


def temp_at_point_vector(V, evo, point):
    theta_k = dolfin.Function(V)
    vector = np.zeros(Nt+1)
    for k in range(Nt+1):
        theta_k.vector().set_local(evo[k])
        vector[k] = theta_k(point)

    return vector


def velocity(theta_k, theta_kp1, velocity_max=velocity_max, implicitness=.5):
    '''Provides the UFL form of the velocity function.

    Parameters:
        theta_k: dolfin.Function(V)
            Function representing the state from the current time slice.
        theta_kp1: dolfin.Function(V)
            Function representing the state from the next time slice.
        velocity_max: float
            The maximal allowed velocity which will not be penalized.
        implicitness: float
            The weight of theta_kp1 in the expression. Normally, should stay
            default. 
        implicitness: float

    Returns:
        form: UFL form
            The UFL form of theta_k and theta_kp1.

    '''

    theta_avg = avg(theta_k, theta_kp1, implicitness=.5)
    grad_norm = ufl.sqrt(inner(grad(theta_avg), grad(theta_avg)) + DOLFIN_EPS)
    form = (theta_kp1 - theta_k) / dt / grad_norm
    # filter to negative values over velocity_max in the solidus-liquidus
    # corridor and invert the sign
    form += dolfin.Constant(velocity_max)
    form *= conditional(le(form, 0.), 1., 0.)
    form *= conditional(
            And(ge(theta_k, solidus), lt(theta_kp1, liquidus)), 1., 0.)
    form *= -1

    return form


def compute_evo_vel(V, V1, evo):
    '''Computes the velocity evolution.

    Parameters:
        V: dolfin.FunctionSpace
            The FEM space of the forward solution.
        V1: dolfin.FunctionSpace
            The FEM space to project the velocity on.
        evo: ndarray
            The coefficients of the solution to the corresponding forward
            problem in the basis of the space V (see solve_forward).

    Returns:
        evo_vel: ndarray
            The coefficients of the calculated velocity in the basis of
            the space V1 at each time step.

    '''

    theta_k = dolfin.Function(V)
    theta_kp1 = dolfin.Function(V)

    Nt = len(evo) - 1
    evo_vel = np.zeros((Nt+1, len(V1.dofmap().dofs())))

    theta_k.vector().set_local(evo[0])
    for k in range(Nt):
        theta_kp1.vector().set_local(evo[k+1])

        form = velocity(theta_k, theta_kp1)
        func = dolfin.project(velocity(theta_k, theta_kp1), V1)

        evo_vel[k] = func.vector().get_local()

        theta_k.assign(theta_kp1)

    return evo_vel


class Problem:
    def __init__(self):
        pass

    # facade
    def solve_forward(self, control):
        return solve_forward(self.V, self.theta_init, control)

