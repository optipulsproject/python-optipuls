from dolfin import *
from mshr import *
import ufl
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial
import json

# from tqdm import trange

import splines as spl

set_log_level(40)
parameters["form_compiler"]["quadrature_degree"] = 1

# Space and time discretization parameters
R = 0.0025
R_laser = 0.0002
Z = 0.0005
T, Nt = 0.010, 30
dt = T/Nt

# Model constants
theta_amb = Constant("295")
enthalpy = Constant("397000")
P_YAG = 1600.
absorb = 0.135
laser_pd = (absorb * P_YAG) / (pi * R_laser**2)
implicitness = Constant("1.0")

# Optimization parameters
alpha = 0. # temporarily exclude the control cost
beta_velocity = 1.
beta_welding = 1.
beta_liquidity = 1.
velocity_max = 0.12
target_point = Point(0, .5*Z)
central_point = Point(0, 0)
threshold_temp = 1102.
pow_ = 6.

control_ref = np.zeros(Nt)

# Aggregate state
liquidus = 923.0
solidus = 858.0


class Domain_2(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.6 * R

class Domain_3(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.4 * R

class Domain_4(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.2 * R

class Domain_5(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.2 * R and x[1] > 0.5 * Z

class LaserBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > Z-DOLFIN_EPS and x[0] < R_laser
    
class EmptyBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            ((x[1] > Z-DOLFIN_EPS and x[0] >= R_laser) or x[1] < DOLFIN_EPS)

class SymAxisBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] < DOLFIN_EPS)
    
# Create and refine mesh
mesh = RectangleMesh(Point(0,0), Point(R,Z), 25, 5)

domain_2 = Domain_2()
domain_3 = Domain_3()
domain_4 = Domain_4()
domain_5 = Domain_5()
# near_laser = NearLaser()

edge_markers = MeshFunction("bool", mesh, mesh.topology().dim()-1)
domain_2.mark(edge_markers, True)
mesh = refine(mesh, edge_markers)

edge_markers = MeshFunction("bool", mesh, mesh.topology().dim()-1)
domain_3.mark(edge_markers, True)
mesh = refine(mesh, edge_markers)

edge_markers = MeshFunction("bool", mesh, mesh.topology().dim()-1)
domain_4.mark(edge_markers, True)
mesh = refine(mesh, edge_markers)

edge_markers = MeshFunction("bool", mesh, mesh.topology().dim()-1)
domain_5.mark(edge_markers, True)
mesh = refine(mesh, edge_markers)

x = SpatialCoordinate(mesh)

# Define function space 
V = FunctionSpace(mesh, "CG", 1)
V1 = FunctionSpace(mesh, "DG", 0)

boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)

laser_boundary = LaserBoundary()
laser_boundary.mark(boundary_markers, 1)

empty_boundary = EmptyBoundary()
empty_boundary.mark(boundary_markers, 2)

sym_axis_boundary = SymAxisBoundary()
sym_axis_boundary.mark(boundary_markers, 3)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)


class Simulation():
    '''Caches the computation results and keeps related values together.'''

    def __init__(self, control, theta_init=project(theta_amb, V)):
        self._control = control
        self._theta_init = theta_init

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, control_):
        raise AttributeError(
            'Assigning a new control in an existing simulation object is '
            'forbidden for safety reasons. Create a new object instead.')

    @property
    def theta_init(self):
        return self._theta_init

    @theta_init.setter
    def theta_init(self, theta_init_):
        if not isinstance(theta_init_, Function):
            raise TypeError('the type is not a Function')

        self._theta_init = theta_init_

    @property
    def evo(self):
        try:
            return self._evo
        except AttributeError:
            self._evo = solve_forward(self.control, self.theta_init)
            return self._evo

    @property
    def evo_adj(self):
        try:
            return self._evo_adj
        except AttributeError:
            self._evo_adj = solve_adjoint(self.evo, self.control)
            return self._evo_adj

    @property
    def Dj(self):
        try:
            return self._Dj
        except AttributeError:
            self._Dj = Dj(self.evo_adj, self.control)
            return self._Dj

    @property
    def Dj_norm(self):
        try:
            return self._Dj_norm
        except AttributeError:
            self._Dj_norm = norm(self.Dj)
            return self._Dj_norm

    @property
    def penalty_velocity_vector(self):
        try:
            return self._penalty_velocity_vector
        except AttributeError:
            self._penalty_velocity_vector = penalty_vector(self.evo, velocity)
            return self._penalty_velocity_vector

    @property
    def penalty_velocity_total(self):
        try:
            return self._penalty_velocity_total
        except AttributeError:
            self._penalty_velocity_total =\
                    dt * sum(self.penalty_velocity_vector)
            return self._penalty_velocity_total

    @property
    def penalty_liquidity_vector(self):
        try:
            return self._penalty_liquidity_vector
        except AttributeError:
            self._penalty_liquidity_vector = penalty_vector(self.evo, liquidity)
            return self._penalty_liquidity_vector

    @property
    def penalty_liquidity_total(self):
        try:
            return self._penalty_liquidity_total
        except AttributeError:
            self._penalty_liquidity_total =\
                    dt * sum(self.penalty_liquidity_vector)
            return self._penalty_liquidity_total

    @property
    def penalty_welding_total(self):
        try:
            return self._penalty_welding_total
        except AttributeError:
            self._penalty_welding_total = J_welding(self.evo, self.control)
            return self._penalty_welding_total

    @property
    def temp_target_point_vector(self):
        try:
            return self._temp_target_point_vector
        except AttributeError:
            self._temp_target_point_vector = \
                    temp_at_point_vector(self.evo, target_point)
            return self._temp_target_point_vector

    @property
    def temp_central_point_vector(self):
        try:
            return self._temp_central_point_vector
        except AttributeError:
            self._temp_central_point_vector = \
                    temp_at_point_vector(self.evo, central_point)
            return self._temp_central_point_vector

    @property
    def J(self):
        try:
            return self._J_total
        except AttributeError:
            self._J_total = J(self.evo, self.control)
            return self._J_total

    def descend(self, step):
        control_next = (self.control - step * self.Dj).clip(0, 1)
        simulation_next = Simulation(control_next, self.theta_init)
        return simulation_next


with open('material.json') as file:
    material = json.load(file)

spline = spl.gen_hermite_spline(
    material['heat capacity']['knots'],
    material['heat capacity']['values'])
c = spl.spline_as_ufl(spline, material['heat capacity']['knots'])

spline = spl.gen_hermite_spline(
    material['density']['knots'],
    material['density']['values'],
    extrapolation='linear')
rho = spl.spline_as_ufl(spline, material['density']['knots'])

spline = spl.gen_hermite_spline(
    material['thermal conductivity']['radial']['knots'],
    material['thermal conductivity']['radial']['values'])
kappa_rad = spl.spline_as_ufl(spline,
                material['thermal conductivity']['radial']['knots'])

spline = spl.gen_hermite_spline(
    material['thermal conductivity']['axial']['knots'],
    material['thermal conductivity']['axial']['values'])
kappa_ax = spl.spline_as_ufl(spline,
                material['thermal conductivity']['axial']['knots'])


def kappa(theta):
    return as_matrix([[kappa_rad(theta), Constant("0.0")],
                      [Constant("0.0"), kappa_ax(theta)]])


def s(theta):
    return c(theta) * rho(theta)


def laser_bc(intensity):
    return laser_pd * intensity


def cooling_bc(theta):
    return - 20 * (theta-theta_amb)\
           - 2.26 * 10**(-9) * (theta**4-theta_amb**4)


def norm(vector):
    return sqrt(dt * sum(vector**2))


def a(u, u_, v, intensity):

    u_m = implicitness * u_ + (1-implicitness) * u

    a = s(u) * (u_ - u) * v * x[0] * dx\
      + dt * inner(kappa(u) * grad(u_m), grad(v)) * x[0] * dx\
      - dt * laser_bc(intensity) * v * x[0] * ds(1)\
      - dt * cooling_bc(u_m) * v * x[0] * (ds(1) + ds(2))

    return a

def solve_forward(control, theta_init=project(theta_amb, V)):
    '''Calculates the solution to the forward problem with the given control.

    For further details, see `indexing diagram`.

    Parameters:
        control: ndarray
            The laser power coefficient for every time step. 
        theta_init: Function(V)
            The initial state of the temperature.

    Returns:
        evolution: ndarray
            The coefficients of the calculated solution in the basis of
            the space V at each time step including the initial state.
            
    '''

    # initialize state functions
    theta_k = Function(V)    # stands for theta[k]
    theta_kp1 = Function(V)  # stands for theta[k+1]

    # FEM equation setup
    v = TestFunction(V)

    Nt = len(control)
    evolution = np.zeros((Nt+1, len(V.dofmap().dofs())))

    # preparing for the first iteration
    theta_k.assign(theta_init)
    evolution[0] = theta_k.vector().get_local()

    # solve forward, i.e. theta_k -> theta p_kp1, k = 0, 1, 2, ..., Nt-1
    for k in range(Nt):
        F = a(theta_k, theta_kp1, v, Constant(control[k]))
        solve(F == 0, theta_kp1)
        evolution[k+1] = theta_kp1.vector().get_local()

        # preparing for the next iteration
        theta_k.assign(theta_kp1)

    return evolution


def solve_adjoint(evolution, control):
    '''Calculates the solution to the adjoint problem.

    The solution to the adjoint equation is calculated using the explicitly
    given evolution (solution to the forward problem) and the control.
    The objective function is provided implicitly and represented in the adjoint
    solver by penalty_expression.

    For better understanding of the indeces see docs/indexing-diagram.txt.

    Parameters:
        evolution: ndarray
            The coefficients of the solution to the corresponding forward
            problem in the basis of the space V (see solve_forward).
        control: ndarray
            The laser power profile. 

    Returns:
        evolution_adj: ndarray
            The coefficients of the calculated adjoint solution in the basis of
            the space V.
            
    '''

    # initialize state functions
    theta_km1 = Function(V)  # stands for theta[k-1]
    theta_k = Function(V)    # stands for theta[k]
    theta_kp1 = Function(V)  # stands for theta[k+1]

    # initialize adjoint state functions
    p_km1 = Function(V)      # stands for p[k-1]
    p_k = Function(V)        # stands for p[k]

    # FEM equation setup
    p = TrialFunction(V)     # stands for unknown p[k-1]
    v = TestFunction(V)

    Nt = len(control)
    evolution_adj = np.zeros((Nt+1, len(V.dofmap().dofs())))

    # PointSource magnitute precalculation
    sum_ = 0
    for k in range(1, Nt+1):
        theta_k.vector().set_local(evolution[k])
        sum_ += theta_k(target_point) ** pow_
    p_norm = sum_ ** (1 / pow_)
    magnitude_1 = beta_welding * (p_norm - threshold_temp) * sum_ ** (1/pow_ - 1)

    # preparing for the first iteration
    # p[Nt] is never used so does not need initialization
    theta_k.vector().set_local(evolution[Nt])

    # solve backward, i.e. p_k -> p = p_km1, k = Nt, Nt-1, Nt-2, ..., 1
    for k in range(Nt, 0, -1):
        theta_km1.vector().set_local(evolution[k-1])

        F = a(theta_km1, theta_k, p, Constant(control[k-1]))\
          + dt * J_expression(k-1, theta_km1, theta_k)

        if k < Nt:
            # is it correct that the next line can be omitted?
            # theta_next_.vector().set_local(evolution[k+1])
            F += a(theta_k, theta_kp1, p_k, Constant(control[k]))\
               + dt * J_expression(k, theta_k, theta_kp1)

        dF = derivative(F, theta_k, v)
        
        # sometimes rhs(dF) is void which leads to a ValueError
        try:
            A, b = assemble_system(lhs(dF), rhs(dF))
        except ValueError:
            A, b = assemble_system(lhs(dF), Constant(0)*v*dx)

        # calculate and apply PointSource total magnitude
        magnitude_2 = theta_k(target_point) ** (pow_-1)
        ps = PointSource(V, target_point, - magnitude_1 * magnitude_2)
        ps.apply(b)

        solve(A, p_km1.vector(), b)
 
        evolution_adj[k-1] = p_km1.vector().get_local()

        # preparing for the next iteration
        p_k.assign(p_km1)
        theta_kp1.assign(theta_k)
        theta_k.assign(theta_km1)

    return evolution_adj


def Dj(evolution_adj, control):
    '''Calculates the gradient of the cost functional for the given control.

    For further details, see `indexing diagram`.

    Parameters:
        evolution_adj: ndarray
            The evolution in time of the adjoint state.
        control: ndarray
            The laser power coefficient for every time step. 

    Returns:
        Dj: ndarray
            The gradient of the cost functional.
    '''

    p = Function(V)
    z = np.zeros(Nt)

    for i in range(Nt):
        p.vector().set_local(evolution_adj[i])
        z[i] = assemble(p * x[0] * ds(1))
    
    Dj = alpha * (control-control_ref) - laser_pd*z

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

        print(f'{"i.j " :>6} {"s":>14} {"J":>14} {"norm(Dj)":>14}')
        print(f'{simulation.J:36.7e} {simulation.Dj_norm:14.7e}')

        for i in range(iter_max):
            if simulation.Dj_norm < tolerance:
                print(f'norm(Dj) = {simulation.Dj_norm:.7e} < tolerance')
                break

            j = 0
            while True:
                simulation_trial = simulation.descend(step)
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

    print('Terminating.')

    return descent


# def J(evolution, control, as_vector=False, **kwargs):
#     '''Calculates the cost functional.'''

#     # cost = 0.
#     theta = Function(V)
#     theta_ = Function(V)

#     vector = np.zeros(Nt)

#     theta.vector().set_local(evolution[0])
#     for k in range(Nt):
#         theta_.vector().set_local(evolution[k+1])
#         # value = dt * assemble(velocity(theta, theta_)**2 * x[0] * dx)
#         # cost += value
#         vector[k] += dt * assemble(velocity(theta, theta_)**2 * x[0] * dx)
#         vector[k] += dt * assemble(liquidity(theta, theta_)**2 * x[0] * dx)
#         theta.assign(theta_)

#     # vector += 0.5 * alpha * (control-control_ref)**2

#     if as_vector:
#         return vector
#     else:



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

    theta_init = simulation.theta_init
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
                simulation_ = Simulation(control_, theta_init)
                diff = (simulation_.J - simulation.J) / eps

            elif diff_type == 'two_sided':
                control_ = (control - eps * direction).clip(0, 1)
                simulation_ = Simulation(control_, theta_init)
                control__ = (control + eps * direction).clip(0, 1)
                simulation__ = Simulation(control__, theta_init)
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


def velocity(theta_k, theta_kp1, velocity_max=velocity_max):
    theta_avg = implicitness * theta_kp1 + (1 - implicitness) * theta_k
    grad_norm = sqrt(inner(grad(theta_avg), grad(theta_avg)) + DOLFIN_EPS)

    expression = (theta_kp1 - theta_k) / dt / grad_norm

    expression *= conditional(
            ufl.And(ge(theta_k, solidus), lt(theta_kp1, liquidus)), 1., 0.)
    expression *= conditional(le(expression, 0.), 1., 0.)
    expression *= conditional(ge(-expression, velocity_max), 1., 0.)
    expression *= -1

    return expression


def liquidity(theta_k, theta_kp1, implicitness=implicitness):
    '''Returns the "liquidity" penalty expression.

    Parameters:
        theta_k: Function(V)
            Function representing the state from the current time slice.
        theta_kp1: Function(V)
            Function representing the state from the next time slice.
        implicitness: float
            The weight of theta_kp1 in the expression. The global implicitness
            parameter is used by default.

    Returns:
        expression: FEniCS expression
            The expression for the adjoint equation.


    '''
    
    theta_avg = implicitness * theta_kp1 + (1 - implicitness) * theta_k

    expression = theta_avg - Constant(solidus)
    expression *= conditional(ge(expression, 0.), 1., 0.)

    return expression



def J_expression(k, theta_k, theta_kp1):

    e = Constant(beta_liquidity) * liquidity(theta_k, theta_kp1)**2 * x[0] * dx\
      + Constant(beta_velocity) * velocity(theta_k, theta_kp1)**2 * x[0] * dx

    return e


def J_vector(evolution, control):
    '''WARNING: control cost is presented here!'''

    theta = Function(V)
    theta_ = Function(V)

    Nt = len(evolution) - 1
    J_vector_ = np.zeros(Nt)

    theta.vector().set_local(evolution[0])
    for k in range(Nt):
        theta_.vector().set_local(evolution[k+1])
        e = J_expression(k, theta, theta_)
        J_vector_[k] = assemble(e)
        theta.assign(theta_)

    J_vector_ += .5 * alpha *(control-control_ref)**2

    return J_vector_


def J_total(evolution, control):

    J_vector_ = J_vector(evolution, control)
    return dt * J_vector_.sum() + J_welding(evolution, control)


def J_welding(evolution, control):
    sum_ = 0
    theta = Function(V)
    for k in range(1,Nt+1):
        theta.vector().set_local(evolution[k])
        sum_ += np.float_power(theta(target_point), pow_)
    norm = np.float_power(sum_, 1/pow_)
    result = .5 * beta_welding * (norm - threshold_temp)**2

    return result

J = J_total


def penalty_vector(evo, penalty_term):
    theta_k = Function(V)
    theta_kp1 = Function(V)

    Nt = len(evo) - 1
    vector = np.zeros(Nt)

    theta_k.vector().set_local(evo[0])
    for k in range(Nt):
        theta_kp1.vector().set_local(evo[k+1])
        expression = penalty_term(theta_k, theta_kp1)**2 * x[0] * dx
        vector[k] = assemble(expression)
        theta_k.assign(theta_kp1)

    return vector


def temp_at_point_vector(evo, point):
    theta_k = Function(V)
    vector = np.zeros(Nt+1)
    for k in range(Nt+1):
        theta_k.vector().set_local(evo[k])
        vector[k] = theta_k(point)

    return vector