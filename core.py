from dolfin import *
from mshr import *
import ufl
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

# from tqdm import trange

import splines as spl

set_log_level(40)

# Space and time discretization parameters
R = 0.0025
R_laser = 0.0002
Z = 0.0005
T, Nt = 0.015, 30
dt = T/Nt

# Model constants
theta_amb = Constant("295")
enthalpy = Constant("397000")
P_YAG = 1600.
absorb = 0.135
laser_pd = (absorb * P_YAG) / (pi * R_laser**2)
implicitness = Constant("1.0")
implicitness_coef = Constant("0.0")

# Optimization parameters
alpha = 0.0001
beta = 10**6
iter_max = 5
tolerance = 10**-18

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

boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)

laser_boundary = LaserBoundary()
laser_boundary.mark(boundary_markers, 1)

empty_boundary = EmptyBoundary()
empty_boundary.mark(boundary_markers, 2)

sym_axis_boundary = SymAxisBoundary()
sym_axis_boundary.mark(boundary_markers, 3)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)


# discete values for c(theta) Hermite interpolation spline
knots_c = np.array([273,373,473,573,673,773,858,890,923,973,1073])
values_c = np.array([896,925,958,990,1016,1040,1058,11000,1070,1070,1070])

spline_c = spl.gen_hermite_spline(knots_c,values_c)

def c(theta):
    result = 0

    for i in range(len(spline_c)-1):
        x_p = Constant(knots_c[i])
        x_n = Constant(knots_c[i+1])
        result += conditional(ufl.And(ge(theta,x_p),lt(theta,x_n)), 1., 0.)\
                    * Polynomial(spline_c[i])(theta)

    result += conditional(ge(theta,Constant(knots_c[-1])), 1., 0.)\
                * Polynomial(spline_c[-1])(theta)
    
    return result

knots_rho = np.array([273,373,473,573,673,773,858,923,973,1073,1173])
values_rho = np.array([2750,2730,2710,2690,2670,2650,2630,2450,2440,2430,2420])

spline_rho = spl.gen_hermite_spline(\
                knots_rho, values_rho, extrapolation='linear')

def rho(theta):
    result = 0

    for i in range(len(spline_rho)-1):
        x_p = Constant(knots_rho[i])
        x_n = Constant(knots_rho[i+1])
        result += conditional(ufl.And(ge(theta,x_p),lt(theta,x_n)), 1., 0.)\
                    * Polynomial(spline_rho[i])(theta)

    result += conditional(ge(theta,Constant(knots_rho[-1])), 1., 0.)\
                * Polynomial(spline_rho[-1])(theta)
    
    return result


def s(theta):
    return c(theta) * rho(theta)

# discete values for lamb(theta) Hermite interpolation spline
knots_lamb = np.array([273,373,473,573,673,773,858,923])
values_lamb_rad = np.array([177,182,187,193,198,200,200,400])
values_lamb_ax = np.array([177,182,187,193,198,200,200,100])

spline_lamb_rad = spl.gen_hermite_spline(knots_lamb,values_lamb_rad)
spline_lamb_ax = spl.gen_hermite_spline(knots_lamb,values_lamb_ax)


def lamb(theta):
    radial = 0

    for i in range(len(spline_lamb_rad)-1):
        x_p = Constant(knots_lamb[i])
        x_n = Constant(knots_lamb[i+1])
        radial += conditional(ufl.And(ge(theta,x_p),lt(theta,x_n)), 1., 0.)\
                    * Polynomial(spline_lamb_rad[i])(theta)

    radial += conditional(ge(theta,Constant(knots_lamb[-1])), 1., 0.)\
                * Polynomial(spline_lamb_rad[-1])(theta)

    axial = 0

    for i in range(len(spline_lamb_ax)-1):
        x_p = Constant(knots_lamb[i])
        x_n = Constant(knots_lamb[i+1])
        axial += conditional(ufl.And(ge(theta,x_p),lt(theta,x_n)), 1., 0.)\
                    * Polynomial(spline_lamb_ax[i])(theta)

    axial += conditional(ge(theta,Constant(knots_lamb[-1])), 1., 0.)\
                * Polynomial(spline_lamb_ax[-1])(theta)
    
    return as_matrix([[radial, Constant("0.0")],\
                      [Constant("0.0"), axial]])


def laser_bc(intensity):
    return laser_pd * intensity


def cooling_bc(theta):
    return - 20 * (theta-theta_amb)\
           - 2.26 * 10**(-9) * (theta**4-theta_amb**4)


def u(t, t1=0.005, t2=0.010):
    if t < t1:
        return 1.
    elif t < t2:
        return (t2-t)/(t2-t1)
    else:
        return 0.
 

def a(u, u_, v, intensity):
    u_m = implicitness * u_ + (1-implicitness) * u

    a = s(u) * (u_ - u) * v * x[0] * dx\
      + dt * inner(lamb(u) * grad(u_m), grad(v)) * x[0] * dx\
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

    theta = Function(V)
    theta_ = Function(V)
    v = TestFunction(V)

    theta.assign(theta_init)

    Nt = len(control)
    evolution = np.zeros((Nt+1, len(V.dofmap().dofs())))
    evolution[0,:] = theta.vector().get_local()

    theta_m = implicitness*theta_ + (1-implicitness)*theta

    for k in range(Nt):
        F = a(theta, theta_, v, Constant(control[k]))
        solve(F == 0, theta_)
        evolution[k+1,:] = theta_.vector().get_local()
        theta.assign(theta_)

    return evolution


def save_as_npy(evolution, filename='evolution.npy'):
    np.save(filename, evolution)

def save_as_pvd(evolution, filename='evolution.pvd'):
    outfile = File(filename)
    theta = Function(V)
    for k in range(Nt+1):
        theta.vector().set_local(evolution[k])
        theta.rename("theta", "temperature")
        outfile << theta, k


def solve_adjoint(evolution, control):

    p_prev = TrialFunction(V)
    p_next = Function(V)
    p = Function(V)
    theta_next = Function(V)
    theta_prev = Function(V)
    theta_next_ = Function(V)
    v = TestFunction(V)

    Nt = len(control)
    evolution_adj = np.zeros((Nt+1, len(V.dofmap().dofs())))
    evolution_adj[Nt,:] = p_next.vector().get_local()

    # solve backward, i.e. p_next -> p_prev
    theta_next.vector().set_local(evolution[Nt])
    for k in range(Nt,0,-1):
        theta_prev.vector().set_local(evolution[k-1])

        F = a(theta_prev, theta_next, p_prev, Constant(control[k-1]))\
          + dt * velocity(theta_prev, theta_next)**2 * x[0] * dx

        if k < Nt:
            theta_next_.vector().set_local(evolution[k+1])
            F += a(theta_next, theta_next_, p_next, Constant(control[k]))\
               + dt * velocity(theta_next,theta_next_)**2 * x[0] * dx

        dF = derivative(F,theta_next,v)
        solve(lhs(dF)==rhs(dF),p)
        evolution_adj[k-1,:] = p.vector().get_local()
        p_next.assign(p)

        theta_next_.assign(theta_next)
        theta_next.assign(theta_prev)

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
    
    Dj = - laser_pd*z

    return Dj


def gradient_descent(control, iter_max=100, s=512.):
    '''Calculates the optimal control.

    Parameters:
        control: ndarray
            Initial guess.
        iter_max: integer
            The maximal allowed number of iterations.

    Returns:
        control_optimal: ndarray

    '''

    # TODO: change the breaking condition

    evolution = solve_forward(control)
    cost = J(evolution, control)
    cost_next = cost

    controls_iterations = [control]

    print('{:>4} {:>12} {:>14} {:>14}'.format('i', 's', 'j', 'norm'))

    for i in range(iter_max):
        
        evolution_adj = solve_adjoint(evolution, control)
        D = Dj(evolution_adj, control)
        norm = dt * np.sum(D**2)
        
        if norm < tolerance:
            print('norm < tolerance')
            break

        first_try = True
        while (cost_next >= cost) or first_try:
            control_next = np.clip(control - s*D, 0, 1)
            evolution_next = solve_forward(control_next)
            cost_next = J(evolution_next, control_next)
            print('{:4} {:12.6f} {:14.7e} {:14.7e}'.\
                format(i, s, cost_next, norm))
            if not first_try: s /= 2
            first_try = False

        s *= 2
        control = control_next
        cost = cost_next
        evolution = evolution_next
        controls_iterations.append(control)

    return controls_iterations


def J(evolution, control):
    '''Calculates the cost functional.'''

    cost = 0.
    theta = Function(V)
    theta_ = Function(V)

    theta.vector().set_local(evolution[0])
    for k in range(0,Nt):
        theta_.vector().set_local(evolution[k+1])
        cost += dt * assemble(velocity(theta,theta_)**2 * x[0] * dx)
        theta.assign(theta_)

    return cost


def gradient_test(control, n=10, diff_type='forward', eps_init=1.):
    '''Checks the accuracy of the calculated gradient Dj.

    The scalar product (Dj,direction) is calculated and
    compared to the finite difference expression for J w.r.t. direction,
    the absolute error and the relative error are calculated.

    Every iteration epsilon is divided by two.

    Parameters:
        control: ndarray
            Control used for testing.
        n: integer
            Number of tests.
        diff_type: 'forward' (default) or 'two_sided'
            The exact for of the finite difference expression.
        eps_init: float
            The initial value of epsilon.

    Returns:
        epsilons: array-like
            Epsilons used for testing.
        deltas: array-like
            Relative errors.

    '''

    evo = solve_forward(control)
    evo_adj = solve_adjoint(evo, control)
    time_space = np.linspace(0, T, num=Nt, endpoint=True) 
    direction = np.cos(time_space*np.pi / (2*T))
    # direction = np.random.rand(Nt)

    D = Dj(evo_adj, control)
    print('{:>16}{:>16}{:>16}{:>16}{:>16}'.\
        format('epsilon', '(Dj,direction)', 'finite diff', 'absolute error',
               'relative error'))

    epsilons = [eps_init * 2**-k for k in range(n)]
    deltas = []

    scalar_product = dt * np.sum(D*direction)

    for eps in epsilons:
        
        if diff_type == 'forward':
            control_eps = control + eps * direction
            evo_eps = solve_forward(control_eps)
            diff = (J(evo_eps, control_eps) - J(evo, control)) / eps
        elif diff_type == 'two_sided':
            control_plus_eps = control + eps * direction
            evo_plus_eps = solve_forward(control_plus_eps)
            control_minus_eps = control - eps * direction
            evo_minus_eps = solve_forward(control_minus_eps)
            diff = (J(evo_plus_eps, control_plus_eps)
                       - J(evo_minus_eps, control_minus_eps)) / (2 * eps)            
        
        delta_abs = scalar_product - diff
        delta_rel = delta_abs / scalar_product
        deltas.append(delta_rel)
        print('{:16.8e}{:16.8e}{:16.8e}{:16.8e}{:16.8e}'.\
            format(eps, scalar_product, diff, delta_abs, delta_rel))

    return epsilons, deltas


def velocity(theta, theta_):
    theta_m = implicitness*theta_ + (1-implicitness)*theta
    grad_norm = sqrt(inner(grad(theta_m), grad(theta_m)))
    grad_norm2 = inner(grad(theta_m), grad(theta_m))
    grad_norm3 = inner(grad(theta_m), grad(theta_m))**2

    expression = (theta_ - theta) / dt\
        / grad_norm\
        * conditional(ge(grad_norm,DOLFIN_EPS),1.,0.)\
        * conditional(ufl.And(ge(theta_m,solidus),lt(theta_m,liquidus)),1.,0.)\

    return beta * expression