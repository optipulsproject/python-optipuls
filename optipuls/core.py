import dolfin
from dolfin import dx, Constant, DOLFIN_EPS
import ufl
from ufl import inner, grad, conditional, ge, gt, lt, le, And
import numpy as np
from matplotlib import pyplot as plt


def laser_bc(control_k, laser_pd):
    return laser_pd * Constant(control_k)


def cooling_bc(theta, temp_amb, convection_coeff, radiation_coeff):
    return - convection_coeff * (theta - temp_amb)\
           - radiation_coeff * (theta**4 - temp_amb**4)


def norm2(dt, vector):
    '''Calculates the squared L2[0,T] norm.'''
    return dt * sum(vector**2)


def norm(dt, vector):
    '''Calculates L2[0,T] norm.'''
    return np.sqrt(norm2(dt, vector))

def integral2(form, x):
    return form**2 * x[0] * dx

def avg(u_k, u_kp1, implicitness):
    return implicitness * u_kp1 + (1 - implicitness) * u_k


def a(u_k, u_kp1, v, control_k,
      vhc, kappa, cooling_bc, laser_bc, dt, implicitness, x, ds):
    u_avg = avg(u_k, u_kp1, implicitness)

    a_ = vhc(u_k) * (u_kp1 - u_k) * v * x[0] * dx\
       + dt * inner(kappa(u_k) * grad(u_avg), grad(v)) * x[0] * dx\
       - dt * laser_bc(control_k) * v * x[0] * ds(1)\
       - dt * cooling_bc(u_avg) * v * x[0] * (ds(1) + ds(2) + ds(4))

    return a_


def solve_forward(a, V, theta_init, control):
    '''Calculates the solution to the nonlinear time-dependent forward problem
    with the given control.

    This is a low level function, which is normally not supposed to be used
    directly by the end user. High level interfaces such as Simulation.evo must
    be used instead.

    For further details, see the indexing diagram.

    Parameters:
        a: (u_k, u_kp1, v, control_k) -> UFL form
            The RHS variational form.
        V: dolfin.FunctionSpace
            The FEM space of the problem being solved.
        theta_init: dolfin.Function(V)
            The initial state.
        control: ndarray
            The laser power coefficient for every time step. 

    Returns:
        evo: ndarray
            The coefficients of the calculated solution in the basis of
            the space V at each time step including the given initial state.
            
    '''

    # initialize state functions
    theta_k = dolfin.Function(V)    # stands for theta[k]
    theta_kp1 = dolfin.Function(V)  # stands for theta[k+1]

    # FEM equation setup
    v = dolfin.TestFunction(V)

    Nt = len(control)
    evo = np.zeros((Nt+1, len(V.dofmap().dofs())))

    # preparing for the first iteration
    theta_k.assign(theta_init)
    evo[0] = theta_k.vector().get_local()

    # solve forward, i.e. theta_k -> theta p_kp1, k = 0, 1, 2, ..., Nt-1
    for k in range(Nt):
        F = a(theta_k, theta_kp1, v, control[k])
        dolfin.solve(F == 0, theta_kp1)
        evo[k+1] = theta_kp1.vector().get_local()

        # preparing for the next iteration
        theta_k.assign(theta_kp1)

    return evo


def solve_adjoint(evo, control, ps_magnitude, target_point, a, V, j):
    '''Calculates the solution to the adjoint problem.

    The solution to the adjoint equation is calculated using the explicitly
    given evolution (solution to the forward problem) and the control.

    For better understanding of the indeces see docs/indexing-diagram.txt.

    Parameters: OUTDATED
        V: dolfin.FunctionSpace
            The FEM space of the problem being solved.
        evo: ndarray
            The coefficients of the solution to the corresponding forward
            problem in the basis of the space V (see solve_forward).
        control: ndarray
            The laser power profile.
        beta_welding
        target_point: dolfin.Point
        threshold_temp
        penalty_term_combined

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

    # preparing for the first iteration
    theta_k.vector().set_local(evo[Nt])

    # solve backward, i.e. p_k -> p = p_km1, k = Nt, Nt-1, Nt-2, ..., 1
    for k in range(Nt, 0, -1):
        theta_km1.vector().set_local(evo[k-1])

        F = a(theta_km1, theta_k, p, control[k-1])\
          + j(k-1, theta_km1, theta_k)

        if k < Nt:
            F += a(theta_k, theta_kp1, p_k, control[k])\
               + j(k, theta_k, theta_kp1)


        dF = dolfin.derivative(F, theta_k, v)
        
        # sometimes rhs(dF) is void which leads to a ValueError
        try:
            A, b = dolfin.assemble_system(dolfin.lhs(dF), dolfin.rhs(dF))
        except ValueError:
            A, b = dolfin.assemble_system(dolfin.lhs(dF), Constant(0)*v*dx)

        # apply welding penalty as a point source
        point_source = dolfin.PointSource(V, target_point, ps_magnitude[k-1])
        point_source.apply(b)

        dolfin.solve(A, p_km1.vector(), b)
 
        evo_adj[k-1] = p_km1.vector().get_local()

        # preparing for the next iteration
        p_k.assign(p_km1)
        theta_kp1.assign(theta_k)
        theta_k.assign(theta_km1)

    return evo_adj


def Dj(evo_adj, control, V, control_ref, beta_control, beta_welding, laser_pd,
       x, ds):
    '''Calculates the gradient of the cost functional for the given control.


    Notice that the impact of all the penalty terms (except the control penalty
    term) on the gradient is already reflected in the adjoint solution.

    For further details, see `indexing diagram`.

    Parameters:
        V: dolfin.FunctionSpace
            The FEM space of the adjoint solution.
        evo_adj: ndarray
            The evolution in time of the adjoint state.
        control: ndarray
            The laser power coefficient for every time step. 

    Returns:
        Dj: ndarray
            The gradient of the cost functional.
    '''

    p = dolfin.Function(V)
    Nt = len(evo_adj) - 1
    z = np.zeros(Nt)

    for i in range(Nt):
        p.vector().set_local(evo_adj[i])
        z[i] = dolfin.assemble(p * x[0] * ds(1))
    
    Dj = beta_control * (control - control_ref) - laser_pd*z

    return Dj


def penalty_welding(evo, control,
                    V, beta_welding, target_point, threshold_temp, pow_):
    '''Penalty due to the maximal temperature at the target point.'''

    sum_ = 0
    theta = dolfin.Function(V)
    for k in range(len(evo)):
        theta.vector().set_local(evo[k])
        sum_ += np.float_power(theta(target_point), pow_)
    norm = np.float_power(sum_, 1/pow_)
    result = .5 * beta_welding * (norm - threshold_temp)**2

    return result


def vectorize_penalty_term(evo, V, penalty_term, *args, **kwargs):
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
        penalty = penalty_term(k, theta_k, theta_kp1, *args, **kwargs)
        if penalty:
            penalty_vector[k] = dolfin.assemble(penalty)
        theta_k.assign(theta_kp1)

    return penalty_vector


def temp_at_point_vector(evo, V, point):
    '''Provides the temperature evolution at a given point.

    Parameters:
        evo: ndarray
            The coefficients of the solution to the corresponding forward
            problem in the basis of the space V (see solve_forward).
        V: dolfin.FunctionSpace
            The FEM space of the problem being solved.
        point: dolfin.Point
            The point to check temperature at. 

    Returns:
        temp_vector: ndarray
            Contains temperatures at the given point for every time step.

    '''
    Nt = len(evo) - 1

    theta_k = dolfin.Function(V)
    temp_vector = np.zeros(Nt+1)
    for k in range(Nt+1):
        theta_k.vector().set_local(evo[k])
        temp_vector[k] = theta_k(point)

    return temp_vector


def velocity(theta_k, theta_kp1, dt, liquidus, solidus, velocity_max):
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


def liquidity(theta_k, theta_kp1, solidus, implicitness=1.):
    theta_avg = avg(theta_k, theta_kp1, implicitness)
    form = theta_avg - Constant(solidus)
    # filter to liquid parts
    form *= conditional(ge(form, 0.), 1., 0.)

    return form


def compute_evo_vel(evo, V, V1, dt, liquidus, solidus, velocity_max):
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

        func = dolfin.project(
                velocity(
                    theta_k, theta_kp1, dt, liquidus, solidus, velocity_max),
                V1)

        evo_vel[k] = func.vector().get_local()

        theta_k.assign(theta_kp1)

    return evo_vel


def compute_ps_magnitude(
        evo, V, target_point, threshold_temp, beta_welding, pow_):
    theta_kp1 = dolfin.Function(V)

    Nt = len(evo) - 1
    magnitude = np.zeros(Nt)

    sum_ = 0
    for k in range(0, Nt):
        theta_kp1.vector().set_local(evo[k+1])
        sum_ += theta_kp1(target_point) ** pow_
        magnitude[k] = theta_kp1(target_point) ** (pow_ - 1)

    p_norm = sum_ ** (1 / pow_)
    magnitude_common = beta_welding * (p_norm - threshold_temp)\
                     * sum_ ** (1/pow_ - 1)
    magnitude *= - magnitude_common

    return magnitude


def project_Dj(Dj, control):
    positive = lambda A: np.maximum(A, 0)
    negative = lambda A: np.minimum(A, 0)
    filter_positive = lambda A: np.where(control < 1, A, positive(A))
    filter_negative = lambda A: np.where(control > 0, A, negative(A))

    PDj = filter_negative(filter_positive(Dj))

    return PDj
