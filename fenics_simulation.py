from dolfin import *
from mshr import *
import ufl
import numpy as np

# TODO: MY_EPS -> DOLFIN_EPS
# FIX: EmptyBoundary definition 

# Space and time discretization parameters
R = 0.0025
R_laser = 0.0002
Z = 0.0005
T, Nt = 0.005, 100
dt = T/Nt

# Model constants
theta_init = Constant("298")
enthalpy = Constant("397000")
P_YAG = Constant("1600")
absorb = Constant("0.27")
laser_pd = (absorb * P_YAG) / (pi * R_laser**2)
implicitness = Constant("1.0")
implicitness_coef = Constant("0.0")

# Aggregate state
liquidus = 923.0
solidus = 858.0

MY_EPS = 0.000001


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
        return on_boundary and x[1] > Z-MY_EPS and x[0] < R_laser
    
class EmptyBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            ((x[1] > Z-MY_EPS and x[0] >= R_laser) or x[1] < MY_EPS)

    
# Create and refine mesh
mesh = RectangleMesh(Point(0,0),Point(R,Z), 25, 5)

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

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Laser control multiplier
def u(t, t1=0.005, t2=0.010):
    return conditional(lt(t, t1), Constant("1.0"),\
               conditional(lt(t, t2), (t2-t)/(t2-t1), Constant("0.0")))

def control_to_vector(func=u):
    '''Convertes the control function to discrete vector.
    TODO: rewrite with *args'''

    time_space = np.linspace(0, T, num=Nt+1, endpoint=True)
    func_vectorized = np.vectorize(func)
    return func_vectorized(time_space)
    

# Parameter functions
def c(theta):    
    # c_lin = conditional(gt(theta, liquidus), 1085, 822.719+0.281273*theta)
    c_lin = conditional(gt(theta, liquidus), 1085., 0.) +\
            conditional(gt(theta, liquidus), 0., 1.0) * (822.719+0.281273*theta)
    c_melting = conditional(gt(theta, liquidus), Constant("0.0"),\
        enthalpy*exp(-0.5*((theta-890.5)/8)**2) / sqrt(2*pi*64))
    return c_lin #+ c_melting
    
def rho(theta):
    return Constant("2800")

def s(theta):
    return c(theta)*rho(theta)

def lamb(theta):
    k_solid, b_solid = 0.05, 163.2
    k_rad_mash, b_rad_mash = 3.14, -2486.8
    k_ax_mash, b_ax_mash = -1.63, 1605.2

    solid = (theta < 858)
    mash = (858 <= theta) * (theta < 923)
    liquid = (923 < theta)

    lambda_rad = conditional(lt(theta,858),\
        (k_solid * theta + b_solid), Constant("0.0"))
    lambda_ax = conditional(lt(theta,858),\
        (k_solid * theta + b_solid), Constant("0.0"))
    lambda_rad += conditional(ufl.And(ge(theta,858), le(theta,923)),\
        k_rad_mash * theta + b_rad_mash, Constant("0.0"))    
    lambda_ax += conditional(ufl.And(ge(theta,858), lt(theta, 923)),\
        k_ax_mash * theta + b_ax_mash, Constant("0.0"))
    lambda_rad += conditional(ge(theta,923), Constant("410."), Constant("0.0"))
    lambda_ax += conditional(ge(theta,923), Constant("100."), Constant("0.0"))

    # return as_matrix([[lambda_rad, Constant("0.0")],\
    #         [Constant("0.0"), lambda_ax]])
    return as_matrix([[Constant("150.0"), Constant("0.0")],\
            [Constant("0.0"), Constant("150.0")]])

def LaserBC(theta, multiplier):
    return laser_pd * multiplier \
           - 5 * (theta-theta_init)\
           - 2.26 * 10**(-9) * (theta**4-theta_init**4)

def EmptyBC(theta):
    return -5*(theta-theta_init) - 2.26 * 10**(-9)*(theta**4-theta_init**4)

theta = Function(V)
theta_old = Function(V)
v = TestFunction(V)
theta_old = project(theta_init, V)
# n = FacetNormal(mesh)

control = control_to_vector() 

# Output file
outfile = File('../output/fenics/theta.pvd')


# Theta = np.zeros((Nt+1,len(V.dofmap().dofs())))
# # Loop over time steps
# for k in range(Nt+1):
#     F = c(theta_old)*rho(theta_old)*(theta-theta_old)*v*x[0]*dx \
#       + dt*inner(lamb(theta_old) * grad(0.5*(theta+theta_old)), grad(v))*x[0]*dx \
#       - dt*LaserBC(0.5*(theta+theta_old),Constant(control[k]))*v*x[0]*ds(1) \
#       - dt*EmptyBC(0.5*(theta+theta_old))*v*x[0]*ds(2)

#     solve(F == 0, theta)
#     Theta[k,:] = theta.vector().get_local()
#     theta_old.assign(theta)

#     # Write solution to file
#     theta.rename("theta", "temperature")
#     outfile << theta,k

def solve_forward(control):
    '''Calculates the solution to the forward problem with the given control.

    Parameters:
        control: ndarray
            The laser power coefficient for every time step. 

    Returns:
        theta_coefficients: ndarray
            The coefficients of the calculated solution in the basis of space V.
            
    '''

    theta_n = Function(V)
    theta_p = Function(V)
    theta_p = project(theta_init, V)
    v = TestFunction(V)

    evolution = np.zeros((Nt+1,len(V.dofmap().dofs())))
    evolution[0,:] = theta_p.vector().get_local()

    theta_m = implicitness*theta_n + (1-implicitness)*theta_p

    for k in range(Nt):
        F = s(theta_p) * (theta_n-theta_p) * v * x[0] * dx \
          + dt * inner(lamb(theta_p) * grad(theta_m), grad(v)) * x[0] * dx \
          - dt * LaserBC(theta_m, Constant(control[k])) * v * x[0] * ds(1) \
          - dt * EmptyBC(theta_m) * v * x[0] * ds(2)

        solve(F == 0, theta_n)
        evolution[k+1,:] = theta_n.vector().get_local()
        theta_p.assign(theta_n)

    return evolution

def save_as_npy(evolution, filename='evolution.npy'):
    np.save(filename, evolution)

def save_as_pvd(evolution, filename='evolution.pvd'):
    outfile = File(filename)
    theta = Function(V)
    for k in range(Nt+1):
        theta.vector().set_local(evolution[k])
        theta.rename("theta", "temperature")
        outfile << theta,k


def solve_adjoint(evolution, control):

    p_prev = TrialFunction(V)
    p_next = Function(V)
    p_next = project(Constant("0.0"), V)
    p = Function(V)
    v = TestFunction(V)

    evolution_adj = np.zeros((Nt+1,len(V.dofmap().dofs())))
    evolution_adj[Nt,:] = p_next.vector().get_local()

    # solve backward, i.e. p_next -> p_prev

    theta_next = Function(V)
    theta_prev = Function(V)
    theta_next_ = Function(V)

    # theta_next_ = project(Constant("0.0"), V)


    for k in range(Nt,0,-1):
        theta_next.vector().set_local(evolution[k])
        theta_prev.vector().set_local(evolution[k-1])
        print('k=',k)

        F = Constant("0.5") * dt * inner(grad(theta_next),grad(theta_next)) * x[0] * dx\
          + s(theta_prev) * (theta_next - theta_prev) * p_prev * x[0] * dx\
          + dt * inner(lamb(theta_prev) * grad(theta_next), grad(p_prev)) * x[0] * dx\
          - dt * LaserBC(theta_next, Constant(control[k-1])) * p_prev * x[0] * ds(1)\
          - dt * EmptyBC(theta_next) * p_prev * x[0] * ds(2)
          # \
          # + s(theta_next) * (theta_next_ - theta_next) * p_next * x[0] * dx\
          # + dt * inner(lamb(theta_next) * grad(theta_next_), grad(p_next)) * x[0] * dx\
          # - dt * LaserBC(theta_next_, Constant(control[k])) * p_next * x[0] * ds(1)\
          # - dt * EmptyBC(theta_next_) * p_next * x[0] * ds(2)

        if k < Nt:
            # print('I am inside')
            theta_next_.vector().set_local(evolution[k+1])
            F += s(theta_next) * (theta_next_ - theta_next) * p_next * x[0] * dx\
               + dt * inner(lamb(theta_next) * grad(theta_next_), grad(p_next)) * x[0] * dx\
               - dt * LaserBC(theta_next_, Constant(control[k])) * p_next * x[0] * ds(1)\
               - dt * EmptyBC(theta_next_) * p_next * x[0] * ds(2)

        dF = derivative(F,theta_next,v)
        solve(lhs(dF)==rhs(dF),p)
        evolution_adj[k-1,:] = p.vector().get_local()
        p_next.assign(p)


    return evolution_adj


def solve_adjoint_old(evolution, control):
    p_ = TrialFunction(V)
    p = Function(V)
    p_old = project(Constant("0.0"), V)
    v = TestFunction(V)

    evolution_adj = np.zeros((Nt+1,len(V.dofmap().dofs())))
    evolution_adj[Nt,:] = p_old.vector().get_local()


    for k in range(Nt-1,-1,-1):
        theta_new = Function(V)
        theta_old = Function(V)
        theta_older = Function(V)

        theta_new.vector().set_local(evolution[k])
        theta_old.vector().set_local(evolution[k+1])
        if k <= Nt-2: theta_older.vector().set_local(evolution[k+2])

        F = c(theta_new)*rho(theta_new)*(theta_old-theta_new)*p_*x[0]*dx \
          + dt*inner(lamb(theta_new) * grad(0.5*(theta_old+theta_new)), grad(p_))*x[0]*dx \
          - dt*LaserBC(0.5*(theta_old+theta_new),Constant(control[k]))*p_*x[0]*ds(1) \
          - dt*EmptyBC(0.5*(theta_old+theta_new))*p_*x[0]*ds(2)
          # + c(theta_new)*rho(theta_new) * p_*v*x[0]*dx 

        R = - dt * inner(grad(theta_old), grad(v)) * x[0] * dx

        if k <= Nt-2:
            R -= derivative(\
                  c(theta_old)*rho(theta_old)*(theta_older-theta_old)*p_old*x[0]*dx \
                  + dt*inner(lamb(theta_old) * grad(0.5*(theta_older+theta_old)), grad(p_old))*x[0]*dx \
                  - dt*LaserBC(0.5*(theta_older+theta_old),Constant(control[k]))*p_old*x[0]*ds(1) \
                  - dt*EmptyBC(0.5*(theta_older+theta_old))*p_old*x[0]*ds(2),\
                  theta_old,v)

            # R += derivative(c(theta_old)*rho(theta_old) * p_old*(theta_older-theta_old)*dx, theta_old, v)

        G = derivative(F,theta_old,v)
        solve(G==R, p)

        evolution_adj[k,:] = p.vector().get_local()
        p_old.assign(p)

    return evolution_adj


evolution = solve_forward(control)
save_as_pvd(evolution,'../output/evo.pvd')
evolution_adj = solve_adjoint(evolution,control)
save_as_pvd(evolution_adj,'../output/evo_adj.pvd')

# save_as_npy(evolution, '../output/evo.npy')
# evolution = np.load('../output/evo.npy')
