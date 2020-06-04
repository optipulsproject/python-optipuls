from fenics_simulation import *

time_space = np.linspace(0, T, num=Nt, endpoint=True)
control = np.vectorize(u)(time_space, t1=0.005, t2=0.010)
evo = solve_forward(control)
# optionally load pre-calculated evolution
# evo = np.load(args.output+'/evo.npy')

# the end of the heating phase
Nt_ = 200
evo_cooling = evo[Nt_:]

V1 = FunctionSpace(mesh, "DG", 0)
velocity_evo = np.zeros((Nt-Nt_,len(V1.dofmap().dofs())))

theta_p = Function(V)
theta_n = Function(V)
theta_p.vector().set_local(evo_cooling[0])
outfile = File(args.output+'/paraview/velocity_evo.pvd')
for k in trange(Nt-Nt_):
    theta_n.vector().set_local(evo_cooling[k+1])
    expression = (theta_n - theta_p) / dt\
        / sqrt(inner(grad(theta_p), grad(theta_p)))\
        * conditional(
            ufl.And(ge(theta_p,solidus-DOLFIN_EPS),lt(theta_p,liquidus+DOLFIN_EPS)),
            1., 0.)
    velocity = project(expression, V1)
    velocity_evo[k,:] = velocity.vector().get_local()
    theta_p.assign(theta_n)

save_as_npy(velocity_evo, args.output+'/velocity_evo.npy')

# export to paraview
for k in range(Nt-Nt_):
    velocity.vector().set_local(velocity_evo[k])
    velocity.rename("velocity", "isoline velocity function")
    outfile << velocity,k


# Calculate the part of the cost functional
velocity_max = 0.08
total_cost = 0.0
for k in range(Nt-Nt_):
    velocity.vector().set_local(velocity_evo[k])
    expression = conditional(gt(-velocity,velocity_max), 1., 0.)\
               * (-velocity - velocity_max)**2 * x[0] * dx
    space_integral = assemble(expression)
    total_cost += space_integral * dt

print('total cost', total_cost)
