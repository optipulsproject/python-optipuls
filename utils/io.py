import dolfin

def save_as_pvd(evo, V, filename='evo.pvd'):
    outfile = dolfin.File(filename)
    theta = dolfin.Function(V)
    for k, vector in enumerate(evo):
        theta.vector().set_local(vector)
        theta.rename("theta", "temperature")
        outfile << theta, k
