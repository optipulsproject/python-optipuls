import dolfin

def save_as_pvd(evo, V, filename='evo.pvd',
                varname='theta', desc='temperature'):

    outfile = dolfin.File(filename)
    func = dolfin.Function(V)
    for k, vector in enumerate(evo):
        func.vector().set_local(vector)
        func.rename(varname, desc)
        outfile << func, k
