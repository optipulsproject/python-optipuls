import dolfin
from dolfin import DOLFIN_EPS


# Space dimensions
R = 0.0015
R_laser = 0.0002
Z = 0.0005

class Domain_2(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.4 * R

class Domain_3(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.2 * R

class Domain_4(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.1 * R

class Domain_5(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.05 * R and x[1] > 0.5 * Z

class LaserBoundary(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > Z-DOLFIN_EPS and x[0] < R_laser

class EmptyBoundary(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            ((x[1] > Z-DOLFIN_EPS and x[0] >= R_laser) or x[1] < DOLFIN_EPS)


# Create and refine mesh
mesh = dolfin.RectangleMesh(dolfin.Point(0,0), dolfin.Point(R,Z), 200, 4)

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
