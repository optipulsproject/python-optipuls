import dolfin

class SpaceDomain:
    def __init__(self, R, R_laser, Z,
                 cells_r=25, cells_z=5,
                 checkpoints = [(0.6, 0), (0.4, 0), (0.2, 0), (0.2, 0.5)]):
        self.R = R
        self.R_laser = R_laser
        self.Z = Z
        self._cells_r = cells_r
        self._cells_z = cells_z
        self._checkpoints = checkpoints

    @property
    def mesh(self):
        try:
            return self._mesh
        except AttributeError:
            self._mesh = self._gen_mesh()
            return self._mesh

    @property
    def x(self):
        try:
            return self._x
        except AttributeError:
            self._x = dolfin.SpatialCoordinate(self.mesh)
            return self._x

    @property
    def ds(self):
        try:
            return self._ds
        except AttributeError:
            self._ds = self._gen_ds()
            return self._ds


    def _gen_mesh(self):
        R = self.R
        Z = self.Z
        cells_r = self._cells_r
        cells_z = self._cells_z
        checkpoints = self._checkpoints

        mesh = dolfin.RectangleMesh(
            dolfin.Point(0,0), dolfin.Point(R,Z), cells_r, cells_z)

        for (point_r, point_z) in checkpoints:
            subdomain = dolfin.CompiledSubDomain(
                'x[0] <= point_r * R && x[1] >= point_z * Z',
                point_r=point_r, point_z=point_z, R=R, Z=Z)

            markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim()-1)
            subdomain.mark(markers, True)
            mesh = dolfin.refine(mesh, markers)

        return mesh


    def _gen_ds(self):
        mesh = self.mesh
        R_laser = self.R_laser
        Z = self.Z

        laser_boundary = dolfin.CompiledSubDomain(
            "near(x[1], top_side) && x[0] < R_laser && on_boundary",
            top_side=Z, R_laser=R_laser)

        empty_boundary = dolfin.CompiledSubDomain(
            '((near(x[1], top_side) && x[0]>= R_laser) || near(x[1], bottom_side)) && on_boundary',
            top_side=Z, bottom_side=0, R_laser=R_laser)

        boundary_markers = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
        laser_boundary.mark(boundary_markers, 1)
        empty_boundary.mark(boundary_markers, 2)
        ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        return ds
