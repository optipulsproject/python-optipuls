import dolfin

class ShiftedExpression(dolfin.UserExpression):
    def __init__(
            self,
            expr_interpolate,
            expr_extrapolate,
            shift,
            bounding_box_tree,
            **kwargs
            ):
        '''Shifts any subclass of dolfin.Expression by the given vector.

        Parameters:
            expr_interpolate (dolfin.Expression):
                The expression evaluated at the points inside of the bounding_box_tree.
            expr_extrapolate (dolfin.Expression):
                The expression evaluated at the points outside of the bounding_box_tree.
            shift (array-like):
                The shift vector.
            bounding_box_tree (dolfin.BoundingBoxTree):
                The bounding_box_tree object of the mesh.

        Example:
            theta_shifted = ShiftedExpression(
                func_interpolate=(lambda p: theta(p)),
                func_extrapolate=(lambda _: 0.),
                shift=(d, 0, 0),
                bounding_box_tree=mesh.bounding_box_tree(),
            )

        '''
        self.expr_interpolate = expr_interpolate
        self.expr_extrapolate = expr_extrapolate
        self.shift = shift
        self.bounding_box_tree = bounding_box_tree
        super().__init__(**kwargs)

    def eval(self, values, x):
        def is_inside(point):
            collisions = self.bounding_box_tree.compute_collisions(point)
            return len(collisions) > 0

        point = dolfin.Point(*(x[i] + self.shift[i] for i in range(3)))
        # if is_inside(point):
        #     values[0] = self.expr_interpolate(point)
        # else:
        #     values[0] = self.expr_extrapolate(point)
        values[0] = self.expr_interpolate(point) if is_inside(point) \
                    else self.expr_extrapolate(point)

    # def value_shape(self):
    #     return (1,)