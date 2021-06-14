from numpy.polynomial import Polynomial

import ufl
from ufl import ge, gt, lt, le, And

class UFLSpline:
    def __init__(self, spline, ufl_element):
        self._ufl_coef = ufl.Coefficient(ufl_element)

        t = self._ufl_coef
        knots = spline.knots
        coef_array = spline.coef_array

        # assigning left polynomial
        form = ufl.conditional(lt(t, knots[0]), 1., 0.)\
                   * Polynomial(coef_array[0])(t)

        # assigning internal polynomials
        for knot, knot_, coef in\
                zip(knots[:-1], knots[1:], coef_array[1:-1]):
            form += ufl.conditional(And(ge(t, knot), lt(t, knot_)), 1., 0.)\
                        * Polynomial(coef)(t)

        # assigning right polynomial
        form += ufl.conditional(ge(t, knots[-1]), 1., 0.)\
                    * Polynomial(coef_array[-1])(t)

        self._ufl_form = form

    def __call__(self, x):
        return ufl.replace(self._ufl_form, {self._ufl_coef: x})
