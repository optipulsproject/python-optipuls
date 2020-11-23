import numpy as np
from numpy.polynomial import Polynomial
from ufl import ge, gt, lt, le, And
import ufl
import dolfin

import numbers



p0 = h00 = np.array([1, 0, -3,  2])  # 1 - 3*x^2 + 2*x^3
m0 = h10 = np.array([0, 1, -2,  1])  # x - 2*x^2 + x^3
p1 = h01 = np.array([0, 0,  3, -2])  # 3*x^2 - 2*x^3
m1 = h11 = np.array([0, 0, -1,  1])  # - *x^2 + x^3


class Spline:
    def __init__(self, knots, coef_array):
        if not np.all(knots[:-1] <= knots[1:]):
            raise ValueError('knots are not sorted')
        if len(coef_array) != len(knots) + 1:
            raise ValueError('dimension inconsistency')

        self.knots = knots
        self.coef_array = coef_array

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            for (knot, coefficients) in zip(self.knots, self.coef_array):
                if x < knot:
                    return Polynomial(coefficients)(x)
            else:
                coefficients = self.coef_array[-1]
                return Polynomial(coefficients)(x)
        elif isinstance(x, dolfin.function.function.Function):
            ufl_form = self.ufl_form
            t = self._ufl_coef
            return ufl.replace(ufl_form, {t: x})

    def derivative(self):
        knots = self.knots
        coef_array = np.array(
                [Polynomial(coefficients).deriv().coef
                for coefficients in self.coef_array])
        return Spline(knots, coef_array)

    def dump(self, filename='spline.npz'):
        np.savez(filename, knots=self.knots, coef_array=self.coef_array)

    @property
    def ufl_form(self):
        try:
            return self._ufl_form
        except AttributeError:
            self._ufl_form = self.gen_ufl_form()
            return self._ufl_form

    def gen_ufl_form(self):
        element = self.problem.V.ufl_element()
        self._ufl_coef = ufl.Coefficient(element)

        t = self._ufl_coef
        knots = self.knots
        coef_array = self.coef_array

        # assigning left polynomial
        form = ufl.conditional(lt(t, knots[0]), 1., 0.)\
                   * Polynomial(coef_array[0])(t)

        # assigning internal polynomials
        for knot, knot_, coefficients in\
                zip(knots[:-1], knots[1:], coef_array[1:-1]):
            form += ufl.conditional(And(ge(t, knot), lt(t, knot_)), 1., 0.)\
                        * Polynomial(coefficients)(t)

        # assigning right polynomial
        form += ufl.conditional(ge(t, knots[-1]), 1., 0.)\
                    * Polynomial(coef_array[-1])(t)

        return form


class HermiteSpline(Spline):
    # https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    def __init__(self, knots, values, derivatives,
                 extrapolation_left='constant',
                 extrapolation_right='constant'):
        if not np.all(knots[:-1] <= knots[1:]):
            raise ValueError('knots are not sorted')
        if not len(knots) == len(values):
            raise ValueError('dimension inconsistency')
        if not len(knots) == len(derivatives):
            raise ValueError('dimension inconsistency')
        
        coef_array = np.zeros((len(knots)+1, 4), dtype=float)

        for i in range(len(knots) - 1):
            p = hermine_interpolating_polynomial(
                    [knots[i], knots[i+1]],
                    [values[i], values[i+1]],
                    [derivatives[i], derivatives[i+1]])
            coef_array[i+1, :len(p.coef)] = p.coef

        # assigning the left and the right polynomials based on the preferred
        # extrapolation method
        if extrapolation_left=='constant':
            coef_array[0] = values[0], 0, 0, 0
        elif extrapolation_left=='linear':
            k = Polynomial(coef_array[1]).deriv()(knots[0])
            b = values[0] - k*knots[0]
            coef_array[0] = b, k, 0, 0
        if extrapolation_right=='constant':
            coef_array[-1] = values[-1], 0, 0, 0
        elif extrapolation_right=='linear':
            k = Polynomial(coef_array[-2]).deriv()(knots[-1])
            b = values[-1] - k*knots[-1]
            coef_array[-1] = b, k, 0, 0
        
        self.knots = knots
        self.coef_array = coef_array


class NaiveHermiteSpline(HermiteSpline):
    def __init__(self, knots, values,
                 extrapolation_left='constant',
                 extrapolation_right='constant'):

        derivatives = np.gradient(values, knots)

        HermiteSpline.__init__(self, knots, values, derivatives,
                               extrapolation_left, extrapolation_right)


def hermine_interpolating_polynomial(knots, values, derivatives):
    '''Generates Hermite interpolating polynomial.

    Parameters:
        knots: [float]
            Two points on the x-axis.
        values: [float]
            The desired values at the given points.
        derivatives: [float]
            The desired derivatives at the given points.

    Returns:
        polynomial: Polynomial
            The generated Hermite interpolating polynomial.

    '''

    x0, x1 = knots
    p0, p1 = values
    m0, m1 = derivatives

    coef_unscaled = h00*p0 + h01*p1 + (h10*m0 + h11*m1) * (x1 - x0)

    # domain and window together with convert() from numpy.polynomial
    # are used as a linear mapping t = (x - knot) / (knot_ - knot) 
    polynomial = Polynomial(coef_unscaled, domain=[x0, x1], window=[0,1])
    polynomial = polynomial.convert()
    
    return polynomial


def load(filename='spline.npz'):
    npz_obj = np.load(filename)
    return Spline(npz_obj['knots'], npz_obj['coef_array'])
