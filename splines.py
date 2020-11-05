import numpy as np
from numpy.polynomial import Polynomial
from ufl import *
from dolfin import *


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
        for (knot, coefficients) in zip(self.knots, self.coef_array):
            if x < knot:
                return Polynomial(coefficients)(x)
        else:
            coefficients = self.coef_array[-1]
            return Polynomial(coefficients)(x)

    def derivative(self):
        knots = self.knots
        coef_array = np.array(
                [Polynomial(coefficients).deriv().coef
                for coefficients in self.coef_array])
        return Spline(knots, coef_array)

    def ccode(self):
        # ccode: '(x>=k1 && x<k2 ? a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a_0 : 0)'
        pass

    def as_ufl(self):
        def ufl_spline(t):
            knots = self.knots
            coef_array = self.coef_array
            
            # assigning left polynomial
            expression = conditional(lt(t, knots[0]), 1., 0.)\
                       * Polynomial(coef_array[0])(t)

            # assigning internal polynomials
            for knot, knot_, coefficients in\
                    zip(knots[:-1], knots[1:], coef_array[1:-1]):
                expression += conditional(And(ge(t, knot), lt(t, knot_)), 1., 0.)\
                            * Polynomial(coefficients)(t)

            # assigning right polynomial
            expression += conditional(ge(t, knots[-1]), 1., 0.)\
                        * Polynomial(coef_array[-1])(t)

            return expression
        return ufl_spline


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

        # domain and window together with convert() from numpy.polynomial
        # are used as a linear mapping t = (x - knot) / (knot_ - knot) 
        for knot, knot_, p, p_, m, m_, coef in\
                zip(knots[:-1], knots[1:],
                    values[:-1], values[1:],
                    derivatives[:-1], derivatives[1:],
                    coef_array[1:-1]):

            coef_unscaled = h00*p + h01*p_ + (h10*m + h11*m_) * (knot_ - knot)

            p = Polynomial(coef_unscaled, domain=[knot, knot_], window=[0,1])
            p = p.convert()
            coef[:len(p.coef)] = p.coef

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


def gen_hermite_spline(knots, values, extrapolation='constant'):
    '''Generates cubic Hermite spline interpolating given knots and values.

    Parameters:
        knots: (n,) ndarray
        values: (n,) ndarray
             
    Returns:
        spline: (n,4) ndarray
            spline[i] contains four coefficients of the polynomial for the
            half-open interval [knots[i],knots[i+1]) and one extrapolated
            polynomial at the right end

    '''

    # must be changed to monotone spline interpolation in the future
    derivatives = np.gradient(values)

    left = np.outer(values[:-1],p0) + np.outer(derivatives[:-1],m0)
    right = np.outer(values[1:],p1) + np.outer(derivatives[1:],m1)

    spline = np.zeros((len(knots),4), dtype=float)

    spline[:-1] = left + right

    # scaling the polynomials
    for i in range(len(spline)-1):
        x_p = knots[i]
        x_n = knots[i+1]
        p = Polynomial(spline[i], domain=[x_p,x_n], window=[0,1])
        p = p.convert()
        spline[i,:len(p.coef)] = p.coef

    # extrapolation to the right depends on the preferred extrapolation method
    if extrapolation=='constant':
        spline[-1] = values[-1], 0, 0, 0
    elif extrapolation=='linear':
        k = Polynomial(spline[-2]).deriv()(knots[-1])
        spline[-1] = values[-1], k, 0, 0

    return spline


def spline_as_ufl(spline, knots):
    def ufl_spline(t):
        expression = 0

        for i in range(len(spline)-1):
            x_p = Constant(knots[i])
            x_n = Constant(knots[i+1])
            expression += conditional(And(ge(t,x_p),lt(t,x_n)), 1., 0.)\
                        * Polynomial(spline[i])(t)

        expression += conditional(ge(t,Constant(knots[-1])), 1., 0.)\
                    * Polynomial(spline[-1])(t)

        return expression

    return ufl_spline
