import unittest

import numpy as np
# from numpy.polynomial import Polynomial

import optipuls.splines as splines


class SplineTestCase(unittest.TestCase):
    def test_hermite_basis_reproducing(self):
        s00 = splines.HermiteSpline([0, 1], [1, 0], [0, 0])
        self.assertTrue((s00.coef_array[1] == splines.h00).all())

        s10 = splines.HermiteSpline([0, 1], [0, 0], [1, 0])
        self.assertTrue((s10.coef_array[1] == splines.h10).all())

        s01 = splines.HermiteSpline([0, 1], [0, 1], [0, 0])
        self.assertTrue((s01.coef_array[1] == splines.h01).all())

        s11 = splines.HermiteSpline([0, 1], [0, 0], [0, 1])
        self.assertTrue((s11.coef_array[1] == splines.h11).all())


    def test_values_on_unit_interval(self):
        # must reproduce x^3 + 2x^2 + 3x - 1
        s = splines.HermiteSpline([0, 1], [-1, 5], [3, 10])

        def p(x):
            return x**3 + 2*x**2 + 3*x - 1

        for x in np.linspace(0, 1, 20, endpoint=True):
            self.assertAlmostEqual(s(x), p(x))


    def test_values(self):
        # must reproduce x^3 + 2x^2 + 3x - 1
        s = splines.HermiteSpline([-1, 1], [-3, 5], [2, 10])

        def p(x):
            return x**3 + 2*x**2 + 3*x - 1

        for x in np.linspace(-1, 1, 20, endpoint=True):
            self.assertAlmostEqual(s(x), p(x))


    def test_linear_extrapolation(self):
        # must reproduce x^3 + 2x^2 + 3x - 1
        s = splines.HermiteSpline([-1, 1], [-3, 5], [2, 10],
                extrapolation_left='linear',
                extrapolation_right='linear')

        self.assertTrue((s.coef_array[0] == [-1, 2, 0, 0]).all())
        self.assertTrue((s.coef_array[-1] == [-5, 10, 0, 0]).all())


    def test_hermine_interpolating_polynomial(self):

        p = splines.Polynomial(2 * np.random.rand(4) - 1)
        knots = 2 * np.random.rand(2) - 1
        values = p(knots)
        derivatives = p.deriv()(knots)

        p_interp = splines.hermine_interpolating_polynomial(
                knots, values, derivatives)

        self.assertTrue((p.domain == p_interp.domain).all())
        self.assertTrue((p.window == p_interp.window).all())
        self.assertTrue(np.allclose(p.coef, p_interp.coef))
