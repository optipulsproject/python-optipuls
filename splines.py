import numpy as np
from numpy.polynomial import Polynomial

# from matplotlib import pyplot as plt

# Hermitian cubic interpolation polynomials
# https://en.wikipedia.org/wiki/Cubic_Hermite_spline

p0 = np.array([1,0,-3,2])
m0 = np.array([0,1,-2,1])
p1 = np.array([0,0,3,-2])
m1 = np.array([0,0,-1,1])


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

    # temporary solution
    # derivatives = np.gradient(values,knots)
    derivatives = np.gradient(values)

    left = np.outer(values[:-1],p0) + np.outer(derivatives[:-1],m0)
    right = np.outer(values[1:],p1) + np.outer(derivatives[1:],m1)

    spline = np.zeros((len(knots),4), float)

    spline[:-1] = left + right

    # scaling the polynomials
    for i in range(len(spline)-1):
        x_p = knots[i]
        x_n = knots[i+1]
        p = Polynomial(spline[i], domain=[x_p,x_n], window=[0,1])
        p = p.convert()
        spline[i,:len(p.coef)] = p.coef

    # extrapolation to the right
    # depends on the preferred extrapolation method
    if extrapolation=='constant':
        spline[-1] = values[-1], 0, 0, 0
    elif extrapolation=='linear':
        k = Polynomial(spline[-2]).deriv()(knots[-1])
        spline[-1] = values[-1], k, 0, 0

    return spline
