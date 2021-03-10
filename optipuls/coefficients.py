'''
This module constructs the temperature-dependent coefficients of the
quasi-linear heat equation using spline fitting to the given reference table
values.

The following functions are constructed for a given material:
- volumetric heat capacity
- radial thermal conductivity
- axial thermal conductivity

'''

import numpy as np

from . import splines as spl


def construct_vhc_spline(material):
    '''Makes spline fitting of the volumetric heat capacity coefficient for a
    given material.'''


    melting_enthalpy = material.melting_enthalpy
    solidus = material.solidus
    liquidus = material.liquidus
    mid = 0.5 * (solidus + liquidus)
    knots = material.knots
    heat_capacity = material.heat_capacity
    density = material.density

    # filter to the values in the solid state
    knots_solid = [k for k in knots if k <= solidus]
    density_solid = [d for (d,k) in zip(density,knots) if k <= solidus]
    vhc_solid = [c*d for (c,d,k) in zip(heat_capacity,density,knots) if k <= solidus]

    # filter to the values in the liquid state
    knots_liquid = [k for k in knots if k >= liquidus]
    density_liquid = [d for (d,k) in zip(density,knots) if k >= liquidus]
    vhc_liquid = [c*d for (c,d,k) in zip(heat_capacity,density,knots) if k >= liquidus]

    # apply linear fitting
    k_solid, b_solid = np.polyfit(knots_solid, vhc_solid, deg=1)
    k_liquid, b_liquid = np.polyfit(knots_liquid, vhc_liquid, deg=1)

    # interpolate the solidus-liquidus corridor with a cubic polynomial
    polynomial_mid = spl.hermine_interpolating_polynomial(
            knots=[solidus, liquidus],
            values=[k_solid * solidus + b_solid, k_liquid * liquidus + b_liquid],
            derivatives=[k_solid, k_liquid])

    # approximate the density at the middle point linearly and
    # evaluate the volumetric melting enthalpy
    knot_left = knots_solid[-1]
    density_left = density_solid[-1]
    knot_right = knots_liquid[0]
    density_right = density_liquid[0]
    density_mid = density_left + (density_right - density_left)\
                  * (mid - knot_right) / (knot_left - knot_right)
    melting_enthalpy_volumetric = melting_enthalpy * density_mid
    
    # calculate the integral of the previously interpolated polynomial
    # over the solidus-liquidus corridor and evaluate the needed correction
    # to reach specified volumetric melting enthalpy value
    I = polynomial_mid.integ()(liquidus) - polynomial_mid.integ()(solidus)
    h = 2 * (melting_enthalpy_volumetric - I) / (liquidus - solidus)

    # generate left and right cubic polynomials for correction
    # preserving smoothness
    polynomial_left = spl.hermine_interpolating_polynomial(
            knots=[solidus, mid],
            values=[0, h],
            derivatives=[0, 0])
    polynomial_right = spl.hermine_interpolating_polynomial(
            knots=[mid, liquidus],
            values=[h, 0],
            derivatives=[0, 0])

    # construct the final spline (including correction)
    # by specifying its knots and coefficients explicitly
    vhc_spline = spl.Spline(
            [solidus, mid, liquidus],
            [
                [b_solid, k_solid, 0, 0],
                polynomial_mid.coef + polynomial_left.coef,
                polynomial_mid.coef + polynomial_right.coef,
                [b_liquid, k_liquid, 0, 0]
            ])

    return vhc_spline


def construct_kappa_spline(material, direction):
    '''Makes spline fitting of either radial or axial thermal conductivity
    coefficient (kappa) for a given material.'''

    solidus = material.solidus
    liquidus = material.liquidus
    mid = 0.5 * (solidus + liquidus)
    knots = material.knots

    if direction == 'rad':
        kappa = material.kappa_rad
    elif direction == 'ax':
        kappa = material.kappa_ax

    # filter to the values in the solid state
    knots_solid = [k for k in knots if k <= solidus]
    kappa_solid = [kap for (kap,k) in zip(kappa,knots) if k <= solidus]

    # filter to the values in the liquid state
    knots_liquid = [k for k in knots if k >= liquidus]
    kappa_liquid = [kap for (kap,k) in zip(kappa,knots) if k >= liquidus]

    # apply linear fitting
    k_solid, b_solid = np.polyfit(knots_solid, kappa_solid, deg=1)
    k_liquid, b_liquid = np.polyfit(knots_liquid, kappa_liquid, deg=1)

    polynomial_mid = spl.hermine_interpolating_polynomial(
            knots=[solidus, liquidus],
            values=[k_solid * solidus + b_solid, k_liquid * liquidus + b_liquid],
            derivatives=[k_solid, k_liquid])

    # construct the final spline by specifying its knots and coefficients
    kappa_spline = spl.Spline(
            [solidus, liquidus],
            [
                [b_solid, k_solid, 0, 0],
                polynomial_mid.coef,
                [b_liquid, k_liquid, 0, 0]
            ])

    return kappa_spline
