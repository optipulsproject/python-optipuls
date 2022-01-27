from functools import cached_property
from importlib.resources import is_resource, read_text
import json
import numpy as np

from . import materials
from . import splines as spl


class Material:
    def __init__(self, material_dict):
        # load public attributes
        for attr in [
                "label",
                "description",
                "solidus",
                "liquidus",
                "melting_enthalpy",
            ]: setattr(self, attr, material_dict[attr])

        # load private attributes
        for attr in [
                "knots",
                "heat_capacity",
                "density",
                "kappa_rad",
                "kappa_ax",
            ]: setattr(self, f'_{attr}', material_dict[attr])

    @classmethod
    def load(cls, resource_name):
        try:
            material_json = read_text(materials, resource_name)
        except FileNotFoundError:
            with open(resource_name) as file:
                material_json = file.readlines()

        material = cls(json.loads(material_json))
        return material

    @property
    def vhc(self):
        try:
            return self._vhc
        except AttributeError:
            self._vhc, self._vhc_bridge = gen_vhc_spline(
                self.melting_enthalpy,
                self.solidus,
                self.liquidus,
                self._knots,
                self._heat_capacity,
                self._density,
                )
            return self._vhc

    @property
    def kappa(self):
        try:
            return self._kappa
        except AttributeError:
            self._kappa = (
                gen_kappa_spline(
                    self.solidus,
                    self.liquidus,
                    self._knots,
                    self._kappa_rad,
                    ),
                gen_kappa_spline(
                    self.solidus,
                    self.liquidus,
                    self._knots,
                    self._kappa_ax,
                    ),
                )

            return self._kappa


def gen_vhc_spline(
        melting_enthalpy,
        solidus,
        liquidus,
        knots,
        heat_capacity,
        density,
    ):
    '''Makes spline fitting of the volumetric heat capacity coefficient for a
    given material.'''


    mid = 0.5 * (solidus + liquidus)

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

    return vhc_spline, polynomial_mid

def gen_kappa_spline(
        solidus,
        liquidus,
        knots,
        kappa,
    ):
    '''Makes spline fitting of either radial or axial thermal conductivity
    coefficient (kappa) for a given material.'''

    mid = 0.5 * (solidus + liquidus)

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
