import json

import numpy as np
from matplotlib import pyplot as plt

import splines as spl


x_sol = 858.
x_liq = 923.
x_mid = 0.5 * (x_sol + x_liq)

L = 397000 * 2540


with open('material.json') as file:
    material = json.load(file)

knots_sol = np.array(material['heat capacity']['knots'][:7])
c_sol = np.array(material['heat capacity']['values'][:7])
rho_sol = np.array(material['density']['values'][:7])
kappa_sol = np.array(
        material['thermal conductivity']['radial']['values'][:7])


knots_liq = np.array(material['heat capacity']['knots'][-3:])
c_liq = np.array(material['heat capacity']['values'][-3:])
rho_liq = np.array(material['density']['values'][-3:])
kappa_rad_liq = np.array(
        material['thermal conductivity']['radial']['values'][-1:])
kappa_ax_liq = np.array(
        material['thermal conductivity']['axial']['values'][-1:])


def construct_vhc_spline():
    vhc_sol = c_sol * rho_sol
    k_sol, b_sol = np.polyfit(knots_sol, vhc_sol, deg=1)

    vhc_liq = c_liq * rho_liq
    k_liq, b_liq = np.polyfit(knots_liq, vhc_liq, deg=1)
    
    pol_mid = spl.hermine_interpolating_polynomial(
            knots=[x_sol, x_liq],
            values=[k_sol * x_sol + b_sol, k_liq * x_liq + b_liq],
            derivatives=[k_sol, k_liq])
    
    I = pol_mid.integ()(x_liq) - pol_mid.integ()(x_sol)
    h = 2 * (L - I) / (x_liq - x_sol)

    pol_left = spl.hermine_interpolating_polynomial(
            knots=[x_sol, x_mid],
            values=[0, h],
            derivatives=[0, 0])
    pol_right = spl.hermine_interpolating_polynomial(
            knots=[x_mid, x_liq],
            values=[h, 0],
            derivatives=[0, 0])

    vhc_spline = spl.Spline(
            [x_sol, x_mid, x_liq],
            [
            [b_sol, k_sol, 0, 0],
            pol_mid.coef + pol_left.coef,
            pol_mid.coef + pol_right.coef,
            [b_liq, k_liq, 0, 0]
            ])

    return vhc_spline


def construct_kappa_splines():
    k_sol, b_sol = np.polyfit(knots_sol, kappa_sol, deg=1)

    pol_mid_rad = spl.hermine_interpolating_polynomial(
            knots=[x_sol, x_liq],
            values=[k_sol * x_sol + b_sol, kappa_rad_liq[0]],
            derivatives=[k_sol, 0])

    kappa_rad_spline = spl.Spline(
            [x_sol, x_liq],
            [
            [b_sol, k_sol, 0, 0],
            pol_mid_rad.coef,
            [kappa_rad_liq[0], 0, 0, 0]
            ])

    pol_mid_ax = spl.hermine_interpolating_polynomial(
            knots=[x_sol, x_liq],
            values=[k_sol * x_sol + b_sol, kappa_ax_liq[0]],
            derivatives=[k_sol, 0])

    kappa_ax_spline = spl.Spline(
            [x_sol, x_liq],
            [
            [b_sol, k_sol, 0, 0],
            pol_mid_ax.coef,
            [kappa_ax_liq[0], 0, 0, 0]
            ])

    return kappa_rad_spline, kappa_ax_spline


vhc = construct_vhc_spline()
kappa_rad, kappa_ax = construct_kappa_splines()