import dolfin
import numpy as np

from . import core
from .uflspline import UFLSpline

class IncompleteProblemException(Exception):
    pass

class Problem:
    def __init__(self):
        pass

    def norm2(self, vector):
        return core.norm2(self.time_domain.dt, vector)

    def norm(self, vector):
        return core.norm(self.time_domain.dt, vector)

    def solve_forward(self, control):
        return core.solve_forward(self.a, self.V, self.theta_init, control)

    def solve_adjoint(self, evo, control, ps_magnitude):
        return core.solve_adjoint(
                evo, control, ps_magnitude,
                a=self.a,
                V=self.V,
                j=self.j,
                target_point=self.target_point)

    def a(self, u_k, u_kp1, v, control_k):
        return core.a(
                u_k, u_kp1, v, control_k,
                self.vhc, self.kappa, self.cooling_bc, self.laser_bc,
                self.time_domain.dt, self.implicitness,
                self.space_domain.x, self.space_domain.ds)

    def laser_bc(self, control_k):
        return core.laser_bc(control_k, self.laser_pd)

    def cooling_bc(self, theta):
        return core.cooling_bc(
            theta,
            self.temp_amb, self.convection_coeff, self.radiation_coeff)

    def Dj(self, evo_adj, control):
        return core.Dj(
                evo_adj, control,
                self.V,
                self.control_ref,
                self.beta_control,
                self.beta_welding,
                self.laser_pd,
                self.space_domain.x,
                self.space_domain.ds)

    def compute_evo_vel(self, evo, velocity_max):
        return core.compute_evo_vel(
                evo,
                self.V, self.V1, self.time_domain.dt,
                self.material.liquidus, self.material.solidus, velocity_max)

    def compute_ps_magnitude(self, evo):
        return core.compute_ps_magnitude(
                evo,
                V=self.V,
                target_point=self.target_point,
                threshold_temp=self.threshold_temp,
                beta_welding=self.beta_welding,
                pow_=self.pow_)

    def vectorize_penalty_term(self, evo, penalty_term, *args, **kwargs):
        return core.vectorize_penalty_term(
                evo, self.V, penalty_term, *args, **kwargs)

    def velocity(self, theta_k, theta_kp1):
        return core.velocity(
                theta_k, theta_kp1,
                dt=self.time_domain.dt,
                liquidus=self.material.liquidus,
                solidus=self.material.solidus,
                velocity_max=self.velocity_max)

    def liquidity(self, theta_k, theta_kp1):
        return core.liquidity(
                theta_k, theta_kp1,
                solidus=self.material.solidus)

    def penalty_welding(self, evo, control):
        return core.penalty_welding(
                evo, control,
                V=self.V,
                beta_welding=self.beta_welding,
                target_point=self.target_point,
                threshold_temp=self.threshold_temp,\
                pow_=self.pow_)

    def j(self, k, theta_k, theta_kp1):
        form = self.time_domain.dt * self.beta_velocity\
             * self.integral2(self.velocity(theta_k, theta_kp1))

        if k == self.time_domain.Nt - 1:
            form += self.time_domain.dt * self.beta_liquidity\
                  * self.integral2(self.liquidity(theta_k, theta_kp1))

        return form

    def cost_total(self, evo, control):
        j_vector = self.vectorize_penalty_term(
                evo=evo,
                penalty_term=self.j)

        cost = sum(j_vector)\
             + .5 * self.beta_control * self.norm2(control - self.control_ref)\
             + self.penalty_welding(evo, control)

        return cost

    def integral2(self, form):
        x = self.space_domain.x
        return core.integral2(form, x)

    @property
    def control_ref(self):
        '''Initialize control_ref with zeros automatically.

        Since a non trivial control_ref might be used only for debugging
        purposes, there is no user-level setter for it. Wisely use _control_ref
        to assign a non-trivial value.

        '''
        try:
            return self._control_ref
        except AttributeError:
            self._control_ref = np.zeros(self.time_domain.Nt)
            return self._control_ref

    @property
    def vhc(self):
        try:
            return self._vhc
        except AttributeError:
            self._vhc = UFLSpline(self.material.vhc, self.V.ufl_element())
            return self._vhc

    @property
    def kappa(self):
        try:
            return self._kappa
        except AttributeError:
            kappa_rad_uflspline = UFLSpline(
                self.material.kappa[0], self.V.ufl_element()
                )
            kappa_ax_uflspline = UFLSpline(
                self.material.kappa[1], self.V.ufl_element()
                )
            self._kappa = lambda theta: dolfin.as_matrix(
                    [[kappa_rad_uflspline(theta), dolfin.Constant(0)],
                     [dolfin.Constant(0), kappa_ax_uflspline(theta)]])

            return self._kappa

    def compute_welding_depth(self, evo):
        return core.compute_welding_size(
                    evo,
                    self.V,
                    self.material.liquidus,
                    self.space_domain.x,
                    self.space_domain.ds(0),
        )

    def compute_welding_radius(self, evo):
        return core.compute_welding_size(
                    evo,
                    self.V,
                    self.material.liquidus,
                    self.space_domain.x,
                    self.space_domain.ds(1) + self.space_domain.ds(2),
        )
