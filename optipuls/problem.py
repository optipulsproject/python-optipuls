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
                self.liquidus, self.solidus, velocity_max)

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
                liquidus=self.liquidus,
                solidus=self.solidus,
                velocity_max=self.velocity_max)

    def liquidity(self, theta_k, theta_kp1):
        return core.liquidity(
                theta_k, theta_kp1,
                solidus=self.solidus)

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

    def temp_target_point_vector(self, evo):
        '''Provides the temperature evolution at the target_point.'''

        return core.temp_at_point_vector(
                evo,
                V=self.V,
                point=self.target_point)

    def temp_at_point_vector(self, evo, point):
        '''Provides the temperature evolution at a given point.'''

        return core.temp_at_point_vector(
                evo,
                V=self.V,
                point=point)

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
            raise IncompleteProblemException(
                'vhc spline must be assigned to complete problem formulation')

    @vhc.setter
    def vhc(self, spline):
        self._vhc = UFLSpline(spline, self.V.ufl_element())


    @property
    def kappa(self):
        try:
            return lambda theta: dolfin.as_matrix(
                    [[self._kappa_rad(theta), dolfin.Constant(0)],
                     [dolfin.Constant(0), self._kappa_ax(theta)]])
        except AttributeError:
            raise IncompleteProblemException(
                'kappa spline must be assigned to complete problem formulation')

    @kappa.setter
    def kappa(self, splines):
        try:
            spline_rad, spline_ax = splines
        except:
            raise ValueError('Two splines must be provided to construct kappa')

        self._kappa_rad = UFLSpline(spline_rad, self.V.ufl_element())
        self._kappa_ax = UFLSpline(spline_ax, self.V.ufl_element())
