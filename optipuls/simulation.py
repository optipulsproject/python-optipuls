from .core import project_Dj
from numpy import sqrt


class Simulation():
    '''Caches the computation results and keeps related values together.'''

    def __init__(self, problem, control):
        self._problem = problem
        self._control = control

    @property
    def problem(self):
        return self._problem

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, control_):
        raise AttributeError(
            'Assigning a new control in an existing simulation object is '
            'forbidden for safety reasons. Create a new object instead.')

    @property
    def evo(self):
        try:
            return self._evo
        except AttributeError:
            self._evo = self.problem.solve_forward(self.control)
            return self._evo

    @property
    def evo_adj(self):
        try:
            return self._evo_adj
        except AttributeError:
            self._evo_adj = self.problem.solve_adjoint(
                    self.evo, self.control, self.ps_magnitude)
            return self._evo_adj

    @property
    def evo_vel(self):
        try:
            return self._evo_vel
        except AttributeError:
            self._evo_vel = self.problem.compute_evo_vel(
                    self.evo, velocity_max=0.)
            return self._evo_vel

    @property
    def Dj(self):
        try:
            return self._Dj
        except AttributeError:
            self._Dj = self.problem.Dj(self.evo_adj, self.control)
            return self._Dj

    @property
    def Dj_norm(self):
        try:
            return self._Dj_norm
        except AttributeError:
            self._Dj_norm = sqrt(self.Dj_norm2)
            return self._Dj_norm

    @property
    def Dj_norm2(self):
        try:
            return self._Dj_norm2
        except AttributeError:
            self._Dj_norm2 = self.problem.norm2(self.Dj)
            return self._Dj_norm2

    @property
    def PDj(self):
        try:
            return self._PDj
        except AttributeError:
            self._PDj = project_Dj(self.Dj, self.control)
            return self._PDj

    @property
    def PDj_norm(self):
        try:
            return self._PDj_norm
        except AttributeError:
            self._PDj_norm = sqrt(self.PDj_norm2)
            return self._PDj_norm

    @property
    def PDj_norm2(self):
        try:
            return self._PDj_norm2
        except AttributeError:
            self._PDj_norm2 = self.problem.norm2(self.PDj)
            return self._PDj_norm2

    @property
    def penalty_velocity_vector(self):
        try:
            return self._penalty_velocity_vector
        except AttributeError:
            self._penalty_velocity_vector = \
                self.problem.time_domain.dt * self.problem.beta_velocity * \
                self.problem.vectorize_penalty_term(
                    self.evo,
                    lambda k, theta_k, theta_kp1: self.problem.integral2(
                            self.problem.velocity(theta_k, theta_kp1)))
            return self._penalty_velocity_vector

    @property
    def penalty_velocity_total(self):
        try:
            return self._penalty_velocity_total
        except AttributeError:
            self._penalty_velocity_total =\
                    sum(self.penalty_velocity_vector)
            return self._penalty_velocity_total

    @property
    def penalty_liquidity_vector(self):
        try:
            return self._penalty_liquidity_vector
        except AttributeError:
            self._penalty_liquidity_vector = \
                self.problem.time_domain.dt * self.problem.beta_liquidity * \
                self.problem.vectorize_penalty_term(
                    self.evo,
                    lambda k, theta_k, theta_kp1: self.problem.integral2(
                            self.problem.liquidity(theta_k, theta_kp1)))
            return self._penalty_liquidity_vector

    @property
    def penalty_liquidity_total(self):
        try:
            return self._penalty_liquidity_total
        except AttributeError:
            self._penalty_liquidity_total =\
                    self.penalty_liquidity_vector[-1]
                    # sum(self.penalty_liquidity_vector)
            return self._penalty_liquidity_total

    @property
    def penalty_welding_total(self):
        try:
            return self._penalty_welding_total
        except AttributeError:
            self._penalty_welding_total = self.problem.penalty_welding(
                    self.evo, self.control)
            return self._penalty_welding_total

    @property
    def penalty_control_total(self):
        beta_control = self.problem.beta_control
        norm2 = self.problem.norm2
        control = self.control
        control_ref = self.problem.control_ref

        return .5 * beta_control * norm2(control - control_ref)

    @property
    def temp_target_point_vector(self):
        try:
            return self._temp_target_point_vector
        except AttributeError:
            self._temp_target_point_vector = \
                self.problem.temp_target_point_vector(self.evo)
            return self._temp_target_point_vector

    def temp_at_point_vector(self, point):
        '''Provides the temperature evolution at a given point.'''

        return self.problem.temp_at_point_vector(self.evo, point)


    @property
    def energy_total(self):
        '''Total used energy [J].'''
        P_YAG = self.problem.P_YAG
        dt = self.problem.time_domain.dt
        control = self.control

        return P_YAG * dt * sum(control)

    @property
    def J(self):
        try:
            return self._J_total
        except AttributeError:
            self._J_total = self.problem.cost_total(self.evo, self.control)
            return self._J_total

    @property
    def ps_magnitude(self):
        try:
            return self._ps_magnitude
        except AttributeError:
            self._ps_magnitude = self.problem.compute_ps_magnitude(self.evo)
            return self._ps_magnitude

    @property
    def report(self):
        return f'''simulation report
=================

penalty_control_total:      {self.penalty_control_total:.7e}
penalty_velocity_total:     {self.penalty_velocity_total:.7e}
penalty_liquidity_total:    {self.penalty_liquidity_total:.7e}
penalty_welding_total:      {self.penalty_welding_total:.7e}
-----------------------------------------
cost_total:                 {self.J:.7e}

energy_total:               {self.energy_total:9.6} [J]
time_total:                 {self.problem.time_domain.T:9.6} [s]
temp_target_point_max:      {self.temp_target_point_vector.max():9.6} [K]
'''


    def spawn(self, control):
        '''Spawns a new simulation instance based for the same problem.'''
        return Simulation(self.problem, control)

    @property
    def welding_depth_vector(self):
        try:
            return self._welding_depth_vector
        except AttributeError:
            self._welding_depth_vector = \
                self.problem.compute_welding_depth(self.evo)
            return self._welding_depth_vector

    @property
    def welding_radius_vector(self):
        try:
            return self._welding_radius_vector
        except AttributeError:
            self._welding_radius_vector = \
                self.problem.compute_welding_radius(self.evo)
            return self._welding_radius_vector
