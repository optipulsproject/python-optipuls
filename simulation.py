import core


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
            self._evo = core.solve_forward(
                    self.problem.V, self.problem.theta_init, self.control)
            return self._evo

    @property
    def evo_adj(self):
        try:
            return self._evo_adj
        except AttributeError:
            self._evo_adj = core.solve_adjoint(
                    self.problem.V, self.evo, self.control, self.problem.opts)
            return self._evo_adj

    @property
    def evo_vel(self):
        try:
            return self._evo_vel
        except AttributeError:
            self._evo_vel = core.compute_evo_vel(
                    self.problem.V, self.problem.V1, self.evo)
            return self._evo_vel

    @property
    def Dj(self):
        try:
            return self._Dj
        except AttributeError:
            self._Dj = core.Dj(self.problem.V, self.evo_adj, self.control)
            return self._Dj

    @property
    def Dj_norm(self):
        try:
            return self._Dj_norm
        except AttributeError:
            self._Dj_norm = core.norm(self.Dj)
            return self._Dj_norm

    @property
    def penalty_velocity_vector(self):
        try:
            return self._penalty_velocity_vector
        except AttributeError:
            self._penalty_velocity_vector = core.vectorize_penalty_term(
                    self.problem.V, self.evo, core.penalty_term_velocity)
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
            self._penalty_liquidity_vector = core.vectorize_penalty_term(
                    self.problem.V, self.evo, core.penalty_term_liquidity)
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
            self._penalty_welding_total = core.penalty_welding(
                    self.problem.V, self.evo, self.control)
            return self._penalty_welding_total

    @property
    def penalty_control_total(self):
        return .5 * core.beta_control * core.norm2(self.control - core.control_ref)

    @property
    def temp_target_point_vector(self):
        try:
            return self._temp_target_point_vector
        except AttributeError:
            self._temp_target_point_vector = \
                    core.temp_at_point_vector(
                        self.problem. V, self.evo, self.problem.opts.target_point)
            return self._temp_target_point_vector

    @property
    def temp_central_point_vector(self):
        try:
            return self._temp_central_point_vector
        except AttributeError:
            self._temp_central_point_vector = \
                    core.temp_at_point_vector(
                        self.problem. V, self.evo, central_point)
            return self._temp_central_point_vector

    @property
    def energy_total(self):
        '''Total used energy [J].'''
        return core.P_YAG * core.dt * sum(self.control)

    @property
    def J(self):
        try:
            return self._J_total
        except AttributeError:
            self._J_total = core.cost_total(self.problem.V, self.evo, self.control)
            return self._J_total

    def report(self):
        print(f'''
simulation report
=================

penalty_control_total:      {self.penalty_control_total:.7e}
penalty_velocity_total:     {self.penalty_velocity_total:.7e}
penalty_liquidity_total:    {self.penalty_liquidity_total:.7e}
penalty_welding_total:      {self.penalty_welding_total:.7e}
-----------------------------------------
cost_total:                 {self.J:.7e}

energy_total:               {self.energy_total:9.6} [J]
time_total:                 {core.T:9.6} [s]
temp_target_point_max:      {self.temp_target_point_vector.max():9.6} [K]
''')