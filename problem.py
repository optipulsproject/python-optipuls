import core

class Problem:
    def __init__(self):
        pass

    def solve_forward(self, control):
        return core.solve_forward(self.a, self.V, self.theta_init, control)

    def solve_adjoint(self, evo, control):
        return core.solve_adjoint(
                self.a, self.V,
                evo, control,
                self.beta_welding, self.target_point, self.threshold_temp,
                self.penalty_term_combined)

    def Dj(self, evo_adj, control):
        return core.Dj(
                self.V,
                evo_adj, control,
                self.control_ref,
                self.beta_welding,
                self.laser_pd)

    def norm2(self, vector):
        return core.norm2(self.dt, vector)

    def norm(self, vector):
        return core.norm(self.dt, vector)


class OptimizationParameters:
    pass
