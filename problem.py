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


class OptimizationParameters:
    pass
