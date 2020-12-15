import core

class Problem:
    def __init__(self):
        self.a = core.a  # dummy

    def solve_forward(self, control):
        return core.solve_forward(self.a, self.V, self.theta_init, control)

class OptimizationParameters:
    pass
