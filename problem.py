class Problem:
    def __init__(self):
        pass

    # facade
    def solve_forward(self, control):
        return solve_forward(self.V, self.theta_init, control)

class OptimizationParameters:
    pass
