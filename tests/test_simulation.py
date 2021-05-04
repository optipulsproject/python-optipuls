import unittest

import numpy as np

import optipuls.simulation

class SimulationTestCase(unittest.TestCase):

    def test_project_Dj(self):
        rng = np.random.default_rng()
        Dj = rng.uniform(low=-1, high=1, size=100)
        control = rng.choice([0, .5, 1], size=100)

        PDj = optipuls.simulation.project_Dj(Dj, control)

        self.assertTrue(np.all(PDj[control == 0] <= 0))
        self.assertTrue(np.all(PDj[control == 1] >= 0))
