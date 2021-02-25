import numpy as np

class TimeDomain:
    def __init__(self, T, Nt):
        self._T = T
        self._Nt = Nt
        self._dt = T/Nt
        self._timeline = np.linspace(0, T, num=Nt, endpoint=False)

    # protect attributes from being modified by turning them into properties

    @property
    def T(self):
        return self._T

    @property
    def Nt(self):
        return self._Nt

    @property
    def dt(self):
        return self._dt

    @property
    def timeline(self):
        return self._timeline
