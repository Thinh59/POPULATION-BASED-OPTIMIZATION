import numpy as np

class BraninFunction:
    def __init__(self, dim=2):
        self.dim = dim
        self.name = "Branin Function"
        self.bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def evaluate(self, x):
        x1 = x[0]
        x2 = x[1]

        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * np.cos(x1)
        term3 = s

        return term1 + term2 + term3

    def get_bounds(self):
        return self.bounds

    def get_name(self):
        return self.name