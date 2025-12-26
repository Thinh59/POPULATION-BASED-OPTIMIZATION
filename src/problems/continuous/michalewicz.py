import numpy as np

class MichalewiczFunction:
    def __init__(self, dim=2, m=10):
        self.dim = dim
        self.m = m 
        self.name = "Michalewicz Function"
        self.bounds = [(0.0, np.pi) for _ in range(dim)]

    def evaluate(self, x):
        d = len(x)
        sum_val = 0.0
        for i in range(d):
            xi = x[i]
            term = np.sin(xi) * (np.sin(((i + 1) * xi**2) / np.pi)) ** (2 * self.m)
            sum_val += term
            
        return -sum_val

    def get_bounds(self):
        return self.bounds

    def get_name(self):
        return self.name