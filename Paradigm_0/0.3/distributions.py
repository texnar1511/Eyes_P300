import numpy as np
from scipy.stats import rv_continuous


class Absolute(rv_continuous):
    
    def _pdf(self, x, x0):
        
        return np.max([np.abs(self.a - x0), np.abs(self.b - x0)]) - np.abs(x - x0)