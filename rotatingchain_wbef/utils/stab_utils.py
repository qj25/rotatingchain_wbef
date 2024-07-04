import numpy as np
from scipy.optimize import newton
from rotatingchain_wbef.algos.shootingc_cpp.Shootc import Shootc

k_spring = 8490790.955648089
l = 0.74
mu = 0.02801961015363869
g = 9.8
n_steps = 11
sr_latter = 1.0 # split ratio latter

def zlr_test(p0,p1):
    if p0[0] * p1[0] < 0:
        return np.interp(
            0.0,
            [p0[0], p1[0]],
            [p0[1], p1[1]]
        )
    return None