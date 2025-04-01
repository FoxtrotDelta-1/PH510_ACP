
# -*- coding: utf-8 -*-
"""
@author: FinnghualaD
"""


import numpy as np

from monte_carlo import MonteCarlo

from n_sphere import n_sphere
from normal_distribution import normal_distribution

print("For 2D n_sphere:")
n_sphere_2D_MONTE = MonteCarlo(n_sphere(2, 100_000), -1, 1, 2, 100_000)
n_sphere_2D_RESULT = n_sphere_2D_MONTE.calculations()

print("For 3D n_sphere:")
n_sphere_3D_MONTE = MonteCarlo(n_sphere(3, 100_000), -1, 1, 3, 100_000)
n_sphere_3D_RESULT = n_sphere_3D_MONTE.calculations()

print("For 4D n_sphere:")
n_sphere_4D_MONTE = MonteCarlo(n_sphere(4, 100_000), -1, 1, 4, 100_000)
n_sphere_4D_RESULT = n_sphere_4D_MONTE.calculations()

print("For 5D n_sphere:")
n_sphere_5D_MONTE = MonteCarlo(n_sphere(5, 100_000), -1, 1, 5, 100_000)
n_sphere_5D_RESULT = n_sphere_5D_MONTE.calculations()

print("For 1D normal distribution:")
MonteCarlo.calculations(MonteCarlo(normal_distribution(1,100_000, 1, 0), -1, 1, 1, 100_000))

print("For 6D normal distribution:")
MonteCarlo.calculations(MonteCarlo(normal_distribution(6,100_000, 1, 0), -1, 1, 6, 100_000))
