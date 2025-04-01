
# -*- coding: utf-8 -*-
"""
@author: FinnghualaD
"""


import numpy as np

from monte_carlo import MonteCarlo

from n_sphere import n_sphere
from normal_distribution import normal_distribution

trial_1 = n_sphere(2, 1000)[0]

trial_1_monte = MonteCarlo(n_sphere(2,1_000_000)[0], -1, 1, 2, 1_000_000)

print(trial_1_monte.calculations())

print(MonteCarlo.calculations(MonteCarlo(n_sphere(2,1_000_000)[0], -1, 1, 2, 1_000_000)))

trial_2 = normal_distribution(1, 1000000, 1, 0)


trial_2_output = trial_2[0]


trial_2_monte = MonteCarlo(trial_2, -1, 1, 1, 1000000)

print(trial_2_monte.calculations())

print(MonteCarlo.calculations(MonteCarlo(normal_distribution(1,1000000, 1, 0), -1, 1, 1, 1000000)))
