
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:25:40 2025

@author: finnd
"""


import numpy as np

from monte_carlo import MonteCarlo

from n_sphere import n_sphere

from normal_distribution import normal_distribution

# def n_sphere(n_dimensions, n_samples):
#     """
#     calculates the number get the gist down first
#     """

#     magnitude = 0
#     included = np.zeros(n_samples)

#     points_array = np.zeros((n_samples,n_dimensions))

#     for i in range(n_samples):
#         for j in range(n_dimensions):
#             points_array[i,j] = np.random.uniform(-1,1,1)

#     for i in range(n_samples):
#         for j in range(n_dimensions):
#             magnitude = magnitude + (points_array[i,j])**2
#         if magnitude <= 1:
#             included[i] = 1
#         magnitude = 0

#     return included, points_array


trial_1 = n_sphere(2, 1000)[0]

trial_1_monte = MonteCarlo(n_sphere(2,1_000_000)[0], -1, 1, 2, 1_000_000)

print(trial_1_monte.calculations())

print(MonteCarlo.calculations(MonteCarlo(n_sphere(2,1_000_000)[0], -1, 1, 2, 1_000_000)))





# def normal_distribution(n_dimensions, n_samples, sigma, mean):
#     """
#     evaluates the integral of a n-dimensional normal distribution by 
#     transformation by defining x as x = t/(1-t**2). this results in an inge
    
#     sigma can be a sigular value or an np.array of values of length n_dimensions
    
#     mean can be a singular value or an np.array of values of length n_dimensions
#     """

#     t_points_array = np.zeros((n_samples,n_dimensions))

#     for i in range(n_samples):
#         for j in range(n_dimensions):
#             t_points_array[i,j] = np.random.uniform(-1,1,1)

#     # x_variable = (t_points_array/(1-t_points_array**2))


#     exponent = np.zeros((n_samples,n_dimensions))
#     normalisaiton = np.zeros((n_samples,n_dimensions))
#     t_correction = np.zeros((n_samples,n_dimensions))

#     exponent = -((np.sum(((t_points_array/(1-t_points_array**2)-mean)**2),axis=1))/(2*sigma**2))
#     normalisaiton = 1/(sigma*np.sqrt(2*np.pi))
#     t_correction = np.prod(((1+t_points_array**2)/(1-t_points_array**2)**2), axis=1)
#     output = np.exp(exponent)*normalisaiton*t_correction

#     return output

trial_2 = normal_distribution(1, 1000000, 1, 0)


trial_2_output = trial_2[0]


trial_2_monte = MonteCarlo(trial_2, -1, 1, 1, 1000000)

print(trial_2_monte.calculations())

print(MonteCarlo.calculations(MonteCarlo(normal_distribution(1,1000000, 1, 0), -1, 1, 1, 1000000)))
