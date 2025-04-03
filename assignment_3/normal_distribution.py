# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:26:33 2025

@author: FinnghualaD
"""

import numpy as np

def normal_distribution(n_dimensions, n_samples, sigma, mean):
    """
    evaluates the integral of a n-dimensional normal distribution by 
    transformation by defining x as x = t/(1-t**2). this results in an inge
    
    sigma can be a sigular value or an np.array of values of length n_dimensions
    
    mean can be a singular value or an np.array of values of length n_dimensions
    """

    t_points_array = np.zeros((n_samples,n_dimensions))

    for i in range(n_samples):
        for j in range(n_dimensions):
            t_points_array[i,j] = np.random.uniform(-1,1,1)

    # x_variable = (t_points_array/(1-t_points_array**2))


    exponent = np.zeros((n_samples,n_dimensions))
    normalisaiton = np.zeros((n_samples,n_dimensions))
    t_correction = np.zeros((n_samples,n_dimensions))

    exponent = -((np.sum(((t_points_array/(1-t_points_array**2)-mean)**2),axis=1))/(2*sigma**2))
    normalisaiton = 1/(sigma*np.sqrt(2*np.pi))
    t_correction = np.prod(((1+t_points_array**2)/(1-t_points_array**2)**2), axis=1)
    output = np.exp(exponent)*normalisaiton*t_correction

    return output