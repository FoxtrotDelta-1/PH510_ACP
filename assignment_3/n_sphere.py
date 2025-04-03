# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:23:12 2025

@author: FinnghualaD
"""

"""
this file contains the function "n_sphere" this function computes the containted region/"area/volume" that an n dimensional ball takes up within a 
n dimensional cube.
"""
import numpy as np

def n_sphere(n_dimensions, n_samples):
    """
    calculates the containted region/"area/volume" of an n-ball within an n-cube

    inputs: n_dimensions is the number of dimensions for the n_ball
            n_samples is the number of samples to be used in the calculation

    an array n_samples by n_dimensions is created and populated with random numbers
    the magnitude of each sample (row) is calculated by summing the squares of each 
    entry in the row (n entries = n_dimensions).

    if the sum of the squares is less than or equal to unity then the included array
    enetry for that sample is flipped from a 0 to a 1.

    included is an array of dimension n_samples.

    returns: included
    """

    magnitude = 0
    included = np.zeros(n_samples)

    points_array = np.zeros((n_samples,n_dimensions))

    for i in range(n_samples):
        for j in range(n_dimensions):
            points_array[i,j] = np.random.uniform(-1,1,1)

    for i in range(n_samples):
        for j in range(n_dimensions):
            magnitude = magnitude + (points_array[i,j])**2
        if magnitude <= 1:
            included[i] = 1
        magnitude = 0

    return included
