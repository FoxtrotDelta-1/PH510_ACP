# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:23:12 2025

@author: FinnghualaD
"""

import numpy as np

def n_sphere(n_dimensions, n_samples):
    """
    calculates the number get the gist down first
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