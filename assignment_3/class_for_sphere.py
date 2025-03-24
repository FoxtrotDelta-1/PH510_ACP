#!/usr/bin/env python3
"""

@author: FinnghualaD
"""

import random as rand

import numpy as np

class sample_generation:
    def __init__(self, n_dimensions, n_samples):
        """
        Generates an array of n_dimensions by n_samples, each element
        is a pseudo random float within interval [-1,1].
        """

        self.n_dimensions = n_dimensions
        self.n_samples = n_samples
        self.points_array = np.zeros((n_samples,n_dimensions))

        i = 0
        j = 0

        while i < self.n_samples:
            j=0
            while j < self.n_dimensions:
                self.points_array[i,j] = np.random.uniform(-1,1,1)
                j+=1
            i+=1

    def unit_balls(self):
        i = 0
        j = 0
        magnitude = 0
        included = 0
        percentage_included = 0

        while i < self.n_samples:
            j = 0
            while j < self.n_dimensions:
                magnitude = magnitude + (self.points_array[i,j])**2
                j+=1
            magnitude = np.sqrt(magnitude)
            if magnitude <= 1:
                included = included + 1
            magnitude = 0
            i+=1

        percentage_included = included/self.n_samples * 100

        return f'{included},{percentage_included}'


samples_1 = sample_generation(2, 1000000)

print(sample_generation.unit_balls(samples_1))