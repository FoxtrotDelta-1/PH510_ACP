#!/usr/bin/env python3
"""

@author: FinnghualaD
"""

import numpy as np
import matplotlib.pyplot as plt

class SampleGeneration:
    """
    o7irotp9y
    """
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
        """
        calculates the % of sample points with in a unit_ball.
        """
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
            if magnitude <= 1:
                included = included + 1
            magnitude = 0
            i+=1

        percentage_included = included/self.n_samples * 100

        return f'{included},{percentage_included}'


samples_1 = SampleGeneration(2, 100_000)

print(SampleGeneration.unit_balls(samples_1))

class MonteCarlo:
    """
    monte_carlo class that can operate on functions dependent on a given number
    of dimensions.
    
    upon initiation
    """

    def __init__(self, function, lower_bound, upper_bound,n_dimensions, n_samples):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_samples = n_samples
        self.n_dimensions = n_dimensions
        self.value = function


    def average(self):
        """
        Calculates the average value of a given function.
        """
        value_2 = self.value**2
        average = 1/self.n_samples * np.sum(self.value)
        average_2 = 1/self.n_samples * np.sum(value_2)
        return average, average_2

    def integral(self):
        """
        Calculates the integral of a given function  between limits 'a' and 'b'.  
        """
        integral = ((self.upper_bound - self.lower_bound)**self.n_dimensions) * self.average()[0]
        return integral

    def variance(self):
        """
        Calculates the variance (error) of a given function.
        """
        variance = 1/self.n_samples * (self.average()[1] - self.average()[0]**2)
        return variance

    def calculations(self):
        """
        Returns all three desired values at once.
        """

        return self.average()[0], self.variance(), self.integral(), np.sqrt(self.variance())

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


# trial_1 = n_sphere(2, 1000)[0]

# trial_1_monte = MonteCarlo(n_sphere(2,1_000_000), 0, 1, 1_000_000)

# print(trial_1_monte.calculations())

# print(MonteCarlo.calculations(MonteCarlo(n_sphere(2,1_000_000), -1, 1, 2, 1_000_000)))


def NormalDistribution(n_dimensions, n_samples):
    """
    """

    magnitude = 0

    output = np.zeros(n_samples)
    Norm_1 = np.zeros(n_samples)
    Norm_2 = np.zeros(n_samples)
    Norm_3 = np.zeros(n_samples)
    
    t_points_array = np.zeros((n_samples,n_dimensions))
    t_magnitude = np.zeros((n_samples))

    for i in range(n_samples):
        for j in range(n_dimensions):
            t_points_array[i,j] = np.random.uniform(-1,1,1)

    sigma = 1
    mean = 0


    Norm_1 = -((np.sum(((t_points_array/(1-t_points_array**2)-mean)**2),axis=1))/(2*sigma**2)) # exponetial
    Norm_2 = 1/(sigma*np.sqrt(2*np.pi)) # normalisation
    Norm_3 = np.product(((1+t_points_array**2)/(1-t_points_array**2)**2), axis=1) # t distribution thingy
    output = np.exp(Norm_1)*Norm_2*Norm_3
    
    return output # , Norm_1, Norm_2, Norm_3, t_points_array ## renable for testing

### test chunk
# trial_2 = NormalDistribution(6,10_000_000)
# trial_2_output = trial_2[0]
# trial_2_Norm_1 = trial_2[1]
# trial_2_Norm_2 = trial_2[2]
# trial_2_Norm_3 = trial_2[3]
# trial_2_monte = MonteCarlo(trial_2[0], -1, 1, 6, 10_000_000)
# print(trial_2_monte.calculations())
### end of test checks

print(MonteCarlo.calculations(MonteCarlo(NormalDistribution(6,1_000_000)[0], -1, 1, 6, 1_000_000)))