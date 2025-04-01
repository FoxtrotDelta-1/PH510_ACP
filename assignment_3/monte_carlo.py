# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:55:23 2025

@author: Finnghuala
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
no_of_ranks = comm.Get_size()
rank = comm.Get_rank()


class MonteCarlo:
    """
    monte_carlo class that can operate on functions dependent on a given number
    of dimensions.
    
    upon initiation
    """

    def __init__(self, function, lower_bound, upper_bound, n_dimensions, n_samples):
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

    def parallelisation(self):
        """
        fill out
        """
        val_mean = comm.reduce(self.average()[0], op=MPI.SUM, root=0)
        val_square_mean = comm.reduce(self.average()[1], op=MPI.SUM, root=0)

        if rank==0:
            integral_term = (self.upper_bound - self.lower_bound)**self.n_dimensions
            integral = (val_mean)*integral_term/no_of_ranks

            variance_1 = 1/no_of_ranks * val_square_mean
            variance_2 = (1/no_of_ranks * val_mean)**2

            variance = 1/self.n_dimensions * (variance_1 - variance_2)
            uncertainty = np.sqrt(variance) * integral_term
            print(f"Average = {self.average()[0]}, Integral = {integral},",
            f"Variance = {variance}")
            print(f"Therefore, the integral is {integral:.4f} ± {uncertainty:.4f}",
            f"units^{self.n_dimensions}")
            print()
            return integral, uncertainty
        return None
    