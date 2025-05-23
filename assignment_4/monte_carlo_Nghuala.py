# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:55:23 2025

@author: FinnghualaD
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
        average = self.average()[0]
        variance = self.variance()
        integral = self.integral()
        standard_deviation = np.sqrt(self.variance())
        print(f"Average = {average}, Integral = {integral},",
        f"Variance = {variance}")
        print(f"Therefore, the integral is {integral:.4f} ± {standard_deviation:.4f}",
        f"units^{self.n_dimensions}")
        print()

    def parallelisation_array(self):
        """
        Equivalent parallelisation for the 'average_array' function. Again, the mean, integral,
        variances and uncertainties are left as arrays.        
        """
        avg, avg_sq = self.average_array()

        # Element-wise reduction using MPI
        val_mean = comm.reduce(avg, op=MPI.SUM, root=0)
        val_sq_mean = comm.reduce(avg_sq, op=MPI.SUM, root=0)

        if rank == 0:
            # Element-wise mean across ranks
            mean = val_mean / no_of_ranks

            # Integral term for all elements (scalar multiplier)
            integral_term = (self.b - self.a) ** self.class_used.d
            integral = mean * integral_term

            # Variance: Var = (1/n^2) * (⟨f^2⟩ - ⟨f⟩^2)
            variance = (val_sq_mean / no_of_ranks - np.square(mean)) / self.class_used.n
            # Element-wise uncertainty
            uncertainty = np.sqrt(variance) * integral_term
            return mean, integral, np.mean(uncertainty)
        return None
