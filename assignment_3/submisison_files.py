#!/usr/bin/python3

"""
@author: FinnghualaD
"""

import time
import numpy as np
from mpi4py import MPI

from monte_carlo import MonteCarlo
from n_sphere import n_sphere
from normal_distribution import normal_distribution



comm = MPI.COMM_WORLD
no_of_ranks = comm.Get_size()
rank = comm.Get_rank()



if rank==0:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"{no_of_ranks} Processors:")
    print()
    start_time = time.time()


no_of_samples = np.int32(100000000/no_of_ranks)

if rank==0:
    print("For 2D n_sphere:")
n_sphere_2D_MONTE = MonteCarlo(n_sphere(2, no_of_samples), -1, 1, 2, no_of_samples)
n_sphere_2D_RESULT = n_sphere_2D_MONTE.parallelisation()

if rank==0:
    print("For 3D n_sphere:")
n_sphere_3D_MONTE = MonteCarlo(n_sphere(3, no_of_samples), -1, 1, 3, no_of_samples)
n_sphere_3D_RESULT = n_sphere_3D_MONTE.parallelisation()

if rank==0:
    print("For 4D n_sphere:")
n_sphere_4D_MONTE = MonteCarlo(n_sphere(4, no_of_samples), -1, 1, 4, no_of_samples)
n_sphere_4D_RESULT = n_sphere_4D_MONTE.parallelisation()

if rank==0:
    print("For 5D n_sphere:")
n_sphere_5D_MONTE = MonteCarlo(n_sphere(5, no_of_samples), -1, 1, 5, no_of_samples)
n_sphere_5D_RESULT = n_sphere_5D_MONTE.parallelisation()

if rank==0:
    print("For 1D normal distribution:")
MonteCarlo.parallelisation(MonteCarlo(normal_distribution(1,no_of_samples, 1, 0), -1, 1, 1, no_of_samples))

if rank==0:
    print("For 6D normal distribution:")
MonteCarlo.parallelisation(MonteCarlo(normal_distribution(6,no_of_samples, 1, 0),
 -1, 1, 6, 100_000))


if rank==0:
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The code took {execution_time} seconds to run for {no_of_ranks} processors")
    print()