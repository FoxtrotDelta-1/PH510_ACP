#!/usr/bin/env python3


#from mpi4py import rc
#rc.initialize = False  # do not initialize MPI automatically
#rc.finalize = False    # do not finalize MPI automatically

from mpi4py import MPI
comm = MPI.COMM_WORLD

import mpmath as mp
mp.dps = 50

nProc = comm.Get_size()
rank = comm.Get_rank()
print("i am processor", rank, "of", nProc)

nMidpoints = 100000

rankMidpoints = nMidpoints//(nProc-1)
remainderMidpoints = nMidpoints-nMidpoints%(nProc-1)

delta = 1/nMidpoints


def integrand(midpoint):
	"""
	integrand for calculating Pi.
	
	Parameters
	----------
	x : real float
	
	Returns
	-------
	numeric value for function given input x
	"""
	
	return mp.mpf(4.0)/(mp.mpf(1.0+mp.mpf(midpoint)**2)

if rank!=0
	rankCounter = (rank-1)*rankMidpoints

if rank==0
	
