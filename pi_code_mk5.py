#!/usr/bin/env python3
"""
Created by Finn Docherty for submission to PH510 for assigment 

uses the midpoint method to evalutate pi

"""
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
print(nproc)
# The first processor is leader, so one fewer available to be a worker
nworkers = nproc - 1
# samples
N_MIDPOINTS = 10000000
DELTA = 1.0 / N_MIDPOINTS
MIDPOINTS_RANK = N_MIDPOINTS//(nworkers)
MIDPOINTS_REMAINDER = N_MIDPOINTS%(nworkers)



# integral
i = 0
I_TOTAL = 0.0
I_PARTIAL = 0.0
SOURCE_RANK = 0
def integrand(inte_input):
    """
    Parameters
    ----------
    inte_input : floating point
        the mid point of the section being calculated

    Returns
    -------
    floating point
        returns floating point which is the hight of the function at the 
        mid point needs to be multiplied by delta to give the area.
    """
    return 4.0 / (1.0 + inte_input * inte_input)

if comm.Get_rank() == 0:
	# Leader: runs Sends to workers the N_MIDPOINTS associated with that rank
	# MIDPOINTS_RANK
	# collect their contributions. Also calculate a sub-set of points.
    for i in range(0,MIDPOINTS_REMAINDER):
        # mid-point rule
        integrand_input = (i+0.5) * DELTA
        integrand_output = integrand(integrand_input) * DELTA
        I_TOTAL += integrand_output
elif rank != 0:
    LOWER_BOUND = MIDPOINTS_REMAINDER + MIDPOINTS_RANK * (rank - 1)
    UPPER_BOUND = MIDPOINTS_REMAINDER + MIDPOINTS_RANK * rank
    print("i am rank", rank, "my limits are", LOWER_BOUND, "to", UPPER_BOUND)
    for i in range (LOWER_BOUND, UPPER_BOUND):
        integrand_input = (i+0.5) * DELTA
        integrand_output = integrand(integrand_input) * DELTA
        I_PARTIAL += integrand_output
    print("I am rank", rank, "of", nproc, "I_PARTIAL = ", I_PARTIAL)
    comm.send(I_PARTIAL,dest=0)
    print("I am rank", rank, "of", nproc, "message containing I_PARTIAL sent")

#receive worker results
#integrand_output = comm.recv(I_PARTIAL, source=?)
if rank == 0:
    for SOURCE_RANK in range(1, nproc):
        I_PARTIAL = comm.recv(source = SOURCE_RANK)
        print("rank 0 has recived", I_PARTIAL, "from rank", SOURCE_RANK)
        I_TOTAL += I_PARTIAL
    print("pi calculated to be = ", f'{I_TOTAL:.20f}')
    print("pi = 3.1415926535897932385")
