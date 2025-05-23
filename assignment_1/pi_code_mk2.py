#!/usr/bin/env python3


from mpi4py import MPI
import numpy as np 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
print(nproc)
# The first processor is leader, so one fewer available to be a worker
nworkers = nproc - 1
# samples
N_MIDPOINTS = 100
DELTA = 1.0 / N_MIDPOINTS
MIDPOINTS_RANK = N_MIDPOINTS//(nworkers)
MIDPOINTS_REMAINDER = N_MIDPOINTS%(nworkers)



# integral
I_Total= 0.0
I_Partial = 0.0
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
		integrand_input = (i+0.5) * DELTA # can be moved out of the loops?
		integrand_output=integrand(integrand_input) * DELTA
		I_Total += integrand_output
	# receive worker results
#	integrand_output = comm.recv(I_Partial, source=?)
	integrand_output = comm.reduce(I_Partial, op=MPI.SUM, root=0)
	I_Total += integrand_output
	print("Integral %.10f" % I)

if rank != 0:
	print("i am rank", rank, "my limits are",MIDPOINTS_REMAINDER + MIDPOINTS_RANK * (rank - 1), "to", (MIDPOINTS_REMAINDER + MIDPOINTS_RANK * rank))
	for i in range ((MIDPOINTS_REMAINDER + MIDPOINTS_RANK * (rank - 1)), (MIDPOINTS_REMAINDER + MIDPOINTS_RANK * rank)):
		integrand_input = (i+0.5) * DELTA # can be moved out of the loops?
		integrand_output = integrand(integrand_input) * DELTA
		I_Partial += integrand_output
	print("I am rank", rank, "of", nproc, "I_Partial = ", I_Partial)
#	comm.send(I_Partial,dest=0)
	print("I am rank", rank, "of", nproc, "message containing I_Partial sent")

