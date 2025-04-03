#!/bin/bash

#======================================================
#
# Job script for running assignment 3  of PH510:ACP
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required (max. of 16)
#SBATCH --ntasks=16 --exclusive
#
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=00:30:00
# Job name
#SBATCH --job-name=assignment3
#
# Output file
#SBATCH --output=assignment3-%j.out
#======================================================

module purge

module add miniconda/3.12.8

module load openmpi/gcc-8.5.0/4.1.1


#======================================================
# Prologue script to record job details
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------
#pylint --extension-pkg-allow-list=mpi4py.MPI ./assignment3

mpirun -np 1 ./assignment3

mpirun -np 2 ./assignment3

mpirun -np 4 ./assignment3

mpirun -np 8 ./assignment3

mpirun -np 16 ./assignment3

#======================================================
# Epilogue script to record job endtime and runtime
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
