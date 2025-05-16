# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:10:57 2025

@author: finnd
"""

import numpy as np

import matplotlib.pyplot as plt

class Relaxaton:

    def 
# Phi
Phi = np.random.random([N,N])
# boundary conditions
Phi[0, :] = 1
Phi[N-1, :] = 1
Phi[:, 0] = 0
Phi[:, N-1] = 0

# PhiPrime
PhiPrime = np.zeros([N,N])
# PhiPrime boundary conditions
PhiPrime[0, :] = 1
PhiPrime[N-1, :] = 1
PhiPrime[:, 0] = 0
PhiPrime[:, N-1] = 0

# change tracking array
change = 1

# f(x,y)
charge = np.zeros([N,N])

# spacing
l = 10e-2 # meters
h = l/(N-2)

# iteration tracker
iteration_N = 0

# if the difference between 
tolerence = 10e-9


while change > tolerence:
    for i in range(1,N-1):
        for j in range(1,N-1):
            PhiPrime[i,j] = (1/4)*(Phi[i+1,j] + Phi[i-1,j] + Phi[i,j+1] + Phi[i,j-1] + (h**2)*charge[i,j])
    change = np.max(np.abs(Phi-PhiPrime))
    np.copyto(Phi, PhiPrime)
    iteration_N +=1


print("iterations till converge = ", iteration_N)


extent = [-h*100, l * 100 + h*100, -h*100, l * 100 + h*100]


plt.figure(0)
plt.imshow(np.round(Phi,6), origin="lower", extent=extent,cmap = "viridis") 
plt.colorbar(label = "Potential(V)")
