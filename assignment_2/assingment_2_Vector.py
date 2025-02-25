#!/usr/bin/env python3

"""
Created on Tue Feb 11 12:10:36 2025

@author: Finnghuala

Creation and use of class objects for cartesian and spherical vectors
for the completion of assignment two of PH510:ACP

"""

import numpy as np

class Vector:
    """
    3d vector for assignment 2, task 1, PH510 ACP
    """
    def __init__(self, x_input, y_input, z_input):
        """
        builds the vector object
        """
        self.x_store = x_input
        self.y_store = y_input
        self.z_store = z_input

    def __str__(self):
        """
        prints floating point to 3 decimal places
        """
        return f"Vector:({self.x_store:.3f}, {self.y_store:.3f}, {self.z_store:.3f})"

    def mag(self):
        """
        returns the magintude of the input vector
        """
        return np.sqrt(self.x_store ** 2 + self.y_store ** 2 + self.z_store ** 2 )

    def __add__(self, other):
        """
        overloads addition operator for addition of two instances
        """
        # if isinstance(self, SphericalPolarVector) and isinstance(other, SphericalPolarVector):
        #     return SphericalPolarVector(self.x_store + other.x_store,
        #                   self.y_store + other.y_store,
        #                   self.z_store + other.z_store)

        return Vector(self.x_store + other.x_store,
                      self.y_store + other.y_store,
                      self.z_store + other.z_store)

    def __sub__(self, other):
        """
        overloads subtraction operator for subtraction of two instances
        """
        return Vector(self.x_store - other.x_store,
                      self.y_store - other.y_store,
                      self.z_store - other.z_store)

    def dot(self, other):
        """
        returns the dot product of two vectors
        """
        return (self.x_store * other.x_store +
                self.y_store * other.y_store +
                self.z_store * other.z_store)

    def cross(self, other):
        """
        returns the cross product of two vectors
        """
        return Vector(self.y_store * other.z_store - self.z_store * other.y_store,
         self.z_store * other.x_store - self.x_store * other.z_store,
         self.x_store * other.y_store - self.y_store * other.x_store)

def triangle_area(vertex_1, vertex_2, vertex_3,):
    """
    self is the first vertex with vertex_2 and vertex_3 defining the other
    two points of the triangle. triangle is calculated using the formula:
        area = 1/2 * |AB x AC|
    """
    side_1 = vertex_2 - vertex_1
    side_2 = vertex_3 - vertex_1

    return f"{((1/2)*Vector.mag(Vector.cross(side_1, side_2))):.3f}"

def triangle_internal_angle(a_input, b_input, c_input):
    """
    Calculates the internal angles of a triangle
    """
    ab_vector = b_input - a_input
    cb_vector = b_input - c_input
    bc_vector= c_input - b_input
    ac_vector= c_input - a_input

    abc = np.arccos(Vector.dot(ac_vector,ab_vector)/
                    (Vector.mag(ac_vector)*Vector.mag(ab_vector))) * 180/np.pi

    bca = np.arccos(Vector.dot(cb_vector,ab_vector)/
                    (Vector.mag(cb_vector)*Vector.mag(ab_vector))) * 180/np.pi

    cab = np.arccos(Vector.dot(bc_vector,ac_vector)/
                    (Vector.mag(bc_vector)*Vector.mag(ac_vector))) * 180/np.pi
    return f'Angle 1 = {abc:.3f}\n Angle 2 = {bca:.3f}\n Angle 3 = {cab:.3f}'

print("start of cartesian checks\n")

a = Vector(1, 1, 1)

print("vector a = ", a,"\n")

b = Vector(2, 2, 2)

print("vector b = ", b,"\n")

c = Vector.mag(a)

print("the magnitude of a is", c,"\n")

d = a + b

print("the sum of a and b is", d,"\n")

e = a - b

print("the difference of a and b is", e,"\n")

f = Vector.dot(a, b)

print("dot product of vector a and b =", f,"\n")

h = Vector.cross(a, b)

print("cross product of vector a and b =", h,"\n")

print("end of cartesian checks\n")

class SphericalPolarVector(Vector):
    """
    Polar extention of the Vector class
    
    Vector input is spherical polar, vector is internally stored as cartesian
    
    input is in degrees then converted to radians
    
    theta is the angle from the z-axis, 0 to pi
    
    phi is the azimuthal angle measured from the x-axis, 0 to 2pi
    """
    def __init__(self, r_input, theta_input, phi_input):

        theta_input = theta_input*np.pi/180
        phi_input = phi_input*np.pi/180

        Vector.__init__(self,
                        r_input*np.sin(theta_input)*np.cos(phi_input),
                        r_input*np.sin(theta_input)*np.sin(phi_input),
                        r_input*np.cos(theta_input))
    def radial(self):
        """
        returns the magnitude/radial component of the vector
        """
        return self.mag()

    def theta(self):
        """
        returns the polar angle
        """
        if self.radial() == 0:
            return 0
        return (np.arccos(self.z_store/self.mag()))*180/np.pi

    def phi(self):
        """
        returns the azimuthal angle
        """
        if self.radial() == 0:
            return 0
        return ((((np.arctan2(self.y_store,self.x_store))*180/np.pi)+360)%360)


    def __str__(self):
        """
        prints floating point to 3 decimal places
        """
        return f"Vector:({self.radial():.3f},{self.theta():.3f}\u03B8,{self.phi():.3f}\u03C6)"

    def __add__(self, other):
        """
        overloads the add funciton for the spherical vector.
        
        inputs are two spherical vectors
        
        addition is performed for x, y, and z and then stored in an intermediate value
        
        this intermeidate value is a cartesian out put that is then restored
        in to a spherical vector construct.
        """

        x_intermediate = self.x_store + other.x_store
        y_intermediate = self.y_store + other.y_store
        z_intermediate = self.z_store + other.z_store

        radial = np.sqrt(x_intermediate**2 + y_intermediate**2 + z_intermediate**2)

        if radial == 0:
            theta = 0
        else: theta = (np.arccos(z_intermediate/radial))*180/np.pi

        if radial == 0:
            phi = 0
        else: phi = ((((np.arctan2(y_intermediate, x_intermediate))*180/np.pi)+360)%360)

        return SphericalPolarVector(radial, theta, phi)

    def __sub__(self, other):
        """
        overloads the add funciton for the spherical vector.
        
        inputs are two spherical vectors
        
        subtraction is performed for x, y, and z and then stored in an intermediate value
        
        this intermeidate value is a cartesian out put that is then restored
        in to a spherical vector construct.
        """

        x_intermediate = self.x_store - other.x_store
        y_intermediate = self.y_store - other.y_store
        z_intermediate = self.z_store - other.z_store

        radial = np.sqrt(x_intermediate**2 + y_intermediate**2 + z_intermediate**2)

        if radial == 0:
            theta = 0
        else: theta = (np.arccos(z_intermediate/radial))*180/np.pi

        if radial == 0:
            phi = 0
        else: phi = ((((np.arctan2(y_intermediate, x_intermediate))*180/np.pi)+360)%360)

        return SphericalPolarVector(radial, theta, phi)

    def cross(self, other):
        """
        returns the cross product of two spherical vectors
        
        inputs are two spherical vectors
        
        cross is performed for x, y, and z and then stored in an intermediate value
        
        this intermeidate value is a cartesian out put that is then restored
        in to a spherical vector construct.
        
        """
        x_intermediate = self.y_store * other.z_store - self.z_store * other.y_store
        y_intermediate = self.z_store * other.x_store - self.x_store * other.z_store
        z_intermediate = self.x_store * other.y_store - self.y_store * other.x_store

        radial = np.sqrt(x_intermediate**2 + y_intermediate**2 + z_intermediate**2)

        if radial == 0:
            theta = 0
        else: theta = (np.arccos(z_intermediate/radial))*180/np.pi

        if radial == 0:
            phi = 0
        else: phi = ((((np.arctan2(y_intermediate, x_intermediate))*180/np.pi)+360)%360)

        return SphericalPolarVector(radial, theta, phi)




# shouldnt need to rewrite my funcitons defeats the purpose of a big bit of the inheratince

print("\nStart of spherical checks\n")
Sp_1 = SphericalPolarVector(1, 0, 0)

Sp_2 = SphericalPolarVector(1, 45, 0)

Sp_3 = SphericalPolarVector(1, 90, 0)

Sp_4 = SphericalPolarVector(1, 90, 45)

print("Sp_1 is the Spherical", Sp_1,
      "\nSp_2 is the Spherical", Sp_2,
      "\nSp_3 is the Spherical", Sp_3,
      "\nSp_4 is the Spherical", Sp_4)

i = SphericalPolarVector.mag(Sp_1)

j = Sp_1 + Sp_2

k = Sp_3 - Sp_1

l = SphericalPolarVector.cross(Sp_1, Sp_3)

m = SphericalPolarVector.dot(Sp_2, Sp_3)

print("\ni is the magnitude of Sp_1 =", i)

print("\nj = Sp_1+Sp_2, j is the", j)

print("\nk = Sp_3-Sp_2, k is the", k)

print("\nl is the cross product of Sp_1 and Sp_3, l is the", l)

print("\nm is the dot product of Sp_2 and Sp_3, m is ", f'{m:.3}')

print("\nEnd of spherical checks\n")


print("\nStart of cartesian triangles\n\n3.a)\n")

A = Vector(0, 0, 0)
B = Vector(1, 0, 0)
C = Vector(0, 1, 0)

ABC_Area = triangle_area(A, B, C)

print("area of triangle 1, ABC = ", ABC_Area)

D = Vector(-1, -1, -1)
E = Vector(0, -1, -1)
F = Vector(-1, 0, -1)

DEF_Area = triangle_area(D, E, F)

print("\narea of triangle 2, DEF = ", DEF_Area)


G = Vector(1, 0, 0)
H = Vector(0, 0, 1)
I = Vector(0, 0, 0)

GHI_Area = triangle_area(G, H, I)

print("\narea of triangle 2, GHI = ", GHI_Area)

J = Vector(0, 0, 0)
K = Vector(1, -1, 0)
L = Vector(0, 0, 1)

JKL_Area = triangle_area(J, K, L)

print("\narea of triangle 4, JKL = ", JKL_Area)

print("\n3.b)")

ABC_Angle = triangle_internal_angle(A, B, C)

print("\nFor triangle 1, ABC:\n", ABC_Angle)

DEF_Angle = triangle_internal_angle(D, E, F)

print("\nFor triangle 2, DEF:\n", DEF_Angle)

GHI_Angle = triangle_internal_angle(G, H, I)

print("\nFor triangle 3, GHI:\n", GHI_Angle)

JKL_Angle = triangle_internal_angle(J, K, L)

print("\nFor triangle 4, JKL:\n", JKL_Angle)

print("\nEnd of cartesian triangles\n\n\nStart of spherical triangles")



M = SphericalPolarVector(0, 0, 0)
N = SphericalPolarVector(1, 0, 0)
O = SphericalPolarVector(1, 90, 0)

MNO_Area = triangle_area(M, N, O)

MNO_Angle = triangle_internal_angle(M, N, O)

print("\nFor triangle 5, MNO,\n Area = ", MNO_Area,"\n", MNO_Angle)


P = SphericalPolarVector(1, 0, 0)
Q = SphericalPolarVector(1, 90, 0)
R = SphericalPolarVector(1, 90, 180)

PQR_Area = triangle_area(P, Q, R)

PQR_Angle = triangle_internal_angle(P, Q, R)

print("\nFor triangle 6, PQR,\n Area = ", PQR_Area,"\n", PQR_Angle)


S = SphericalPolarVector(0, 0, 0)
T = SphericalPolarVector(2, 0, 0)
U = SphericalPolarVector(2, 90, 0)

STU_Area = triangle_area(S, T, U)

STU_Angle = triangle_internal_angle(S, T, U)

print("\nFor triangle 7, STU,\n Area = ", STU_Area,"\n", STU_Angle)


V = SphericalPolarVector(1, 90, 0)
W = SphericalPolarVector(1, 90, 180)
X = SphericalPolarVector(1, 90, 270)

VWX_Area = triangle_area(V, W, X)

VWX_Angle = triangle_internal_angle(V, W, X)

print("\nFor triangle 8, VWX,\n Area = ", VWX_Area,"\n", VWX_Angle)

check1 = Vector(1/2, 0, 0)
check2 = Vector(0, np.sqrt(3)/2, 0)
check3 = Vector(0, 0, 0)

check_Area = triangle_area(check1, check2, check3)

check_Angle = triangle_internal_angle(check1, check2, check3)

print("\nFor check triangle\n Area = ", check_Area,"\n", check_Angle)
