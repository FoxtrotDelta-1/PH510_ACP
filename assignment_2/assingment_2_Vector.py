#!/usr/bin/env python3


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
        return f"Cartesian Vector:({self.x_store:.3f}, {self.y_store:.3f}, {self.z_store:.3f})"

    def mag(self):
        """
        returns the magintude of the input vector
        """
        return np.sqrt(self.x_store ** 2 + self.y_store ** 2 + self.z_store ** 2 )

    def __add__(self, other):
        """
        overwrites addition operator for addition of two instances
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
        overwrites subtraction operator for subtraction of two instances
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

    return (1/2)*Vector.mag(Vector.cross(side_1, side_2))

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
    return f' Angle 1 = {abc:.3f}\n Angle 2 = {bca:.3f}\n Angle 3 = {cab:.3f}'


a = Vector(7, 5, 3)

print(a)

b = Vector(10, 2, 10)

print(b)

c = Vector.mag(a)

print(c)

d = a + b

print(d)

e = a - b

print(e)

f = Vector.dot(a, b)

print(f)

h = Vector.cross(a, b)

print(h)


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

        x_intermediate = self.x_store + other.x_store
        y_intermediate = self.y_store + other.y_store
        z_intermediate = self.z_store + other.z_store

        self.x_store = np.sqrt(x_intermediate**2 + y_intermediate**2 + z_intermediate**2)

        if np.sqrt(x_intermediate**2 + y_intermediate**2 + z_intermediate**2) == 0:
            self.y_store = 0
        else: self.y_store = (np.arccos(z_intermediate/
        np.sqrt(x_intermediate**2 + y_intermediate**2 + z_intermediate**2)))*180/np.pi

        if np.sqrt(x_intermediate**2 + y_intermediate**2 + z_intermediate**2) == 0:
            self.z_store = 0
        else: self.z_store = ((((np.arctan2(y_intermediate, x_intermediate))*180/np.pi)+360)%360)

        return SphericalPolarVector(self.x_store, self.y_store, self.z_store)



# shouldnt need to rewrite my funcitons defeats the purpose of a big bit of the inheratince


Sp_1 = SphericalPolarVector(1, 135, 0)

Sp_2 = SphericalPolarVector(1, 90, 0)

i = Sp_1 + Sp_2

print("i is the", i)


A = Vector(0, 0, 0)
B = Vector(1, 0, 0)
C = Vector(0, 1, 0)

ABC_Area = triangle_area(A, B, C)

print("area of triangle 1, ABC = ", ABC_Area)

D = Vector(-1, -1, -1)
E = Vector(0, -1, -1)
F = Vector(-1, 0, -1)

DEF_Area = triangle_area(D, E, F)

print("area of triangle 2, DEF = ", DEF_Area)


G = Vector(1, 0, 0)
H = Vector(0, 0, 1)
I = Vector(0, 0, 0)

GHI_Area = triangle_area(G, H, I)

print("area of triangle 2, GHI = ", GHI_Area)

J = Vector(0, 0, 0)
K = Vector(1, -1, 0)
L = Vector(0, 0, 1)

JKL_Area = triangle_area(J, K, L)

print("area of triangle 4, JKL = ", JKL_Area)


ABC_Angle = triangle_internal_angle(A, B, C)

print("For triangle 1, ABC:", ABC_Angle)

DEF_Angle = triangle_internal_angle(D, E, F)

print("For triangle 2, DEF:", DEF_Angle)

GHI_Angle = triangle_internal_angle(G, H, I)

print("For triangle 3, GHI", GHI_Angle)

JKL_Angle = triangle_internal_angle(J, K, L)

print("For triangle 4, JKL", JKL_Angle)
