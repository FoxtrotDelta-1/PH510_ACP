#!/usr/bin/env python3


import numpy as numpy

class Vector:
    """
    3d vector for assignment 2, task 1, PH510 ACP
    """
    def __init__(self, x_input, y_input, z_input):
        """
        builds the vector object
        """
        self.x = x_input
        self.y = y_input
        self.z = z_input

    def __str__(self):
        """
        prints floating point to 3 decimal places
        """
        return f"Vector:({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __add__(self, other):
        """
        overwrites addition operator for addition of two instances
        """
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """
        overwrites subtraction operator for subtraction of two instances
        """
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        """
        returns the dot product of two vectors
        """
        return self.x * other.x + self.y * other.y + self.z * self.z + other.z

    def cross(self, other):
        """
        returns the cross product of two vectors
        """
        return Vector(self.y * other.z - self.z * other.y,
         self.z * other.x - self.x * other.z,
         self.x * other.y - self.y * other.x)
