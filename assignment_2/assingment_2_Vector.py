#!/usr/bin/env python3


import numpy as numpy

class Vector:
    """
    3d vector for assignment 2, task 1, PH510 ACP
    """
    def__init__(self,x,y,z):
        """
        builds the vector object
        """
        self.x = x
        self.y = y
        self.z = z
    
    def__str__(self):
        """
        prints floating point to 3 decimal places
        """
        return f"Vector:({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    def__add__(self,other):
        """
        overwrites addition operator for two addition of vectors 
        """
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
                
    def__sub__(self, other)

