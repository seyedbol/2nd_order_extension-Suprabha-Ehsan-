#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:31:45 2019

@author: ehsan
"""
import fenics
import dolfin
from dolfin import *
import CGAL
from CGAL import *
# =============================================================================
# ################### triangular mesh 
# =============================================================================

mesh = dolfin.RectangleMesh(Point(0,0), Point(2, 1), 10,10)
print("Plotting a RectangleMesh")
plot(mesh, title="Rectangle (right/left)")
V = dolfin.FunctionSpace(mesh, "Lagrange", 1)
interactive()
# =============================================================================
# structure mesh
# =============================================================================
