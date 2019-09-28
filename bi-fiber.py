#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:05:09 2019

@author: suprabha
"""
from __future__ import print_function
from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

L = 2.
H = 1.
nx = 20 
ny = 10
mesh = RectangleMesh(Point(0., 0.), Point(L, H), nx, ny)

# Initializing material parameters
mu = 0.101
E1 = 2/mu
E2 = 3/mu
c1 = 0.1/mu
c2 = 0.1/mu
mu = 1


# Defining non-linear terms...
def c(x1):
    return x1.dx(0)

def d(x2):
    return x2.dx(0)

def g(x1):
    return x1.dx(1)

def s(x2):
    return x2.dx(1)

# Defining test and trial functions in splitted form
element = VectorElement('P', triangle, 1, dim=8)
V = FunctionSpace(mesh, element)

v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8 = TestFunctions(V)
u = Function(V)
x1, x2, q, r, h, t, A, B = split(u)

# Defining boundary conditions
class left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0)
class right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],L)
class top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],H)
class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0)

facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
left().mark(facets, 1)
right().mark(facets, 2)
top().mark(facets, 3)
bottom().mark(facets, 4)
ds = Measure("ds", subdomain_data=facets)

bc = DirichletBC(V, Constant((0.,0.)), facets, 1)

# Define source terms
x = SpatialCoordinate(mesh)

f_1 = (c1*q.dx(0)+c2*h.dx(1))*ds(2)
f_2 = (c1*r.dx(0)+c2*t.dx(1))*ds(2)
f_3 = x1.dx(0)*ds(2)
f_4 = x2.dx(0)*ds(2)
f_5 = x1.dx(1)*ds(2)
f_6 = x2.dx(1)*ds(2)
f_7 = c1*r.dx(0)*ds(2)
f_8 = c1*q.dx(0)*ds(2)

# Define expressions used in variational forms
E1 = Constant(E1)
E2 = Constant(E2)
c1 = Constant(c1)
c2 = Constant(c2)
mu = Constant(mu)

# Define variational problem
F = (2*mu*(q+h)*v_1-A*s(x2)*v_1+B*d(x2)*v_1+c1*q.dx(0)*v_1.dx(0) / 
    + c2*h.dx(1)*v_1.dx(1)-0.5*E1*q*v_1-0.5*E2*h*v_1 /
    +0.5*E1*(3*q*c(x1)*c(x1)+q*d(x2)*d(x2)+2*r*c(x1)*d(x2))*v_1 / 
    +0.5*E2*(3*h*g(x1)*g(x1)+h*s(x2)*s(x2)+2*t*g(x1)*s(x2))*v_1)*dx +(2*mu*(r+t)*v_2 /
    +A*g(x1)*v_2-B*c(x1)*v_2+c1*r.dx(0)*v_2.dx(0) /  
    +c2*t.dx(1)*v_2.dx(1)-0.5*E1*r*v_2-0.5*E2*t*v_2 /
    +0.5*E1*(3*r*d(x2)*d(x2)+r*c(x1)*c(x1)+2*q*d(x2)*c(x1))*v_2 / 
    +0.5*E2*(3*t*s(x2)*s(x2)+t*g(x1)*g(x1) / 
    +2*h*s(x2)*g(x1))*v_2)*dx +(q*v_3-x1.dx(0)*v_3.dx(0))*dx+(r*v_4 / 
    -x2.dx(0)*v_4.dx(0))*dx +(h*v_5-x1.dx(1)*v_5.dx(1))*dx+(t*v_6-x2.dx(1)*v_6.dx(1))*dx +(A*v_7 / 
    -mu*(q+h)*v_7+c1*r.dx(0)*v_7.dx(0))*dx +(B*v_8-mu*(r+t)*v_8 / 
    +c1*q.dx(0)*v_8.dx(0))*dx - f_1*v_1*dx - f_2*v_2*dx - f_3*v_3*dx - f_4*v_4*dx - f_5*v_5*dx - f_6*v_6*dx - f_7*v_7*dx - f_8*v_8*dx
    

# Solve PDEs
solve(F == 0, u, bc)

# Showing Results
x1, x2, q, r, h, t, A, B = u.split()

plot(mesh)
plt.show()
