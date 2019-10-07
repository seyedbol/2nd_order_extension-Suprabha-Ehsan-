from __future__ import print_function
from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from random import random 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:36:00 2019

@author: ehsan
"""
# =============================================================================
# 
# =============================================================================
L = 2.
H = 1.
nx = 20 
ny = 10
mesh = RectangleMesh(Point(0., 0.), Point(L, H), nx, ny)

# =============================================================================
# we need to dicuss the material parameters( I have some doubts about the values)
# =============================================================================
# Initializing material parameters

mu = Constant(0.001)
E1 = Constant(135/mu)
E2 = Constant(90/mu)
c1 = Constant(150/mu)
c2 = Constant(100/mu)

# =============================================================================
# 
# =============================================================================

# Defining test and trial functions in splitted form
element = VectorElement("CG", triangle, 2, dim=13)
element_2 = VectorElement("CG", triangle, 2, dim=12)
V = FunctionSpace(mesh, element)
VV = FunctionSpace(mesh, element_2)
plot(mesh)
plt.show()
v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9, v_10, v_11, v_12, v_13 = TestFunctions(V)
u = Function(V)
x1, x2, q, r, h, t, A, B, C, D, G, S , KK = u
# =============================================================================
# non_linear terms
# =============================================================================
def c(x1):
    return x1.dx(0)

def d(x2):
    return x2.dx(0)

def g(x1):
    return x1.dx(1)

def s(x2):
    return x2.dx(1)
def k(x1):
    return c(x1).dx(0).dx(0)

# =============================================================================
#  meshing and boundary conditions 
# =============================================================================
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
ds = Measure("ds",subdomain_data=facets)

bc = [DirichletBC(V.sub(0), Constant(0.), facets, 1),
      DirichletBC(V.sub(1), Constant(0.), facets, 1)]
# =============================================================================
# w1= Expression('2*mu', degree =1 )
# w2 = Expression('E1', degree =1 )
# w3 = Expression('-c1', degree =1 )
# =============================================================================

# =============================================================================
# 
# =============================================================================
# =============================================================================
    # Define variational problem
F = (  ( 2 * mu * (q+h) ) - ( A * s(x2) ) + ( B * d(x2) ) - ( 0.5*E1*q )-( 0.5*E2*h )##1
+   ( 0.5*E1 ) *( ( 3*q*c(x1)*c(x1) )+( q*d(x2)*d(x2) )+ ( 2*r*c(x1)*d(x2)) )
+   ( 0.5*E2 ) *( ( 3*h*g(x1)*g(x1) )+( h*s(x2)*s(x2) )+ ( 2*t*g(x1)*s(x2)) ) )*v_1*dx 
+   ( ( c1*q.dx(0)*v_1.dx(0)  )+( c2*h.dx(1)*v_1.dx(1) ) )*dx 
+   (  ( 2 * mu * (r+t) ) + ( A * g(x1) ) - ( B * c(x1) ) - ( 0.5*E1*r ) -( 0.5*E2*t )  ##2
+   ( 0.5*E1 ) *( (3*r*d(x2)*d(x2)) + ( r*c(x1)*c(x1) )  + ( 2*q*d(x2)*d(x2))  ) 
+   ( 0.5*E2 ) *( (3*t*s(x2)*s(x2)) + ( t*g(x1)*g(x1) )  + ( 2*h*s(x2)*g(x1) )) )*v_2*dx                                                           
+   ( ( c1*r.dx(0)*v_2.dx(0)   )+( c2*t.dx(1)*v_2.dx(1) ) )*dx 
+   ( ( c(x1)*x2.dx(1) ) - ( d(x2)*x1.dx(1) + 1 ) )*v_3*dx  ##3
+   ( ( q*v_4 ) + ( x1.dx(0)*v_4.dx(0) ) )*dx  ##4
+   ( ( r*v_5 ) + ( x2.dx(0)*v_5.dx(0) ) )*dx ##5
+   (  c(x1)-x1.dx(0) )*v_6*dx ##6
+   (  d(x2)-x2.dx(0) )*v_7*dx ##7
+   ( ( t*v_8 ) + ( x2.dx(1)*v_8.dx(1) ) )*dx ##8
+   ( s(x2)-x2.dx(1)  )*v_9*dx ##9
+   ( g(x1)-x1.dx(1)  )*v_10*dx #10
+   ( ( h*v_11 ) + ( x1.dx(1)*v_11.dx(1) ) )*dx##11
+   ( A*v_12 + mu*( x1.dx(0)*v_12.dx(0) ) + mu* ( x1.dx(1)*v_12.dx(1) )  )*dx ##12
+   ( B*v_13 + mu*( x2.dx(0)*v_13.dx(0) ) + mu* ( x2.dx(1)*v_13.dx(1) )  )*dx ##13
+    0.75 * v_3 * ds(2)
+   (c1*q.dx(0)+c2*h.dx(1))*v_1*ds(2)
+   (c1*r.dx(0)+c2*t.dx(1))*v_2*ds(2)
+   ( x1.dx(0) )*v_4*ds(2)
+   ( x2.dx(0) )*v_5*ds(2)
+   ( x2.dx(1) )*v_8*ds(2)
+   ( x1.dx(1) )*v_11*ds(2)
+   ( (x1.dx(0)+x1.dx(1)) )*v_12*ds(2)
+   ( (x2.dx(0)+x2.dx(1)) )*v_13*ds(2) + ( (2*mu+E1) * c(x1)*v_9*ds(2) ) - ( 2 * mu * s(x2)*v_12*ds(2) ) -  ( c1 * KK *v_13*ds(2) ) 
# =============================================================================
# ==========================================
# =============================================================================
###I am not sure  these are the best parameters for the first guess we can discuss it
# =============================================================================
# =============================================================================
# =============================================================================
gg = ('random()','random()','random()','random()','random()','random()','random()','random()','random()','random()','random()','random()','random()')
tup =Expression(gg,degree=1)
u_k = interpolate(tup, V)  # previous (known) u
eps = 1.0           # error measure ||u-u_k||
tol = 1.0E-10      # tolerance
iter = 0            # iteration counter
maxiter = 100       # max no of iterations allowed
while eps > tol and iter < maxiter:
    # =============================================================================
    # 
    # =============================================================================
    # Picard iterations
    iter += 1
    solve(F==0,u, bc)
    print("salam", u_k.vector().get_local())
    print("salam", u.vector().get_local())
    diff = u.vector().get_local() - u_k.vector().get_local()
    eps = np.linalg.norm(diff, ord=np.Inf)
    print('iter=%d: norm=%g' % (iter, eps) )
    u_k.assign(u)   # update for next iteration
    
# Showing Result
    


# Solve PDEs



# =============================================================================
# plot(x1, title="x1 plot" , mode= "color")
# plt.show()
# plot(x2, title="x2 plot", mode='color'  )
# plt.show()
# 
# vtkfile_x1 = File('Bi-Directional_Fiber/x1.pvd')
# vtkfile_x2 = File('Bi-Directional_Fiber/x2.pvd')
# 
# vtkfile_x1 << (x1)
# vtkfile_x2 << (x2)
# =============================================================================

