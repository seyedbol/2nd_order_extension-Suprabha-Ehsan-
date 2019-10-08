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
mu = Constant(0.101)
E1 = Constant(2/mu)
E2 = Constant(3/mu)
c1 = Constant(0.1/mu)
c2 = Constant(0.1/mu)
mu = Constant(1)
# =============================================================================
# 
# =============================================================================

# Defining test and trial functions in splitted form
element = VectorElement("P", triangle, 2, dim=12)
V = FunctionSpace(mesh, element)
plot(mesh)
plt.show()
v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9, v_10, v_11, v_12= TestFunctions(V)
u = Function(V)
x1, x2, q, r, h, t, A, B, C, D, G, S = u
# =============================================================================
# non_linear terms
# =============================================================================

# =============================================================================
# def C0(x1):
#     return x1.dx(0)
# 
# def D0(x2):
#     return x2.dx(0)
# 
# def G0(x1):
#     return x1.dx(1)
# 
# def S0(x2):
#     return x2.dx(1)
# 
# =============================================================================

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
      DirichletBC(V.sub(1), Constant(0.), facets, 1),
      DirichletBC(V.sub(0), Constant(1.), facets, 2),
      DirichletBC(V.sub(1), Expression('-(-0.08*pow(x[0],2)+0.3*x[0])', degree =2 ), facets, 3),
      DirichletBC(V.sub(1), Expression('-0.08*pow(x[0],2)+0.3*x[0]', degree =2 ), facets, 4)]

# =============================================================================

gg = ("0","0","0","0","0","0","0","0","0","0","0","0")
tup =Expression(gg,degree=1)
u_k = interpolate(tup, V)  # previous (known) u
x1_2, x2_2, q_2, r_2, h_2, t_2, A_2, B_2, C0, D0, G0, S0 = split(u_k)
# =============================================================================

F = (  ( 2 * mu * (q+h) ) - ( A * S0 ) + ( B * D0 ) - ( 0.5*E1*q )-( 0.5*E2*h )##1
+   ( 0.5*E1 ) *( ( 3*q*C0*C0)+( q*D0*D0 )+ ( 2*r*C0*D0 ) )
+   ( 0.5*E2 ) *( ( 3*h*G0*G0 )+( h*S0*S0 )+ ( 2*t*G0*S0) ) )*v_1*dx 
+   ( ( c1*q.dx(0)*v_1.dx(0)  )+( c2*h.dx(1)*v_1.dx(1) ) )*dx 
+   (  ( 2 * mu * (r+t) ) + ( A * G0 ) - ( B * C0 ) - ( 0.5*E1*r ) -( 0.5*E2*t )  ##2
+   ( 0.5*E1 ) *( (3*r*D0*D0) + ( r*C0*C0 )  + ( 2*q*D0*D0)  ) 
+   ( 0.5*E2 ) *( (3*t*S0*S0) + ( t*G0*G0 )  + ( 2*h*S0*G0) ) )*v_2*dx                                                           
+   ( ( c1*r.dx(0)*v_2.dx(0)   )+( c2*t.dx(1)*v_2.dx(1) ) )*dx 
+   ( ( q*v_3 ) + ( C*v_3.dx(0) ) )*dx  ##3
+   ( ( r*v_4 ) + ( D*v_4.dx(0) ) )*dx ##4
+   ( ( t*v_5 ) + ( S*v_5.dx(1) ) )*dx ##5
+   ( ( h*v_6 ) + ( G*v_6.dx(1) ) )*dx##6
+   ( A*v_7 - mu*( q+h ) + c1*r.dx(0)*v_7.dx(0) )*dx ##7
+   ( B*v_8 - mu*( r+t ) + c1*q.dx(0)*v_8.dx(0) )*dx ## 8
+   (  C-x1.dx(0) )*v_9*dx ##9
+   (  D-x2.dx(0) )*v_10*dx ##10
+   (  G-x1.dx(1)  )*v_11*dx #11
+   (  S-x2.dx(1)  )*v_12*dx ##12
-   (c1*q.dx(0)+c2*h.dx(1))*v_1*ds(2)
-   (c1*r.dx(0)+c2*t.dx(1))*v_2*ds(2)
-   ( C  )*v_3*ds(2)
-   ( D  )*v_4*ds(2)
-   ( S  )*v_5*ds(2)
-   ( G  )*v_6*ds(2)
-   ( c1*r.dx(0) )*v_7*ds(2)
-   ( c1*q.dx(0) )*v_8*ds(2) 

# =============================================================================

eps = 1.0           # error measure ||u-u_k||
tol = 1.0E-2      # tolerance
iter = 0            # iteration counter
maxiter = 100       # max no of iterations allowed

while eps > tol and iter < maxiter:
    # Picard iterations
    a, L = lhs(F), rhs(F)
    iter += 1
    solve( a==L, u, bc)
    print("%.30f" %u_k.vector().get_local()[1])
    print("%.30f" %u.vector().get_local()[1])
    diff = u.vector().get_local() - u_k.vector().get_local()
    eps = np.linalg.norm(diff, ord=np.Inf)
    print('iter=%d: norm=%g' % (iter, eps) )
    u_k.assign(u)   # update for next iteration
    x1_2, x2_2, q_2, r_2, h_2, t_2, A_2, B_2, C0, D0, G0, S0 = u_k
# =============================================================================
    
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

