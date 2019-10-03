from __future__ import print_function
from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:36:00 2019

@author: ehsan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:05:09 2019
@author: suprabha
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
element = VectorElement("CG", triangle, 2, dim=12)
V = FunctionSpace(mesh, element)
Q = FunctionSpace(mesh, "CG", 2)

plot(mesh)
plt.show()

v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9, v_10, v_11, v_12 = TestFunctions(V)
u = Function(V)
x1, x2, q, r, h, t, A, B, C, D, G, S = split(u)

# Defining Derivatives
#x1dx = project(x1.dx(0), Q)
#x1dy = project(x1.dx(1), Q)
#qdx = project(q.dx(0), Q)
#v_1dx = project(v_1.dx(0), Q)
#x2dx = project(x2.dx(0), Q)
#x2dy = project(x2.dx(1), Q)

# Defining non-linear terms...
def c(x1):
    return x1.dx(0)

def d(x2):
    return x2.dx(0)

def g(x1):
    return x1.dx(1)

def s(x2):
    return x2.dx(1)

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

bc = [DirichletBC(V.sub(0), Constant(0.), facets, 1),
      DirichletBC(V.sub(1), Constant(0.), facets, 1)]


f_1 = (c1*q.dx(0)+c2*h.dx(1))*ds
f_2 = (c1*r.dx(0)+c2*t.dx(1))*ds
f_3 = x1.dx(0)*ds(2)
f_4 = x2.dx(0)*ds
f_5 = 0
f_6 = 0 
f_7 = x2.dx(1)*ds
f_8 = 0
f_9 = 0
f_10 = x1.dx(1)*ds
f_11 = (x1.dx(0)+x1.dx(1))*ds
f_12 = (x2.dx(0)+x2.dx(1))*ds 
g = 1
#-5 + ( (2*mu/c1) * u[8] ) + ( E1/(2*c1) )*(pow(u[8],2)+pow(u[9],2)-1)*
###I am not sure  these are the best parameters for the first guess we can discuss it
A_z=np.zeros((100,1))
B_z=np.zeros((100,1))
C_z=np.zeros((100,1))
D_z=np.zeros((100,1))
S_z=np.zeros((100,1))
G_z=np.zeros((100,1))
for i in range(0,100,1):

# Define variational problem
    F = (  ( 2 * mu * (q+h) ) - ( A_z[i,0] * s(x2) ) + ( B_z[i,0] * d(x2) ) - ( 0.5*E1*q )-( 0.5*E2*h )##1
    +   ( 0.5*E1 ) *( ( 3*q*C_z[i,0]*C_z[i,0] )+( q*D_z[i,0]*D0[i,0] )+ ( 2*r*C_z[i,0]*D0[i,0]) )
    +   ( 0.5*E2 ) *( ( 3*h*G_z[i,0]*G0[i,0] )+( h*S_z[i,0]*S0[i,0] )+ ( 2*t*G_z[i,0]*S_z[i,0]) ) )*v_1*dx 
    +   ( ( c1*q.dx(0)*v_1.dx(0)  )+( c2*h.dx(1)*v_1.dx(1) ) )*dx 
    +   (  ( 2 * mu * (r+t) ) + ( A_z[i,0] * g(x1) ) - ( B_z[i,0] * c(x1) ) - ( 0.5*E1*r ) -( 0.5*E2*t )  ##2
    +   ( 0.5*E1 ) *( (3*r*D_z[i,0]*D_z[i,0]) + ( r*C_z[i,0]*C_z[i,0] )  + ( 2*q*D_z[i,0]*C_z[i,0] )  ) +
    +   ( 0.5*E2 ) *( (3*t*S_z[i,0]*S_z[i,0]) + ( t*G_z[i,0]*G_z[i,0] )  + ( 2*h*S_z[i,0]*G_z[i,0] )) )*v_2*dx                                                           
    +   ( ( c1*r.dx(0)*v_2.dx(0)   )+( c2*t.dx(1)*v_2.dx(1) ) )*dx 
# =============================================================================
#     +   ( ( c(x1)*x2.dx(1) ) - ( d(x2)*x1.dx(1) + 1 ) )*v_3*dx  ##3
# =============================================================================
    +   ( ( q*v_3 ) + ( x1.dx(0)*v_3.dx(0) ) )*dx  ##3
    +   ( ( r*v_4 ) + ( x2.dx(0)*v_4.dx(0) ) )*dx ##4
    +   (  c(x1)-x1.dx(0) )*v_5*dx ##5
    +   (  d(x2)-x2.dx(0) )*v_6*dx ##6
    +   ( ( t*v_7 ) + ( x2.dx(1)*v_7.dx(1) ) )*dx ##7
    +   ( s(x2)-x2.dx(1)  )*v_8*dx ##8
    +   ( g(x1)-x1.dx(1)  )*v_9*dx #9
    +   ( ( h*v_10 ) + ( x1.dx(1)*v_10.dx(1) ) )*dx##10 
    +   ( A*v_11 + mu*( x1.dx(0)*v_11.dx(0) ) + mu* ( x1.dx(1)*v_11.dx(1) )  )*dx ##11
    +   ( B*v_12 + mu*( x2.dx(0)*v_12.dx(0) ) + mu* ( x2.dx(1)*v_12.dx(1) )  )*dx ##12 
    - f_1*v_1- f_2*v_2- f_3*v_3 - f_4*v_4- f_5*v_5 - f_6*v_6 - f_7*v_7 - f_8*v_8 
    - f_9*v_9 - f_10*v_10 -f_11*v_11-f_12*v_12 +v_3*g*ds(2)
    solve(F == 0, u, bc)
# Showing Results
    x1, x2, q, r, h, t, A, B, C, D, G, S = u.split()
    A0[i+1,1]= A   
    B0[i+1,1]= B
    C0[i+1,1]= C
    D0[i+1,1]= D
    S0[i+1,1]= s
    G0[i+1,1]= G
        
        
        
         



# Solve PDEs



plot(x1, title="x1 plot" , mode= "color")
plt.show()
plot(x2, title="x2 plot", mode='color')
plt.show()

vtkfile_x1 = File('Bi-Directional_Fiber/x1.pvd')
vtkfile_x2 = File('Bi-Directional_Fiber/x2.pvd')

vtkfile_x1 << (x1)
vtkfile_x2 << (x2)