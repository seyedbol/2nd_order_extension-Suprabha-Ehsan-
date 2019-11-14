from __future__ import print_function
from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import math

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:44:01 2019

@author: ehsan
"""
L = 2
H = 1
nx = 20 
ny = 20
mesh = RectangleMesh(Point(-L/2, -H/2), Point(L/2, H/2), nx, ny)
sag = mesh.coordinates()
# Initializing material parameters
mu = Constant(1)
E1 = Constant(20/mu)
E2 = Constant(20/mu)
c1 = Constant(20/mu)
c2 = Constant(20/mu)
p11 = Constant(200)




# Defining test and trial functions in splitted form
element = VectorElement('P', triangle, 1, dim=8)
V = FunctionSpace(mesh, element)
Q = FunctionSpace(mesh, "DG", 0)

plot(mesh)
plt.show()

v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8 = TestFunctions(V)
u = Function(V)
x1, x2, q, r, h, t, A, B = split(u)

x, y = SpatialCoordinate(mesh)

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
        return near(x[0],-L/2)

#def left(x, on_boundary):
#    return near(x[0],0.)    

class right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],L/2)
class top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],H/2)
class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],-H/2)
class middle(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0)

facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
left().mark(facets, 1)
right().mark(facets, 2)
top().mark(facets, 3)
bottom().mark(facets, 4)
middle().mark(facets, 5)
ds = Measure("ds", subdomain_data=facets)

##modification added for left and right edge 
u_R = Expression('x[1]', degree=1)
u_R1 = Expression('x[0]', degree=1)
bc = [DirichletBC(V.sub(0),Constant(0), facets, 5),
      DirichletBC(V.sub(1),u_R , facets, 1),#completely fixed edge (x2 remains the same on the left edge)
      DirichletBC(V.sub(1), u_R , facets, 2)]##completely fixed edge (x2 remains the same on the right edge)

      #DirichletBC(V.sub(0), Constant(1.), facets, 2),
      #DirichletBC(V.sub(0), Expression('0.5*x[0]', degree =2 ), facets, 3),
      #DirichletBC(V.sub(0), Expression('0.5*x[0]', degree =2 ), facets, 4),
# =============================================================================
#       DirichletBC(V.sub(1), Expression('-(-0.08*pow(x[0]-1.1,2))', degree =2 ), facets, 3),
#       DirichletBC(V.sub(1), Expression('-0.08*pow(x[0]-1.1,2)', degree =2 ), facets, 4)]
# =============================================================================
# Define source terms
# =============================================================================
# q_dx =  Constant(0)
# h_dy =  Constant(0)
# r_dx =  Constant(0)
# t_dy =  Constant(0)
# x1_dx = Constant(0)
# x1_dy = Constant(0)
# x2_dx = Constant(0)
# x2_dy = Constant(0)
# =============================================================================

# =============================================================================
# f_1 = (c1*q.dx(0)+c2*h.dx(1))*ds(2)
# f_2 = (c1*r.dx(0)+c2*t.dx(1))*ds(2)
# f_3 = x1.dx(0)*ds(2)
# f_4 = x2.dx(0)*ds(2)
# f_5 = x1.dx(1)*ds(2)
# f_6 = x2.dx(1)*ds(2)
# f_7 = c1*r.dx(0)*ds(2)
# f_8 = c1*q.dx(0)*ds(2)
# =============================================================================
f_1 = project((c1*q.dx(0)+c2*h.dx(1)),Q)
f_2 = project((c1*r.dx(0)+c2*t.dx(1)),Q)
f_3 = project(x1.dx(0),Q)
f_4 = project(x2.dx(0),Q)
f_5 = project(x1.dx(1),Q)
f_6 = project(x2.dx(1),Q)
f_7 = project(c1*r.dx(0),Q)
f_8 = project(c1*q.dx(0),Q)
# =============================================================================
f_9 = project((p11)/(2*mu + E1),Q)   
#f_10 = project((p22)/(2*mu + E1),Q) 
# =============================================================================
# Define variational problem
F = (2*mu*(q+h)*v_1-A*s(x2)*v_1+B*d(x2)*v_1+c1*q.dx(0)*v_1.dx(0) 
    + c2*h.dx(1)*v_1.dx(1)-0.5*E1*q*v_1-0.5*E2*h*v_1 
    +0.5*E1*(3*q*c(x1)*c(x1)+q*d(x2)*d(x2)+2*r*c(x1)*d(x2))*v_1  
    +0.5*E2*(3*h*g(x1)*g(x1)+h*s(x2)*s(x2)+2*t*g(x1)*s(x2))*v_1)*dx +(2*mu*(r+t)*v_2 
    +A*g(x1)*v_2-B*c(x1)*v_2+c1*r.dx(0)*v_2.dx(0)   
    +c2*t.dx(1)*v_2.dx(1)-0.5*E1*r*v_2-0.5*E2*t*v_2 
    +0.5*E1*(3*r*d(x2)*d(x2)+r*c(x1)*c(x1)+2*q*d(x2)*c(x1))*v_2  
    +0.5*E2*(3*t*s(x2)*s(x2)+t*g(x1)*g(x1) 
    +2*h*s(x2)*g(x1))*v_2)*dx +(q*v_3+x1.dx(0)*v_3.dx(0))*dx+(r*v_4  
    +x2.dx(0)*v_4.dx(0))*dx +(h*v_5+x1.dx(1)*v_5.dx(1))*dx+(t*v_6+x2.dx(1)*v_6.dx(1))*dx +(A*v_7 
    -mu*(q+h)*v_7+c1*q.dx(0)*v_7.dx(0))*dx +(B*v_8-mu*(r+t)*v_8  
    +c1*r.dx(0)*v_8.dx(0))*dx - f_1*v_1*ds - f_2*v_2*ds - f_3*v_3*ds - f_4*v_4*ds - f_5*v_5*ds - f_6*v_6*ds-f_7*v_7*ds -f_8*v_8*ds-f_9*v_3*ds(2)+f_9*v_3*ds(1)
# =============================================================================
#     -f_9*v_3*ds(2)
# =============================================================================

##PICARD Iteration
u_ = Expression(('0','0','0','0','0','0','0','0' ), degree =1)
u_k= interpolate(u_,V)
eps = 1.0
tol = 1.0E-2
iter = 0
maxiter = 25

while eps > tol and iter < maxiter:
    iter +=1
    solve(F == 0, u, bc)
    diff = u.vector().get_local() - u_k.vector().get_local()
    eps = np.linalg.norm(diff, ord=np.Inf)
    print ('iter= ', iter )
    print ('eps= ', eps)
    u_k.assign(u)

##to find the nodal values for each characteristic like x1,x2,...
u_nodal_values = np.zeros((441,2))
j=-1
for i in range (0,3528,8):
    j=j+1
    u_nodal_values[j][0] = u.vector().get_local()[i]
    u_nodal_values[j][1] = u.vector().get_local()[i+1]
def_x1= u_nodal_values[:,0] - sag[:,0]
def_x2= u_nodal_values[:,1] - sag[:,1]
for i in range(0,441,1):
    plt.plot(u_nodal_values[i][0],u_nodal_values[i][1],"o")
plt.show()    
# Showing Results
x1, x2, q, r, h, t, A, B = u.split()
# =============================================================================
# # =============================================================================
# # phi =  project( ( (d(x2)) / (1+c(x1)) ) + ( (g(x1)) / (1+s(x2)) ), Q ) 
# # =============================================================================
# # =============================================================================
# phi = project( abs(   d(x2)-( (d(x2)**3)/3 )+ ( (d(x2)**3)/5 ) )+ abs(   g(x1)-( (g(x1)**3)/3 )+ ( (g(x1)**3)/5 ) ) , Q )
# # =============================================================================
# # ( d(x2)-( (d(x2)**3)/3)+(d(x2)**5)/5)      , Q )
# # =============================================================================
# # =============================================================================
# # =============================================================================
# # phi =  project( ( ( ( r )*( 1+c(x1) ) - d(x2)*q  ) / ( (d(x2)**2)+( (1+(c(x1))) )**2)  ) , Q ) 
# =============================================================================
# =============================================================================
plot(x1, title="x1 plot" , mode= "color")
plt.show()
plot(x2, title="x2 plot", mode='color')
plt.show()


vtkfile_x1 = File('Bi-Directional_Fiber/x1.pvd')
vtkfile_x2 = File('Bi-Directional_Fiber/x2.pvd')
vtkfile_phi = File('Bi-Directional_Fiber/phi.pvd')

vtkfile_x1 << (x1)
vtkfile_x2 << (x2) 
vtkfile_phi << (phi) 