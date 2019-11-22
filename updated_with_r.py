from __future__ import print_function
from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.tri as tri
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
initial_mesh_coordinate = mesh.coordinates()
# Initializing material parameters
mu = Constant(1)
E1 = Constant(150/mu)
E2 = Constant(100/mu)
c1 = Constant(200/mu)
c2 = Constant(40/mu)
p11 = Constant(200)




# Defining test and trial functions in splitted form
element = VectorElement('P', triangle, 1, dim=12)
V = FunctionSpace(mesh, element)
Q = FunctionSpace(mesh, "DG", 0)

plot(mesh)
plt.show()

v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9, v_10, v_11, v_12 = TestFunctions(V)
u = Function(V)
x1, x2, q, r, h, t, A, B, C, S, G, D = split(u)

x, y = SpatialCoordinate(mesh)

# Defining Derivatives
#x1dx = project(x1.dx(0), Q)
#x1dy = project(x1.dx(1), Q)
#qdx = project(q.dx(0), Q)
#v_1dx = project(v_1.dx(0), Q)
#x2dx = project(x2.dx(0), Q)
#x2dy = project(x2.dx(1), Q)

# Defining non-linear terms...

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
class sag(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0)
    
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
left().mark(facets, 1)
right().mark(facets, 2)
top().mark(facets, 3)
bottom().mark(facets, 4)
middle().mark(facets, 5)
sag().mark(facets, 6)
ds = Measure("ds", subdomain_data=facets)

##modification added for left and right edge 
u_R = Expression('x[1]', degree=1)
u_R1 = Expression('x[0]', degree=1)
bc = [DirichletBC(V.sub(0),Constant(0), facets, 5),#(x1 is 0 in the middle of fiber(symmetric))
      DirichletBC(V.sub(1), Constant(0) , facets, 6),#(x2 is 0 on the middle line(symmetric))
      DirichletBC(V.sub(1),u_R , facets, 1),#(x2 remains the same on the left edge)
      DirichletBC(V.sub(1), u_R , facets, 2),#(x2 remains the same on the right edge)
      DirichletBC(V.sub(3), Constant(0) , facets, 1),#(r remains 0 the same on left right edge)
      DirichletBC(V.sub(3), Constant(0) , facets, 2)#(r remains 0 the same on the right edge)
      
# =============================================================================
# =============================================================================
#       DirichletBC(V.sub(5),Constant(0) , facets, 1),#completely fixed edge (x2 remains the same on the left edge)
#       DirichletBC(V.sub(5),Constant(0) , facets, 2)
# # =============================================================================
# # =============================================================================
# #       DirichletBC(V.sub(11),Constant(0) , facets, 1),#completely fixed edge (x2 remains the same on the left edge)
# #       DirichletBC(V.sub(11),Constant(0) , facets, 2),
# =============================================================================
# =============================================================================
      ]##completely fixed edge (x2 remains the same on the right edge)


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
# =============================================================================
# =============================================================================
# =============================================================================
f_9 = project((p11)/(2*mu + E1),Q)  
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # =============================================================================
# f_9 = project(sqrt( ( (  -(2*mu )*t+p11   )/(E1/2) )+1-(t**2) ),Q)
# =============================================================================
# =============================================================================

# =============================================================================
# =============================================================================
# =============================================================================
# f_9 = project((p11)/(2*mu + E1),Q)  
# f_10= project(0.00001,Q)  
# =============================================================================
#f_10 = project((p22)/(2*mu + E1),Q) 
# =============================================================================
# Define variational problem
F = (2*mu*(q+h)*v_1-A*S*v_1+B*D*v_1+c1*q.dx(0)*v_1.dx(0) 
    + c2*h.dx(1)*v_1.dx(1)-0.5*E1*q*v_1-0.5*E2*h*v_1 
    +0.5*E1*(3*q*C*C+q*D*D+2*r*C*D)*v_1  
    +0.5*E2*(3*h*G*G+h*S*S+2*t*G*S)*v_1)*dx +(2*mu*(r+t)*v_2 
    +A*G*v_2-B*C*v_2+c1*r.dx(0)*v_2.dx(0)   
    +c2*t.dx(1)*v_2.dx(1)-0.5*E1*r*v_2-0.5*E2*t*v_2 
    +0.5*E1*(3*r*D*D+r*C*C+2*q*D*C)*v_2  
    +0.5*E2*(3*t*S*S+t*G*G
    +2*h*S*G)*v_2)*dx +(q*v_3+x1.dx(0)*v_3.dx(0))*dx+(r*v_4  
    +x2.dx(0)*v_4.dx(0))*dx +(h*v_5+x1.dx(1)*v_5.dx(1))*dx+(t*v_6+x2.dx(1)*v_6.dx(1))*dx +(A*v_7 
    -mu*(q+h)*v_7+c1*q.dx(0)*v_7.dx(0))*dx +(B*v_8-mu*(r+t)*v_8  
    +c1*r.dx(0)*v_8.dx(0))*dx + ( (C-x1.dx(0))*v_9*dx )+ ( (S-x2.dx(1))*v_10*dx )+ ( (G-x1.dx(1))*v_11*dx )+ ( (D-x2.dx(0))*v_12*dx )- f_1*v_1*ds - f_2*v_2*ds - f_3*v_3*ds - f_4*v_4*ds - f_5*v_5*ds - f_6*v_6*ds-f_7*v_7*ds -f_8*v_8*ds-f_9*v_3*ds(2)+f_9*v_3*ds(1)
# =============================================================================
#     -f_9*v_3*ds(2)
# =============================================================================

##PICARD Iteration
u_ = Expression(('0','0','0','0','0','0','0','0','0','0','0','0'), degree =1)
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
u_nodal_values = np.zeros((441*4,2))
j=-1
for i in range (0,5292,12):
    j=j+1
    u_nodal_values[j][0] = u.vector().get_local()[i]
    u_nodal_values[j][1] = u.vector().get_local()[i+1]
# =============================================================================
# def_x1= u_nodal_values[:,0] - initial_mesh_coordinate[:,0]
# def_x2= u_nodal_values[:,1] - initial_mesh_coordinate[:,1]
# =============================================================================
for i in range(0,441,1):
    plt.plot(u_nodal_values[i][0],u_nodal_values[i][1],"o", color='green')
# =============================================================================
# plt.axis("equal")
# =============================================================================
plt.show() 
  
# Showing Results
x1, x2, q, r, h, t, A, B, C, S, G, D = u.split()

# =============================================================================
# =============================================================================
phi = project( -abs(   C-( (C**3)/3 )+ ( (C**5)/5 ) )-abs(   S-( (S**3)/3 )+ ( (S**5)/5 ) ) , Q )
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# phi = project(  abs(r*(1+C)-(D*H) )/((D**2)+(1+C)**2) , Q )
# =============================================================================
# plt.triplot(u_nodal_values[:,0], u_nodal_values[:,1],mesh.cells())    
# plt.show()
# =============================================================================
AA = u.vector().get_local() 
triang = tri.Triangulation(u_nodal_values[:,0], u_nodal_values[:,1],mesh.cells())
fig1, ax1 = plt.subplots()
ax1.triplot(triang, 'bo-',lw=2)
ax1.set_title('triplot of Delaunay triangulation')
plt.show()

# =============================================================================
# def_x1= u_nodal_values[:,0] - initial_mesh_coordinate[:,0]
# def_x2= u_nodal_values[:,1] - initial_mesh_coordinate[:,1]
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