import numpy as np
import sys
import os
import gcophrotor
from parsetrajfile import *

N = 500
tf = 15
epochs = 300
stiffness = 500
stiff_mult = 2.0
q = (0.2,0.2,0,#rotation log
     0,0,0,#position
     1,1,1,#rotation rate
     20,20,20)#velocity
qf = (10,10,10,#rotation log
     50,50,50,#position
     0.5,0.5,0.5,#rotation rate
     5,5,5)#velocity
r = (.1,.1,.1,1)
yaw_g = 10

p0 = (2.76422811, 0.89121437, 1.25966752)
R0 = (-0.96938357, -0.19929622, -0.14344512,  0.19339973, -0.97962837,  0.0540814, -0.15130113,  0.02468337,  0.98817949)
orange_pos = (-0.383630856872, -0.2572642564775, 0.681550323963)
yawf = 0.9585624146196554 
tree_pos = (-0.19192045927, 0.0157305616885, 0.0) 
tree_R = 0.3335851424961914 
tree_H = 1.8


ref_traj = gcophrotor.trajgen_R(N,tf,epochs,p0,R0,orange_pos,yawf,tree_pos,tree_R,tree_H,q,qf,r,yaw_g,0,0,stiffness,stiff_mult)
#Save off points in local frame
pts_per_sec = float(N)/tf
#print(pts_per_sec)
indices = np.floor(np.array([1, 2, 3])*pts_per_sec).astype(int)
#print(indices)
point_list = []
p0_np = np.array(p0)
R0_np = np.array(ref_traj[0][1])
for ii in indices:
  point = np.array(ref_traj[ii][0])
  #print('PreTransform ' + str(point))
  point = point - p0_np
  #print('Subtracted ' + str(point))
  point = np.matmul(R0_np.T,point)
  #print('Output: ' + str(point))
  point_list.append(point)

ts = np.linspace(0,tf,N+1)
make_plots(ts,ref_traj,orange_pos,tree_pos,tree_R,tree_H,None, point_list)
