import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

f = open('points.txt','r')

data = []
for ii in range(0,5):
  data.append([])

l = f.readline()
while l:
  pt_type = int(l[4]) - int('0')
  #print(pt_type)
  pt = l[15:-2].split(" ")
  pt = pt[1:]
  pt = [float(temp) for temp in pt if temp is not '']
  #print(pt)
  data[pt_type].append(pt)
  l = f.readline()
  
for ii in range(0,5):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  #print(data[ii])
  data_ii = data[ii]
  for temp in data_ii:
    if len(temp) < 3:
      continue
    #print(temp)
    x = [temp[0]]
    y = [temp[1]]
    z = [temp[2]]
    ax.plot3D(x,y,z,marker='+')#data[ii][:][0],data[ii][:][1],data[ii][:][2], marker = '+')
#  x = [temp[0] for temp in data_ii]
#  y = [temp[1] for temp in data_ii]
#  z = [temp[2] for temp in data_ii]
#  ax.plot3D(x,y,z,marker='+')#data[ii][:][0],data[ii][:][1],data[ii][:][2], marker = '+')
  plt.show()


f.close()

