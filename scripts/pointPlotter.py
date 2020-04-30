import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pickle

bag_dir = 'data/bag_v3/'
bag_list = os.listdir(bag_dir)

data_list = []
points_list = []
for bag in bag_list:
  if os.path.isdir(bag_dir+'/'+bag): 
    bag_num = bag[3:]
    with open(bag_dir+'/'+bag+'/data.pickle','rb') as f:
      data = pickle.load(f)
      data_list.append(data)

for ii in range(len(data_list)):
  for jj in data_list[ii]:
    points = data_list[ii][jj]
    points_list.append(points)
  
for ii in range(0,3):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  #print(data[ii])
  data_ii = [points_list[temp][ii] for temp in range(len(points_list))]
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
#  ax.+plot3D(x,y,z,marker='+')#data[ii][:][0],data[ii][:][1],data[ii][:][2], marker = '+')
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  ax.set_zlabel('z (m)')
  plt.show()



