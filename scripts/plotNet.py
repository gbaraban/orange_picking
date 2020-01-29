import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import os
import pickle
import argparse

def backProject(state,coord_type, f):
  xs = state[0::3]
  ys = state[1::3]
  zs = state[2::3]
  out_state = []
  for x,y,z in zip(xs,ys,zs):
    y = y*(x/f)
    z = z*(x/f)
    if coord_type is "simple":
      x = x
    elif coord_type is "inverse":
      x = f/x
    elif coord_type is "xbyf":
      x = x*f
    out_state.append(x)
    out_state.append(y)
    out_state.append(z)
  print(out_state)
  return np.array(out_state)
      


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('fname', help='pickle file')
  parser.add_argument('--coord_type', help='pickle file')
  args = parser.parse_args()
  print(args.coord_type)
  with open(args.fname,'rb') as f:
    data = pickle.load(f)
  #fig_list = []
  #ax_list = []
  fig_list = plt.figure()
  ax_list = plt.axes(projection='3d')
  for ii in data['idx']:
    #fig_list.append(plt.figure())
    #ax_list.append(plt.axes(projection='3d'))
    truth = data['truth'][ii]
    if args.coord_type is not None:
      truth = backProject(truth,args.coord_type, data['foc_l'])
    ax_list.plot3D(truth[0::3],truth[1::3],truth[2::3],'g+')
    for jj in range(len(data['data'][ii])):
      num_samples = len(data['data'][ii])
      output = data['data'][ii][jj]
      if args.coord_type is not None:
        output = backProject(output,args.coord_type, data['foc_l'])
      c = float(jj+1)/num_samples
      ax_list.plot3D(output[0::3],output[1::3],output[2::3], color=(1,0,0,c))
      #l,r = ax_list.xlim()
      #ax_list.set_xlim(0,4)
      #ax_list.set_ylim(-2,2)
      #ax_list.set_zlim(-2,2)
      #l,r = ax_list.ylim()
      #ax_list.ylim(min(0,l),max(0,r))
      #l,r = ax_list.zlim()
      #zlim(min(0,l),max(0,r))

  plt.show()
