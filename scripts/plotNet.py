import numpy as np
import scipy.special as scispec
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
  print(coord_type)
  print(out_state)
  return np.array(out_state)

def probsfromlogits(state):
  out_list = []
  for ii in range(3):
    coord_data = state[ii,:]
    prob_data = scispec.softmax(coord_data)
    out_list.append(prob_data)
  #maxes = np.argmax(state,axis=1)
  print(out_list)
  return np.array(out_list)
    
      
def xyzfrombins(state,data):
  state = state.reshape(3,-1)
  xs = state[0,:]
  ys = state[1,:]
  zs = state[2,:]
  out_list = []
  for x,y,z in zip(xs,ys,zs):
    o = np.array((x,y,z))*(data['max'] - data['min'])/float(data['bins']) + data['min']
    out_list.append(o[0])
    out_list.append(o[1])
    out_list.append(o[2])
  return np.array(out_list)
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('fname', help='pickle file')
  parser.add_argument('--coord_type', help='pickle file')
  parser.add_argument('--bin', help='pickle file')
  args = parser.parse_args()
  with open(args.fname,'rb') as f:
    data = pickle.load(f)
  fig_list = []
  ax_list = []
  #fig_list = plt.figure()
  #ax_list = plt.axes(projection='3d')
  color_list = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
  for ii in data['idx']:#Iterate over each example
    if args.bin is not None:
      truth = xyzfrombins(np.array(data['truth'])[:,ii,:], data)
    else:
      truth = data['truth'][ii]
    #if args.coord_type is not None:
    #  truth = backProject(truth,args.coord_type, data['foc_l'])
    for jj in range(len(data['data'][ii])):#Iterate over epochs
      print('Point: ' + str(ii) + ' of ' + str(len(data['idx'])) + ' Iteration: ' + str(jj) + ' of ' + str(len(data['data'][ii])))
      num_samples = len(data['data'][ii])
      output = data['data'][ii][jj]
      #if args.coord_type is not None:
      #  output = backProject(output,args.coord_type, data['foc_l'])
      if args.bin is not None:
        probs = probsfromlogits(output)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(truth[0::3],truth[1::3],truth[2::3],'g+')
        for xx in range(len(probs[0,:])):
          if probs[0,xx] > 0.1:
            for yy in range(len(probs[1,:])):
              if probs[1,yy] > 0.1:
                for zz in range(len(probs[2,:])):
                  if probs[2,zz] > 0.1:
                    localprob = probs[0,xx]*probs[1,yy]*probs[2,zz]
                   # if localprob > 0.5:
                    xyz = xyzfrombins(np.array([xx,yy,zz]),data)
                    ax.plot3D([xyz[0]],[xyz[1]],[xyz[2]],color=(color_list[ii]+(localprob,)),marker='o')
                    print([xx,yy,zz,localprob])
        plt.show()
      else:
        c = color_list[ii] + (float(jj+1)/num_samples,)
        ax_list.plot3D(output[0::3],output[1::3],output[2::3], color=c, marker='o')
      #l,r = ax_list.xlim()
      #ax_list.set_xlim(0,4)
      #ax_list.set_ylim(-2,2)
      #ax_list.set_zlim(-2,2)
      #l,r = ax_list.ylim()
      #ax_list.ylim(min(0,l),max(0,r))
      #l,r = ax_list.zlim()
      #zlim(min(0,l),max(0,r))

  plt.show()
