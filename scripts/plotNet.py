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
  num_points = state.shape[0]
  for ii in range(num_points):
    temp_list = []
    for jj in range(3):
      coord_data = state[ii,jj,:]
      prob_data = scispec.softmax(coord_data)
      temp_list.append(prob_data)
    out_list.append(temp_list)
  #maxes = np.argmax(state,axis=1)
  #print(out_list)
  return np.array(out_list)
    
      
def xyzfrombins(state,data):
  #print(state.shape)
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
  prob_threshold = 1e-3
  parser = argparse.ArgumentParser()
  parser.add_argument('fname', help='pickle file')
  #parser.add_argument('--coord_type', help='pickle file')
  parser.add_argument('--bin', default=1, help='pickle file')
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
      #truth = xyzfrombins(np.array(data['truth'])[:,ii,:], data)
      truth = np.array(data['truth'])[:,ii,:,:]
    else:
      truth = data['truth'][ii]
    #if args.coord_type is not None:
    #  truth = backProject(truth,args.coord_type, data['foc_l'])
    num_epochs = len(data['data'][ii])
    step = int(num_epochs/5)
    if step < 2:
      epoch_range = range(0,num_epochs)
    else:
      epoch_range = range(0,num_epochs,step)
    for jj in epoch_range:#Iterate over epochs
      print('Point: ' + str(ii+1) + ' of ' + str(len(data['idx'])) + ' Iteration: ' + str(jj+1) + ' of ' + str(len(data['data'][ii])))
      truth_bin_nums = np.argmax(truth,axis=2)
      print('True Value(s): ' + str(truth_bin_nums.T))
      num_samples = len(data['data'][ii])
      output = data['data'][ii][jj]
      #if args.coord_type is not None:
      #  output = backProject(output,args.coord_type, data['foc_l'])
      if args.bin is not None:
        probs = probsfromlogits(output)
        print(probs)
        flag = False
        fig = None
        ax = None
        for pt in range(probs.shape[0]):
          for xx in range(len(probs[pt,0,:])):
            if probs[pt,0,xx] > prob_threshold:
              for yy in range(len(probs[pt,1,:])):
                if probs[pt,1,yy] > prob_threshold:
                  for zz in range(len(probs[pt,2,:])):
                    if probs[pt,2,zz] > prob_threshold:
                      localprob = probs[pt,0,xx]*probs[pt,1,yy]*probs[pt,2,zz]
                      if localprob > prob_threshold:
                        if fig is None:
                          fig = plt.figure()
                          ax = plt.axes(projection='3d')
                          flag = True
                        xyz = xyzfrombins(np.array([xx,yy,zz]),data)
                        ax.plot3D([xyz[0]],[xyz[1]],[xyz[2]],color=(color_list[ii]+(localprob,)),marker='o')
                        print([xx,yy,zz,localprob])
        if flag:
          #for pt in range(probs.shape[0]):
          x_bin_num = truth_bin_nums[0,:]
          y_bin_num = truth_bin_nums[1,:]
          z_bin_num = truth_bin_nums[2,:]
          true_xyz = xyzfrombins(np.array([x_bin_num,y_bin_num,z_bin_num]),data)
          ax.plot3D([true_xyz[0]],[true_xyz[1]],[true_xyz[2]],'g+')
          ax.set_xlim3d(data['min'][0],data['max'][0])
          ax.set_ylim3d(data['min'][1],data['max'][1])
          ax.set_zlim3d(data['min'][2],data['max'][2])
          plt.show()
        else:
          print('No points above probability threshold of: ' + str(prob_threshold))

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
