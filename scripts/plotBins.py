import numpy as np
import scipy.special as scispec
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import os
import pickle
import argparse
from scipy.spatial.transform import Rotation as R
from orangenetarch import OrangeNet8
import gzip
import torch

min_points = [(-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -0.5, -0.25, -0.5, -0.1, -0.1)]
max_points = [(1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.0, 0.5, 0.25, 0.5, 0.1, 0.1)]
          

def probsfromlogits(state):
  print(state.shape)
  out_list = []
  num_points = 3
  for ii in range(num_points):
    temp_list = []
    for jj in range(3):
      coord_data = state[ii,jj,:]
      prob_data = np.array(scispec.softmax(coord_data))
      temp_list.append(prob_data)
    temp_list = np.array(temp_list)
    out_list.append(temp_list)
  #maxes = np.argmax(state,axis=1)
  #print(out_list)
  return np.array(out_list)
    
      
def xyzfrombins(state,pt):
  out_list = []
  for ii in range(3):
      c = state[ii]*(max_points[pt][ii] - min_points[pt][ii])/100.0 + min_points[pt][ii]
      out_list.append(c)
  return np.array(out_list)

def plotOrigin(fig,ax):
    ax.plot3D((0,0.1),(0,0),(0,0),color="red")
    ax.plot3D((0,0),(0,0.1),(0,0),color="green")
    ax.plot3D((0,0),(0,0),(0,0.1),color="blue")

def plotPoint(ax,p,c=None,m=None):
    ax.plot3D([p[0]],[p[1]],[p[2]],color=c,marker=m)

def plotTruth(ax,p0,R0,data,pt):#Check truth format
    #color_list = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
    color_list = [(0,0,0),(0,0,0),(0,0,0),(1,1,0),(1,0,1),(0,1,1)]
    global_p = p0 + R0.apply(data[0:3])
    plotPoint(ax,global_p,c=color_list[pt],m="+")

def plotProbs(ax,p0,R0,probs,pt):#Check results format
    color_list = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
    prob_threshold = 1e-3
    for xx in range(len(probs[0,:])):
        if probs[0,xx] > prob_threshold:
            for yy in range(len(probs[1,:])):
                if probs[1,yy] > prob_threshold:
                  for zz in range(len(probs[2,:])):
                    if probs[2,zz] > prob_threshold:
                      localprob = probs[0,xx]*probs[1,yy]*probs[2,zz]
                      if localprob > prob_threshold:
                        localprob *= 10
                        if localprob > 1:
                          localprob = 1
                        xyz = xyzfrombins(np.array([xx,yy,zz]),pt)
                        xyz = p0 + R0.apply(xyz)
                        plotPoint(ax,xyz,c=(color_list[pt]+(localprob,)),m='o')

if __name__ == "__main__":
  ### model params
  capacity = 1.0
  num_images = 1
  num_pts = 3
  bins = 100
  h = 380
  w = 640
  ###
  parser = argparse.ArgumentParser()
  #parser.add_argument('fname', help='input file')
  #parser.add_argument('--coord_type', help='pickle file')
  parser.add_argument('--bin', default=1, help='use classification')
  parser.add_argument('--last', default=True,help='only show the last epoch')
  args = parser.parse_args()
  #TODO: load in file pair here
  trial = "trial0_0.28"
  temp_idx = 300
  image_fn = "/home/gabe/ws/ros_ws/src/orange_picking/data/Run24_np/" + trial + "/image" + str(temp_idx) + ".npy"
  image_data = np.load(image_fn)
  image_data = np.transpose(image_data,[2,0,1])
  f = gzip.GzipFile("/home/gabe/ws/ros_ws/src/orange_picking/data/Run24_seg/"+trial+'/image'+str(temp_idx)+'.npy.gz', 'r')
  mean_seg = np.load("/home/gabe/ws/ros_ws/src/orange_picking/data/mean_imgv2_data_seg_Run24.npy")
  temp_seg = np.load(f) - mean_seg.copy() # - mean_seg
  temp_seg = np.transpose(temp_seg, [2,0,1])
  image_data = np.concatenate((image_data,temp_seg),axis=0)
  image_data = image_data.reshape((1,5,h,w))
  image_tensor = torch.tensor(image_data)
  with open('/home/gabe/ws/ros_ws/src/orange_picking/data/Run24/'+trial +'/trajdata.pickle','rb') as f:
    data = pickle.load(f)#,encoding='latin1')
    if len(data) == 2:
      data = data[0]
    traj_list = data
    print(len(traj_list))
  with open('/home/gabe/ws/ros_ws/src/orange_picking/data/Run24/'+trial +"/metadata.pickle",'rb') as data_f:
    data = pickle.load(data_f)#, encoding='latin1')
    N = data['N'] #run_data[trial_dir]
    tf = data['tf']
    hz = float(N)/tf
  
  idx = temp_idx
  p0 = np.array(traj_list[idx][0])
  R0 = np.array(traj_list[idx][1])
  truth_points = []

  indices = np.floor(np.add(np.array([1, 2, 3]) * hz, idx)).astype(int)

  for i in range(len(indices)):
      if (indices[i] >= len(traj_list)):
          if i == 0:
              indices[i] = idx
          else:
              indices[i] = indices[i-1]

  point_list = [p0]
  rot_list = [R0]

  for x, ii in enumerate(indices):
    if (ii < idx):
            print(idx, ii)
    p = np.array(traj_list[ii][0])
    Ri = np.array(traj_list[ii][1])
    point_list.append(p)
    rot_list.append(Ri)

  p0 = np.array(point_list[0])
  R0 = np.array(rot_list[0])

  relative_pose = True

  for ii in range(1,len(point_list)):
      if (relative_pose):
        prev_p = np.array(point_list[ii-1])
        prev_R = np.array(rot_list[ii-1])
      else:
        prev_p = p0
        prev_R = R0
      p = np.array(point_list[ii])
      Ri = np.array(rot_list[ii])
      p = list(np.matmul(prev_R.T,p-prev_p))
      Ri = np.matmul(prev_R.T,Ri)
      Ri_zyx = list(R.from_dcm(Ri).as_euler('ZYX'))
      p.extend(Ri_zyx)
      p = np.array(p)
      truth_points.append(p)

  truth_points = np.array(truth_points)

  #load in model or pre-run results here
  model = OrangeNet8(capacity,num_images,num_pts,bins,min_points,max_points,n_outputs=6,real_test=False,retrain_off="",input=1, num_channels=5,real=False)
  load = "/home/gabe/ws/ros_ws/src/orange_picking/model/logs/variable_log/2021-02-28_20-47-08/model11.pth.tar"
  gpu = 1

  if os.path.isfile(load):
    if not gpu is None:
      checkpoint = torch.load(load,map_location=torch.device('cuda'))
    else:
      checkpoint = torch.load(load)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Loaded Model: ",load)
  else:
    print("No checkpoint found at: ", load)
    exit(0)

  if not gpu is None:
    device = torch.device('cuda:'+str(gpu))
    model = model.to(device)
  else:
    device = torch.device('cpu')
    model = model.to(device)

  image_tensor = image_tensor.to(device,dtype=torch.float)
  image_data = torch.tensor(image_tensor)
  results = model(image_data)
  results = results.to('cpu')
  results = results.view(-1,6,3,100).detach().numpy()
  probs = probsfromlogits(results[0,:,:,:])
  #Plot truth and probabilities
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  plotOrigin(fig,ax)
  p0 = (0,0,0)
  R0 = R.from_euler('ZYX', (0,0,0))
  for pt in range(3):
      plotTruth(ax,p0,R0,truth_points[pt],pt)#Check truth format
      plotProbs(ax,p0,R0,probs[:,pt,:],pt)#Check results format
      p0 = truth_points[pt][0:3]
      R0 = R.from_euler('ZYX',truth_points[pt][3:6])
  x_lim = ax.get_xlim()
  x_dist = x_lim[1] - x_lim[0]
  y_lim = ax.get_ylim()
  y_dist = y_lim[1] - y_lim[0]
  z_lim = ax.get_zlim()
  z_dist = z_lim[1] - z_lim[0]
  max_dist = max((x_dist,y_dist,z_dist))
  x_avg = sum(x_lim)/2
  y_avg = sum(y_lim)/2
  z_avg = sum(z_lim)/2
  ax.set_xlim3d(x_avg-(max_dist/2),x_avg+(max_dist/2))
  ax.set_ylim3d(y_avg-(max_dist/2),y_avg+(max_dist/2))
  ax.set_zlim3d(z_avg-(max_dist/2),z_avg+(max_dist/2))
  #ax.set_ylim3d(min_points[1],max_points[1])
  #ax.set_zlim3d(min_points[2],max_points[2])
  plt.show()
