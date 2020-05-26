import gcophrotor
import matplotlib.pyplot as plt
import PIL.Image as img
import numpy as np
import sys
import os
import pickle
from mlagents_envs.environment import UnityEnvironment
from scipy.spatial.transform import Rotation as R
import argparse
from parsetrajfile import *

#import tracemalloc

def gcopVecToUnity(v):
  temp = np.array((v[0],v[2],v[1]))
  return temp

def makeCamAct(action, brainNames, cameraPos, cameraRot):
  #Set up camera action
  #TODO: convert into unity rotation space (maybe use euler??)
  #worldoffset = R.from_dcm(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
  r = R.from_dcm(cameraRot)
  euler = r.as_euler(seq = 'zxy',degrees = True) #using weird unity sequence
  unityEuler = (euler[1],-euler[0],euler[2]) 
  cameraAction = np.hstack((gcopVecToUnity(cameraPos),unityEuler))
  action[brainNames[0]] = cameraAction
  return action


def setUpEnv(brainNames, cameraPos, cameraRot, treePos, orangePos, treeScale = 0.125, orangeScale = 0.07):
  action = makeCamAct(dict(), brainNames,cameraPos,cameraRot)
  treeScale = treeScale*np.array([1.0,1.0,1.0])
  orangeScale = orangeScale*np.array([1.0,1.0,1.0])
  #Set up tree action
  treeAction = np.hstack((gcopVecToUnity(treePos)))#,gcopVecToUnity(treeScale)))
  #Set up orange action
  orangeAction = np.hstack((gcopVecToUnity(orangePos)))#,gcopVecToUnity(orangeScale)))
  #Add to the action dict
  action[brainNames[1]] = treeAction
  action[brainNames[2]] = orangeAction
  return action

#Parse stuff
parser = argparse.ArgumentParser()
parser.add_argument('--env', help='env file')
parser.add_argument('--loop', type=bool, help='make infinite data')
parser.add_argument('--plotonly', type=bool, help='skip image generation')
args = parser.parse_args()
#Bridge to Unity
#tracemalloc.start()
if (args.plotonly == None):
  env_name = 'unity/env_v1'
  train_mode = True
  env = UnityEnvironment(file_name=env_name)
  brainNames = env.brain_names
  env.reset(train_mode=train_mode)

#Global Parameters
N = 500
tf = 15
epochs = 200
stiffness = 1000
stiff_mult = 2.0
np.random.seed(0)
q = (0.2,0.2,0,#rotation log
     0,0,0,#position
     1,1,1,#rotation rate
     5,5,5)#velocity
qf = (10,10,10,#rotation log
     50,50,50,#position
     0.5,0.5,0.5,#rotation rate
     5,5,5)#velocity
r = (.1,.1,.1,1)
yaw_g = 10
rp_g = 0
direction_gain = 0 
#Baseline Positions
x0 = np.array((-10,0,1))
xmult = np.array((2,2,0.5))
yaw0 = 0
ymult = np.pi
cameraRot = np.array([[1,0,0],[0,1,0],[0,0,1]])
treePos = np.array((0,0,0))
treemult = np.array((0.5,0.5,0))
treeHeight = 1.6
orangePos = np.array((0,0,0.5))
orangeR = 0.6
orangeRmult = 0.3
orangeHmult = 0.5
run_num = 0
globalfolder = 'data/Run' + str(run_num) + '/'
while os.path.exists(globalfolder):
  run_num += 1
  globalfolder = 'data/Run' + str(run_num) + '/'

#presnap = tracemalloc.take_snapshot()
exp = -1
trials = 2
while (exp < trials) or args.loop:
  exp += 1
  print('Trial Number ' + str(exp))
  #Filenames
  foldername = "trial" + str(exp) + "/"
  os.makedirs(globalfolder + foldername)
  overviewname = "traj_plot" + str(exp)
  picturename = "image"
  suffix = ".png"
  if (args.env is not None):
    #Load in Env Data
    with open(args.env,'rb') as f:
      loadData = pickle.load(f)
      print(loadData)
      x0_i = loadData['x0']
      treePos_i = loadData['cyl_o']
      orangePos_i = loadData['xf']
      yaw0_i = loadData['yaw0']
  else:
    #Random Offset
    xoffset = 2*xmult*(np.random.rand(3) - 0.5)
    x0_i = x0 + xoffset
    yaw0_i = yaw0 + 2*ymult*(np.random.rand(1) - 0.5)
    treeoffset = 2*treemult*(np.random.rand(3) - 0.5)
    treePos_i = treePos + treeoffset
    theta = 2*np.pi*np.random.random_sample()
    R_i = orangeR + orangeRmult*np.random.random_sample()# + 0.2
    orangeoffset = np.array((R_i*np.cos(theta), R_i*np.sin(theta),
                             orangePos[2] + orangeHmult*np.random.random_sample()))
    orangePos_i = treePos_i + orangeoffset

  #Set up Unity
  if (args.plotonly == None):
    act = setUpEnv(brainNames,x0_i,cameraRot,treePos_i,orangePos_i)
    env.step(act)
  #Call GCOP
  yawf = np.arctan2(treePos_i[1]-orangePos_i[1],treePos_i[0]-orangePos_i[0])
  with open(globalfolder + foldername + 'metadata.pickle','wb') as f:
    metadata = {'N':N,'tf':tf,'epochs':epochs,'stiffness':stiffness,'stiff_mult':stiff_mult,
                'x0':x0_i, 'yaw0':yaw0_i,'xf':orangePos_i,'yawf':yawf,
                'cyl_o':treePos_i,'cyl_r':orangeR, 'h':treeHeight,
                'q':q, 'qf':qf, 'r':r, 'yaw_gain':yaw_g, 'rp_gain':rp_g, 'dir_gain':direction_gain}
    pickle.dump(metadata,f,pickle.HIGHEST_PROTOCOL)
  ref_traj = gcophrotor.trajgen(N,tf,epochs,tuple(x0_i),yaw0_i,tuple(orangePos_i),yawf,
                                tuple(treePos_i),orangeR,treeHeight,
                                tuple(q),tuple(qf),tuple(r),yaw_g,rp_g,direction_gain,
                                stiffness,stiff_mult)
  
  #Plot Trajectories
  ts = np.linspace(0,tf,N+1)
  make_plots(ts,ref_traj,orangePos_i,treePos_i,orangeR, treeHeight, globalfolder + foldername)

  #save ref_traj
  with open(globalfolder + foldername + 'trajdata.pickle','wb') as f:
    pickle.dump(ref_traj,f,pickle.HIGHEST_PROTOCOL)

  #Step through trajectory
  if (args.plotonly == None):
    ctr = 0
    for state in ref_traj:
      #print('State ' + str(ctr))
      #Unpack state
      pos = state[0]
      rot = np.array(state[1])
      #rot = rot.T
      #print(pos)
      act = makeCamAct(act, brainNames,pos,rot)
      env_info = env.step(act)[brainNames[0]]
      obs = np.array(env_info.visual_observations)
      im_np = (obs[0,0,:,:,:]*255).astype('uint8')
      image = img.fromarray(im_np)
      image.save(globalfolder + foldername + picturename + str(ctr) + suffix)
      ctr += 1
  if(args.env):
      break
#postsnap = tracemalloc.take_snapshot()
if (args.plotonly == None):
  env.close()
#stats = postsnap.compare_to(presnap, 'lineno')
#print("Stat comparison")
#for stat in stats[:10]:
#  print(stat)
