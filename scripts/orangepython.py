import gcophrotor
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
from mlagents.envs.environment import UnityEnvironment
from scipy.spatial.transform import Rotation as R

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
  unityEuler = (euler[1],euler[0],euler[2]) 
  cameraAction = np.hstack((gcopVecToUnity(cameraPos),euler))
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

#Bridge to Unity
#tracemalloc.start()
env_name = None
train_mode = True
env = UnityEnvironment(file_name=env_name)
brainNames = env.brain_names
env.reset(train_mode=train_mode)

#Global Parameters
N = 100
tf = 15.0
epochs = 50
stiffness = 1#10e6
stiff_mult = 2.0
np.random.seed(0)
#Baseline Positions
x0 = np.array((-10,0,1))
xmult = np.array((2,1,0.5))
cameraRot = np.array([[1,0,0],[0,1,0],[0,0,1]])
treePos = np.array((0,0,0))
treemult = np.array((0.5,0.5,0))
treeHeight = 1.6
orangePos = np.array((0,0,1))
orangeR = 1.2
orangeRmult = 0.5
orangeHmult = 0.5
run_num = 0
globalfolder = 'data/Run' + str(run_num) + '/'
while os.path.exists(globalfolder):
  run_num += 1
  globalfolder = 'data/Run' + str(run_num) + '/'

#presnap = tracemalloc.take_snapshot()
for exp in range(2):
  print('Trial Number ' + str(exp))
  #Filenames
  foldername = "trial" + str(exp) + "/"
  os.makedirs(globalfolder + foldername)
  overviewname = "traj_plot" + str(exp)
  picturename = "image"
  suffix = ".png"
  #Random Offset
  xoffset = 2*xmult*(np.random.rand(3) - 0.5)
  x0_i = x0 + xoffset
  treeoffset = 2*treemult*(np.random.rand(3) - 0.5)
  treePos_i = treePos + treeoffset
  theta = 2*np.pi*np.random.random_sample()
  R_i = orangeR + orangeRmult*np.random.random_sample() + 0.2
  #print("OrangeR: " + str(orangeR))
  #print("R_i: " + str(R_i))
  orangeoffset = np.array((R_i*np.cos(theta), R_i*np.sin(theta),
                           orangePos[2] + orangeHmult*np.random.random_sample()))
  orangePos_i = treePos_i + orangeoffset

  #Plot Environment
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  #Plot cylinder
  theta = np.linspace(0, 2*np.pi, 50)
  zs = np.linspace(treePos_i[2], treePos_i[2]+treeHeight, 50)
  thetac, zc = np.meshgrid(theta,zs)
  xc = orangeR * np.cos(thetac) + treePos_i[0]
  yc = orangeR * np.sin(thetac) + treePos_i[1]
  ax.plot_surface(xc,yc,zc)
  #Plot x0 and xf
  ax.plot3D((x0_i[0],orangePos_i[0]),(x0_i[1],orangePos_i[1]),(x0_i[2],orangePos_i[2]),'blue')
#  ax.plot3D((orangePos_i[0]),(orangePos_i[1]),(orangePos_i[2]),'green')
#  plt.show()
  #continue

  #Set up Unity
  act = setUpEnv(brainNames,x0_i,cameraRot,treePos_i,orangePos_i)
  env.step(act)
  #Call GCOP
  with open(globalfolder + foldername + 'metadata.pickl','wb') as f:
    metadata = {'N':N,'tf':tf,'epochs':epochs,'stiffness':stiffness,'stiff_mult':stiff_mult,
                'x0':x0_i, 'xf':orangePos_i,'cyl_o':treePos,'r':orangeR, 'h':treeHeight}
    pickle.dump(metadata,f,pickle.HIGHEST_PROTOCOL)
  ref_traj = gcophrotor.trajgen(N,tf,epochs,tuple(x0_i),tuple(orangePos_i),
                                tuple(treePos),orangeR,treeHeight,
                                stiffness,stiff_mult)
  
  #Plot trajectory
  xtraj = [temp[0][0] for temp in ref_traj]
  ytraj = [temp[0][1] for temp in ref_traj]
  ztraj = [temp[0][2] for temp in ref_traj]
  ax.plot3D(xtraj,ytraj,ztraj)
  fig.savefig(globalfolder + foldername + 'traj_plot' + str(exp) + '.png')
  fig.clf()
  plt.close()


  #save ref_traj
  with open(globalfolder + foldername + 'trajdata.pickle','wb') as f:
    pickle.dump(ref_traj,f,pickle.HIGHEST_PROTOCOL)

  #Step through trajectory
  ctr = 0
  for state in ref_traj:
    ctr += 1
    print('State ' + str(ctr))
    #Unpack state
    pos = state[0]
    rot = np.array(state[1])
    #rot = rot.T
    #print(pos)
    act = makeCamAct(act, brainNames,pos,rot)
    env_info = env.step(act)[brainNames[0]]
    obs = np.array(env_info.visual_observations)
    fig = plt.figure()
    ax = plt.axes()
    ax.imshow(obs[0,0,:,:,:])
    #plt.show()
    fig.savefig(globalfolder + foldername + picturename + str(ctr) + suffix)
    fig.clf()
    plt.close()
#postsnap = tracemalloc.take_snapshot()
env.close()
#stats = postsnap.compare_to(presnap, 'lineno')
#print("Stat comparison")
#for stat in stats[:10]:
#  print(stat)
