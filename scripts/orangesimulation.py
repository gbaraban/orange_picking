import numpy as np
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img
from scipy.spatial.transform import Rotation as R
from orangenetarch import *
import torch
import torch.nn as nn
import argparse
import os

def gcopVecToUnity(v):
  temp = np.array((v[0],v[2],v[1]))
  return temp

def makeCamAct(x):
  #Set up camera action
  cameraPos = x[0:3]  
  r = R.from_euler('zyx',x[3:6])
  euler = r.as_euler(seq = 'zxy',degrees = True) #using weird unity sequence
  unityEuler = (euler[1],-euler[0],euler[2]) 
  print("euler: ", unityEuler)
  cameraAction = np.hstack((gcopVecToUnity(cameraPos),unityEuler,1))
  return np.array([cameraAction])

def setUpEnv(env, x0, treePos, orangePos, treeScale = 0.125, orangeScale = 0.07):
    env.reset()
    camAct = makeCamAct(x0)
    treeAct = np.array([np.hstack((treePos,1))])
    orangeAct = np.array([np.hstack((orangePos,1))])
    names = env.get_behavior_names()
    camName = ""
    for n in names:
        print(n)
        if "Tree" in n:
            env.set_actions(n,treeAct)
            print("tree set")
            continue
        if "Orange" in n:
            env.set_actions(n,orangeAct)
            print("orange set")
            continue
        if "Cam" in n:
            env.set_actions(n,camAct)
            print("orange set")
            camName = n
            continue
    return camName

def unity_image(env,act,cam_name):
  env.set_actions(cam_name,act)
  env.step()
  (ds,ts) = env.get_steps(cam_name)
  obs = ds.obs[0][0,:,:,:]
  return obs

def save_image_array(image_in,path,name):
  im_np = (image_in*255).astype('uint8')
  image = img.fromarray(im_np)
  image.save(path + name + '.png')

def sys_f_linear(x,goal,dt,goal_time=1):
  goal_pt = goal[0]
  dx = np.array(goal_pt - x[0:3])
  yaw = np.arctan2(dx[1],dx[0])
  pitch = np.arctan(dx[2],dx[0])
  roll = 0
  time_frac = dt/goal_time
  new_pos = x[0:3] + time_frac*dx
  new_x = np.hstack((new_pos,yaw,pitch,roll))
  return new_x

def run_sim(sys_f,env,model,x0,orange,tree,eps=0.1, max_steps=99,dt=0.1,save_path = None):
  x_list = []
  camName = setUpEnv(env,x0,orange,tree)
  if camName is "":
      print("camName not set")
      return
  dist = np.linalg.norm(x0[0:3] - orange)
  x = x0
  for step in range(max_steps):
    x_list.append(x)
    if dist < eps:
      return 0
    #Get Image
    camAct = makeCamAct(x)
    image_arr = unity_image(env,camAct,camName)
    image_tensor = torch.tensor(image_arr)#TODO:Add cuda stuff here
    #Optionally save image
    if save_path is not None:
      save_image_array(image_arr,save_path,"sim_image"+str(step))
    #Calculate new goal
    logits = model(image_arr)
    logits = np.array(logits.view(1,3,model.num_points,model.bins))
    predict = np.argmax(logits,axis=3)
    goal = []
    for pt in range(model.num_points):
      point = []
      for coord in range(3):
        bin_size = (model.max[pt][coord] - model.min[pt][coord])/model.bins
        point.append(model.min[pt][coord] + predict[0,coord,pt])
      goal.append(np.array(point))
    goal = np.array(goal)
    #Calculate new x
    x = sys_f(x,goal,dt)
  return 1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('load', help='model to load')
  parser.add_argument('--gpu', help='gpu to use')
  parser.add_argument('--num_images', type=int, default=1, help='number of input images')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
  parser.add_argument('--capacity', type=float, default=1, help='network capacity')
  parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
  parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
  parser.add_argument('--bins', type=int, default=30, help='number of bins per coordinate')
  parser.add_argument('--iters', type=int, default=30, help='number of simulations')
  parser.add_argument('--env', type=str, default="unity/env_v3", help='unity filename')
  args = parser.parse_args()
  args.min = [(0,-0.5,-0.1),(0,-1,-0.15),(0,-1.5,-0.2),(0,-2,-0.3),(0,-3,-0.5)]
  args.max = [(1,0.5,0.1),(2,1,0.15),(4,1.5,0.2),(6,2,0.3),(7,0.3,0.5)]
  #Load model
  model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max)
  if os.path.isfile(args.load):
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint)
    print("Loaded Model: ",args.load)
  else:
    print("No checkpoint found at: ", args.load)
    return
  #Create environment
  env = UnityEnvironment(file_name=args.env,seed=0)
  #env.reset()
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
  #Iterate simulations
  run_num = 0
  globalfolder = 'data/Sim' + str(run_num) + '/'
  while os.path.exists(globalfolder):
    run_num += 1
    globalfolder = 'data/Run' + str(run_num) + '/'
  trial_num = -1
  while trial_num < args.iters or args.iters is -1:
    trial_num += 1
    print('Trial Number ',trial_num)
    #Filenames
    foldername = "trial" + str(trial_num) + "/"
    os.makedirs(globalfolder + foldername)
    #randomize environment
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
    x0_i = np.hstack((x0_i,yaw0_i,0,0))
    err_code = run_sim(sys_f,env,model,x0_i,orangePos_i,treePos_i)
    if err_code is not 0:
      print('simulation did not converge.  code is: ', err_code) 
  env.close()

if __name__ == '__main__':
  main()
