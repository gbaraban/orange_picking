import numpy as np
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img
from scipy.spatial.transform import Rotation as R
from orangenetarch import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import argparse
from plotting.parsetrajfile import *
from orangesimulation import *
import pickle
import os
import gcophrotor

class DAggerSet(Dataset):
    def __init__(self,batch_list):
        self.data = batch_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        #TODO: doublecheck format here
        value = self.data[i]
        return {'image':value[0],'points':values[1]}

def DAggerCompare(x,goals,ref_goals,cyl_o,cyl_h=1.6,cyl_r=0.6):
    #TODO: adjust gamma
    gamma = 1
    r = R.from_euler('zyx',x[3:6])
    for g in goals:
        g_trans = r.apply(g[0:3]) + x[0:3]
        if (g_trans[2] < cyl_o[2] + cyl_h) and (g_trans[2] > cyl_o[2]):
            dx = (cyl_o[0] - gtrans[0])
            dy = (cyl_o[1] - gtrans[2])
            if cyl_r*cyl_r > (dx*dx + dy*dy):
                return float('inf')
    metric = 0
    for (g,ref) in zip(goals,ref_goals):
        metric += np.linalg.norm(g[0:3]-ref[0]) #distance
        g_dir = R.from_euler('zyx',g[3:6]).apply([1,0,0])
        ref_dir = ref[1][:,0]
        metric += gamma*(1 - np.dot(g_dir,ref_dir))
    return metric

def wp_from_traj(expert_path,t,tf=15,goal_times=[1,2,3]):
    time_span = tf-t
    N = len(expert_path)
    h = float(time_span)/N
    goal_idx = [int(temp/h) for temp in goal_times]
    waypoints = []
    for idx in goal_idx:
        if idx >= N:
            waypoints.append(expert_path[N-1])
        elif idx < 0:
            waypoints.append(expert_path[0])
        else:
            waypoints.append(expert_path[idx])
    return waypoints

def transform_local(points,x):
    #transform points from gcop to torch
    torch_points = []
    r_inv = R.from_euler('zyx',x[3:6]).inv()
    for p in points:
        pos = r_inv.apply(p[0] - x[0:3]))
        rot = (r_inv*R.from_dcm(p[1])).as_euler('zyx')
        torch_points.append(np.hstack((pos,rot)))
    return torch_points

def list_to_ds(batch,save_path = None,name = None): 
    if save_path is not None:
        with open(save_path + name,'wb') as f:
            pickle.dump(batch,f,pickle.HIGHEST_PROTOCOL)
    return DAggerSet(batch)

def retrain_model(model,loader,epochs=2):
    #Run epochs of training on model
    #Include tensorboard output

def run_DAgger(sys_f,env,model,data_list=[],batch=512,j=4,
        plot_step_flag=False,max_steps=100,dt=0.1,save_path=None,mean_image=None):
    eps = 0.5
    ctr = 0
    data_batch=[]
    while eps > 1e-2:
        num_fails = 0
        print("Running Environment ",ctr)
        (x,camName,orange,tree) = shuffleEnv(env)
        if camName is "":
            print("camName not found")
            return 2
        for step in range(max_steps):
            #Get Image
            camAct = makeCamAct(x)
            image_arr = unity_image(env,camAct,camName)
            #Optionally save image
            if save_path is not None:
                save_image_array(image_arr,save_path,"sim_image"+str(step))
            #Calculate new goal
            goals = run_model(model,image_arr,mean_image)
            expert_path = run_gcop(x,tree,orange,step*dt)
            expert_goals = wp_from_traj(expert_path,step*dt)
            cost = DAggerCompare(x,goal,expert_goals,tree[0:3])
            if cost > eps:
                data_batch.append((image_arr,transform_local(expert_goals,x))
                num_fails += 1
                x = sys_f(x,expert_goals,dt)
            else:
                x = sys_f(x,goals,dt)
        if num_fails < 20:
            eps = eps/2
            print("Only ",num_fails," interventions, reducing eps to ",eps)
        if len(data_batch) > 100:
            print("Retraining")
            #Retrain model
            new_dataset = list_to_ds(data_batch)
            data_list.append(new_dataset)
            dataloader = DataLoader(ConcatDataset(data_list),batch_size=batch,shuffle=True,num_works=j)
            retrain_model(model,dataloader)
        ctr += 1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('load', help='model to load')
  parser.add_argument('--gpu', help='gpu to use')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  #Model Options
  parser.add_argument('--num_images', type=int, default=1, help='number of input images')
  parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
  parser.add_argument('--capacity', type=float, default=1, help='network capacity')
  parser.add_argument('--bins', type=int, default=30, help='number of bins per coordinate')
  parser.add_argument('--outputs', type=int, default=3, help='number of coordinates')
  parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
  parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
  #Simulation Options
  parser.add_argument('--iters', type=int, default=30, help='number of simulations')
  parser.add_argument('--steps', type=int, default=100, help='Steps per simulation')
  parser.add_argument('--hz', type=float, default=10, help='Recalculation rate')
  parser.add_argument('--env', type=str, default="unity/env_v4", help='unity filename')
  parser.add_argument('--plot_step', type=bool, default=False, help='PLot each step')
  parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run18.npy', help='Mean Image')
  args = parser.parse_args()
  args.max = [(1,0.5,0.1),(2,1,0.15),(4,1.5,0.2),(6,2,0.3),(7,0.3,0.5)]
  np.random.seed(args.seed)
  #Load model
  model = OrangeNet8(capacity=args.capacity,num_img=args.num_images,num_pts=args.num_pts,bins=args.bins,n_outputs=args.outputs)
  model.min = args.min
  model.max = args.max
  if os.path.isfile(args.load):
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Loaded Model: ",args.load)
  else:
    print("No checkpoint found at: ", args.load)
    return
  #Load Mean Image
  if not (os.path.exists(args.mean_image)):
      print('mean image file not found', args.mean_image)
      return 0
  else:
      print('mean image file found')
  #Create environment
  env = UnityEnvironment(file_name=args.env,seed=0)
  #env.reset()
  #Iterate simulations
  run_num = 0
  globalfolder = 'data/Sim' + str(run_num) + '/'
  while os.path.exists(globalfolder):
    run_num += 1
    globalfolder = 'data/Run' + str(run_num) + '/'
  trial_num = -1
  while trial_num < (args.iters-1) or args.iters is -1:
    trial_num += 1
    print('Trial Number ',trial_num)
    #Filenames
    foldername = "trial" + str(trial_num) + "/"
    os.makedirs(globalfolder + foldername)
    #TODO: RUN DAGGER HERE
    datasets = []
    datasets.append(#Run18 HERE
    err code = run_DAgger(sys_f,env,model,datasets)#TODO: add optional args
    if err_code is not 0:
        print("Dagger failed with code: ",err_code)
  env.close()

if __name__ == '__main__':
  main()
