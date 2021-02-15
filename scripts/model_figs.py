import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import argparse
from scipy.spatial.transform import Rotation as R
import PIL.Image as img
from orangenetarch import *
from orangeimages import *

def run_trajectory(x0,N,steps,dt,physics_hz,model,image_spacing,env,camName,save_path,mean_image,device):
  #Generate a trajectory
  trajectory = [x0 for i in range(N+1)]
  for step in range(steps + 1):
    step_time = step*dt
    x = trajectory[int(step_time*physics_hz)]
    #Get Image
    image_arr = None
    for ii in range(model.num_images):
        temp_idx = max(0,int(step_time*physics_hz - int(ii*image_spacing)))
        camAct = makeCamAct(expert_path[temp_idx])
        if image_arr is None:
            image_arr = unity_image(env,camAct,camName)
            #print(image_arr.shape)
        else:
            image_arr = np.concatenate((image_arr,unity_image(env,camAct,camName)),axis=2)#TODO: check axis number
            #print(image_arr.shape)
    save_image_array(image_arr[:,:,0:3],save_path,"sim_image"+str(step)) #TODO: use concatenation axis from above
    #Calculate new goal
    goals = run_model(model,image_arr,mean_image, device)# TODO: fix-->'DataParallel' object has no attribute 'outputs': when adding device with multi gpu
    x_new = sys_f_gcop(x,goals,dt,plot_flag=True)[0]
    trajectory[int(step_time*physics_hz):int(step_time*physics_hz)+len(x_new)] = x_new
  return trajectory

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
  parser.add_argument('--outputs', type=int, default=6, help='number of coordinates')
  parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
  parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
  parser.add_argument('--resnet18', type=int, default=0, help='ResNet18')
  #Simulation Options
  #parser.add_argument('--iters', type=int, default=30, help='number of simulations')
  parser.add_argument('--steps', type=int, default=100, help='Steps per simulation')
  parser.add_argument('--hz', type=float, default=5, help='Recalculation rate')
  parser.add_argument('--physics', type=float, default=5, help='Recalculation rate')
  parser.add_argument('--env', type=str, default="unity/env_v7", help='unity filename')
  parser.add_argument('--plot_step', type=bool, default=False, help='PLot each step')
  parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run23.npy', help='Mean Image')
  parser.add_argument('--worker_id', type=int, default=0, help='Worker ID for Unity')
  #Training Options
  args = parser.parse_args()
  args.min = [(0.,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0.,-1.,-0.15,-np.pi,-np.pi/2,-np.pi),(0.,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0.,-2.,-0.3,-np.pi,-np.pi/2,-np.pi),(0.,-3.,-0.5,-np.pi,-np.pi/2,-np.pi)]
  args.max = [(1.,0.5,0.1,np.pi,np.pi/2,np.pi),(2.,1.,0.15,np.pi,np.pi/2,np.pi),(4.,1.5,0.2,np.pi,np.pi/2,np.pi),(6.,2.,0.3,np.pi,np.pi/2,np.pi),(7.,0.3,0.5,np.pi,np.pi/2,np.pi)]
  np.random.seed(args.seed)
  #Load model
  if not args.resnet18:
      model = OrangeNet8(capacity=args.capacity,num_img=args.num_images,num_pts=args.num_pts,bins=args.bins,mins=args.min,maxs=args.max,n_outputs=args.outputs)
  else:
      model = OrangeNet18(capacity=args.capacity,num_img=args.num_images,num_pts=args.num_pts,bins=args.bins,mins=args.min,maxs=args.max,n_outputs=args.outputs)

  model.min = args.min
  model.max = args.max

  #if args.worker_id == 100:
  #    args.worker_id = args.seed

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
      mean_image = np.load(args.mean_image)

  import copy
  temp_image = copy.deepcopy(mean_image)
  if args.num_images != 1:
      for i in range(args.num_images - 1):
          mean_image = np.concatenate((mean_image,temp_image),axis=2)

  #Pick CUDA Device
  use_cuda = torch.cuda.is_available()
  print('Cuda Flag: ',use_cuda)
  if use_cuda:
      if args.gpu:
          device = torch.device('cuda:'+str(args.gpu))
          model = model.to(device)
      else:
          device = torch.device('cuda')
          model = model.to(device)
          #if (torch.cuda.device_count() > 1):
          #    model = nn.DataParallel(model)
  else:
      device = torch.device('cpu')
  #Make Run Folder
  run_num = 0
  globalfolder = 'data/dagger/Fig' + str(run_num) + '_' + str(args.steps) + '_' + str(args.hz) + '_' + str(args.physics) + '/'
  while os.path.exists(globalfolder):
      run_num += 1
      globalfolder = 'data/dagger/Fig' + str(run_num) +  '_' + str(args.steps) + '_' + str(args.hz) + '_' + str(args.physics) + '/'
  num_trials_each = 100
  trajectories = []
  final_error = []
  head_ons = []
  sides = []
  backs = []
  dt = 1/args.hz
  orange_offset = (0,0,0) #TODO: change this
  pos_threshold = 0.02
  tf = args.steps*dt
  N = int(tf*physics_hz)
  trial_num = -1
  while (len(head_ons) < num_trials_each) and (len(sides) < num_trials_each) and (len(backs) < num_trials_each):
    trial_num += 1
    foldername = "trial" + str(trial_num) + "/"
    os.makedirs(globalfolder+foldername)
    save_path = globalfolder+foldername
    flag = True
    #Generate an environment
    while flag:
      (env,x0,camName,envName,orange,tree,occ,orangePosTrue) = shuffleEnv(args.env_name,trial_num=exp,include_occlusion=True,args=args)
      angle = ...#TODO: fill in
      angle = abs(angle)
      if angle <  45:
        if len(head_ons) >= num_trials_each:
          continue
        else:
          head_ons.append(trial_num)
          flag = False
      elif angle < 135:
        if len(sides) >= num_trials_each:
          continue
        else:
          sides.append(trial_num)
          flag = False
      else:
        if len(backs) >= num_trials_each:
          continue
        else:
          backs.append(trial_num)
          flag = False
    #Generate a trajectory
    trajectory = run_trajectory(x0,N,steps,dt,physics_hz,model,image_spacing,env,camName,save_path,mean_image,device):
    #Post Process
    trajectories.append(trajectory)
    final_x = trajectory[len(trajectory)-1]
    ee_pos = final_x[0:3] + R.from_euler('zyx',final_x[3:6]).apply(orange_offset)
    final_error.append(np.linalg.norm(ee_pos-orange))
  #Calculate success rates
  head_on_sucessess = sum([1 if (pos_threshold > final_error[ii]) else 0 for ii in head_ons])
  side_sucessess = sum([1 if (pos_threshold > final_error[ii]) else 0 for ii in sides])
  back_sucessess = sum([1 if (pos_threshold > final_error[ii]) else 0 for ii in backs])
  print("Total Success Rate: ",head_on_successes+side_successes+back_successes," out of ",len(head_ons)+len(sides)+len(backs))
  print("Head On Success Rate: ",head_on_success," out of ",len(head_ons))
  print("Side Success Rate: ",side_success," out of ",len(sides))
  print("Back Success Rate: ",back_success," out of ",len(backs))
  #Occlusion Tests
  occlusion_bins = 10
  occlusion_nums = [[] for _ in range(occlusion_bins)]
  min_occ = 0#min(occlusions)
  max_occ = 0.7#max(occlusions)
  trial_num = -1
  occlusion_trajectories = []
  occlusions = []
  occlusion_final_error = []
  while min([len(occlusion_nums[ii]) for ii in range(occlusion_bins)]) < num_trials_each:
    trial_num += 1
    foldername = "occ_trial" + str(trial_num) + "/"
    os.makedirs(globalfolder+foldername)
    save_path = globalfolder+foldername
    flag = True
    #Generate an environment
    while flag:
      (env,x0,camName,envName,orange,tree,occ,orangePosTrue) = shuffleEnv(args.env_name,trial_num=exp,include_occlusion=True,args=args)
      angle = ...#TODO: fill in
      angle = abs(angle)
      if angle >  45:
        continue
      else:
        occ_bin = int((occ-min_occ)*occlusion_bins/(max_occ-min_occ))
        if len(occlusion_nums[occ_bin]) >= num_trials_each:
          continue
        else:
          occlusions.append(occ)
          occlusion_nums.append(trial_num)
          flag = False
    trajectory = run_trajectory(x0,N,steps,dt,physics_hz,model,image_spacing,env,camName,save_path,mean_image,device):
    #Post Process
    occlusion_trajectories.append(trajectory)
    final_x = trajectory[len(trajectory)-1]
    ee_pos = final_x[0:3] + R.from_euler('zyx',final_x[3:6]).apply(orange_offset)
    occlusion_final_error.append(np.linalg.norm(ee_pos-orange))
  fig = plt.figure()
  ax = plt.axes()
  ax.set_xlim((min_occ,max_occ))
  occ_labels = np.linspace(min_occ,max_occ,occlusion_bins,endpoint=False)
  occ_successes = [sum([1 if (occlusion_final_error[ii] < pos_threshold) else 0 for ii in occlusion_nums[jj]]) for jj in range(occlusion_bins)]
  occ_percents = [float(occ_successes[ii])/len(occlusion_nums[ii]) for ii in range(occlusion_bins)]
  print("Occlussion Success Rates: ", occ_percents)
  ax.plot(occ_labels,occlussion_percents)
  plt.show()

if __name__ == '__main__':
  main()

