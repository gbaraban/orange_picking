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

def init_angle(tree,orange,x0):
    tree = np.array(tree)
    orange = np.array(orange)
    x0 = np.array(x0)
    vec_o = np.array(orange[0:2]-tree[0:2])
    vec_o = vec_o/np.linalg.norm(vec_o)
    vec_q = np.array(x0[0:2]-tree[0:2])
    vec_q = vec_q/np.linalg.norm(vec_q)
    cos_angle = np.dot(vec_o,vec_q)
    return np.degrees(np.arccos(cos_angle))


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
  #parser.add_argument('--physics', type=float, default=5, help='Recalculation rate')
  parser.add_argument('--env', type=str, default="unity/env_v7", help='unity filename')
  parser.add_argument('--plot_step', type=bool, default=False, help='PLot each step')
  parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run23.npy', help='Mean Image')
  parser.add_argument('--worker_id', type=int, default=0, help='Worker ID for Unity')
  parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run24.npy', help='Mean Image')
  parser.add_argument('--seg', type=bool, default=False, help='Mean Image')
  parser.add_argument('--seg_mean_image', type=str, default='data/mean_imgv2_data_seg_Run24.npy', help='Mean Image')
  parser.add_argument('--relative_pose', type=bool, default=False, help='Relative Position')
  parser.add_argument('--name', type=int, default=0, help="model name")
  #Training Options
  args = parser.parse_args()
  if not args.relative_pose:
    args.min = [(-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -1.0, -0.4, -np.pi/4, -0.1, -0.1), (-0.1, -1.25, -0.6, -np.pi/2, -0.1, -0.1)]
    args.max = [(1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.75, 1.0, 0.4, np.pi/4, 0.1, 0.1), (2.5, 1.25, 0.6, np.pi/2, 0.1, 0.1)]
  else:
    args.min = [(-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -0.5, -0.25, -0.5, -0.1, -0.1)]
    args.max = [(1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.0, 0.5, 0.25, 0.5, 0.1, 0.1)]np.random.seed(args.seed)
  #Load model
  if args.seg:
    n_channels = 5
  else:
    n_channels = 3

  if not args.resnet18:
      model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=args.outputs,num_channels=n_channels,real=False)
  else:
      model = OrangeNet18(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=args.outputs)

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
      if args.seg:
          seg_mean = np.load(args.seg_mean_image)
          mean_image = np.concatenate((mean_image,seg_mean),axis=2)
  
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
  run_num = args.name
  globalfolder = 'data/dagger/Fig' + str(run_num) + '_' + str(args.steps) + '_' + str(args.hz) + '/' #+ str(args.physics) + '/'
  while os.path.exists(globalfolder):
      run_num += 1
      globalfolder = 'data/dagger/Fig' + str(run_num) +  '_' + str(args.steps) + '_' + str(args.hz) + '/' #+ str(args.physics) + '/'
  
  num_trials_each = 50
  num_trials_each_occ = 10
  
  trajectories = []
  final_error = []
  head_ons = []
  head_on_data = []
  sides = []
  side_data = []
  backs = []
  dt = 1/args.hz
  orange_offset = (0,0,0) #TODO: change this
  pos_threshold = 0.02
  tf = args.steps*dt
  N = int(tf*physics_hz)
  
  occlusion_bins = 7
  occlusion_nums = [[] for _ in range(occlusion_bins)]
  num_trials_each = 10
  min_occ = 0#min(occlusions)
  max_occ = 0.7#max(occlusions)
  occlusion_trajectories = []
  occlusions = []
  occlusion_final_error = []

  trial_num = -1
  exp = -1

  flag1 = True
  flag2 = True
  occ_flag = True
  main_flag = True
  while main_flag:
    trial_num += 1
    exp += 1
    foldername = "trial" + str(exp) + "/"
    os.makedirs(globalfolder+foldername)
    save_path = globalfolder+foldername
    flag = True
    #Generate an environment
    while flag:
      occ = 1.0
      setup_data = None
      while occ > 0.7:      
        #(env,x0,camName,envName,orange,tree,occ,orangePosTrue)
        setup_data = shuffleEnv(args.env_name,trial_num=trial_num,include_occlusion=True,args=args,exp=exp)
        occ = setup_data[6]
        env = setup_data[0]
        tree = setup_data[5]
        orange = setup_data[4]
        x0 = setup_data[1]
        if occ > 0.7:
          trial_num += 1
          env.close()
          time.sleep(5)
      #ret_val = 1
      angle = init_angle(tree,orange,x0)
      if angle <  45:
        occ_bin = int((occ-min_occ)*occlusion_bins/(max_occ-min_occ))
        if len(occlusion_bins[occ_bin]) >= num_trials_each_occ:
          occ_bin_flag = False
        else:
          occ_bin_flag = True

        if len(head_ons) >= num_trials_each:
          flag1 = False
        
        if flag1 or occ_bin_flag:
          ret_val = 1

          if flag1:
            head_ons.append(exp)
            head_on_data.append(setup_data)
            head_on_suc.append(ret_val)

          if occ_bin_flag:
            occlusion_nums.append()
          flag = False
      else:
        if len(sides) >= num_trials_each:
          continue
        else:
          sides.append(exp)
          side_data.append(setup_data)
          side_data_suc.append(ret_val)
          flag = False
    

    if not flag1 and not flag2 and not occ_flag:
      main_flag = False
      
    '''
    #Generate a trajectory
    trajectory = run_trajectory(x0,N,steps,dt,physics_hz,model,image_spacing,env,camName,save_path,mean_image,device)
    #Post Process
    trajectories.append(trajectory)
    final_x = trajectory[len(trajectory)-1]
    ee_pos = final_x[0:3] + R.from_euler('zyx',final_x[3:6]).apply(orange_offset)
    final_error.append(np.linalg.norm(ee_pos-orange))
    '''
  #Calculate success rates
  '''
  head_on_sucessess = sum([1 if (pos_threshold > final_error[ii]) else 0 for ii in head_ons])
  side_sucessess = sum([1 if (pos_threshold > final_error[ii]) else 0 for ii in sides])
  back_sucessess = sum([1 if (pos_threshold > final_error[ii]) else 0 for ii in backs])
  '''
  print("Total Success Rate: ",head_on_successes+side_successes+back_successes," out of ",len(head_ons)+len(sides)+len(backs))
  print("Head On Success Rate: ",head_on_success," out of ",len(head_ons))
  print("Side Success Rate: ",side_success," out of ",len(sides))
  print("Back Success Rate: ",back_success," out of ",len(backs))
  #Occlusion Tests
  
  
  while min([len(occlusion_nums[ii]) for ii in range(occlusion_bins)]) < num_trials_each:
    trial_num += 1
    foldername = "occ_trial" + str(trial_num) + "/"
    os.makedirs(globalfolder+foldername)
    save_path = globalfolder+foldername
    flag = True
    #Generate an environment
    while flag:
      (env,x0,camName,envName,orange,tree,occ,orangePosTrue) = shuffleEnv(args.env_name,trial_num=exp,include_occlusion=True,args=args)
      angle = init_angle(tree,orange,x0)
      if angle >  45:
        continue
      else:
        if len(occlusion_nums[occ_bin]) >= num_trials_each:
          continue
        else:
          occlusions.append(occ)
          occlusion_nums.append(trial_num)
          flag = False
    trajectory = run_trajectory(x0,N,steps,dt,physics_hz,model,image_spacing,env,camName,save_path,mean_image,device)
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

