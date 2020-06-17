import numpy as np
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img
from scipy.spatial.transform import Rotation as R
from orangenetarch import *
import torch
import torch.nn as nn
import argparse
from plotting.parsetrajfile import *
import os
import gcophrotor

def gcopVecToUnity(v):
    temp = np.array((v[0],v[2],v[1]))
    return temp

def makeCamAct(x):
    #Set up camera action
    if (len(x) is 2):
        cameraPos = x[0]
        r = R.from_dcm(x[1])
    elif (len(x) is 6):
        cameraPos = x[0:3]
        r = R.from_euler('zyx',x[3:6])
    else:
        print("Unsupported x format")
        return
    euler = r.as_euler(seq = 'zxy',degrees = True) #using weird unity sequence
    unityEuler = (euler[1],-euler[0],euler[2]) 
    #print("euler: ", unityEuler)
    cameraAction = np.hstack((gcopVecToUnity(cameraPos),unityEuler,1))
    #print("cameraAction: ",cameraAction)
    return np.array([cameraAction])

def shuffleEnv(env,plot_only=False):
    #Baseline Positions
    x0 = np.array((-10,0,1))
    xmult = np.array((2,2,0.5))
    yaw0 = 0
    ymult = np.pi
    treePosYaw = np.array((0,0,0,0))
    treemult = np.array((0.5,0.5,0,180))
    treeHeight = 1.6
    orangePos = np.array((0,0,0.5))
    orangeR = 0.6
    orangeRmult = 0.3
    orangeHmult = 0.5
    #randomize environment
    xoffset = 2*xmult*(np.random.rand(3) - 0.5)
    x0_i = x0 + xoffset
    yaw0_i = yaw0 + 2*ymult*(np.random.rand(1) - 0.5)
    treeoffset = 2*treemult*(np.random.rand(4) - 0.5)
    treePos_i = treePosYaw + treeoffset
    theta = 2*np.pi*np.random.random_sample()
    R_i = orangeR + orangeRmult*np.random.random_sample()# + 0.2
    orangeoffset = np.array((R_i*np.cos(theta), R_i*np.sin(theta),
                             orangePos[2] + orangeHmult*np.random.random_sample()))
    orangePos_i = treePos_i[0:3] + orangeoffset
    x0_i = np.hstack((x0_i,yaw0_i,0,0))
    envAct = np.array((np.random.randint(6),np.random.randint(6),0))
    if not plot_only:
        camName = setUpEnv(env,x0_i,treePos_i,orangePos_i,envAct, orangeColor = np.random.randint(9))
    else:
        camName = ""

    return (x0_i, camName, orangePos_i,treePos_i)

def setUpEnv(env, x0, treePos, orangePos, envAct=(0,1,0), treeScale = 0.125, orangeScale = 0.07,
             orangeColor = 0, future_version=False):
    env.reset()
    camAct = makeCamAct(x0)
    treeAct = np.array([np.hstack((gcopVecToUnity(treePos[0:3]),treePos[3],1))])
    orangeAct = np.array([np.hstack((gcopVecToUnity(orangePos),orangeColor,1))])
    envAct = np.array([np.hstack((envAct,1))])
    if not future_version: #future_version is master branch of mlagents as of 05/12/2020 
        names = env.get_behavior_names()
    else:
        names = env.behavior_specs

    camName = ""
    for n in names:
        #print(n)
        if "Tree" in n:
            env.set_actions(n,treeAct)
            #print("tree set")
            continue
        if "Orange" in n:
            env.set_actions(n,orangeAct)
            #print(orangeAct)
            continue
        if "Environment" in n:
            env.set_actions(n,envAct)
            #print(envAct)
            continue
        if "Cam" in n:
            env.set_actions(n,camAct)
            #print("Camera set")
            camName = n
            continue
    env.step()
    #print("env step")
    #Fixing the first image bug
    env.set_actions(camName,camAct)
    env.step()
    return camName

def unity_image(env,act,cam_name):
    #print(act)
    env.set_actions(cam_name,act)
    env.step()
    #print("step called")
    (ds,ts) = env.get_steps(cam_name)
    obs = ds.obs[0][0,:,:,:]
    return obs

def save_image_array(image_in,path,name):
  im_np = (image_in*255).astype('uint8')
  image = img.fromarray(im_np)
  image.save(path + name + '.png')

def run_model(model,image_arr,mean_image=None,device=None):
    #Calculate new goal
    if mean_image is None:
        mean_subtracted = (image_arr).astype('float32')
    else:
        mean_subtracted = (image_arr-mean_image).astype('float32')
    image_tensor = torch.tensor(mean_subtracted)
    if device is not None:
        image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.permute(2,0,1).unsqueeze(0)
    logits = model(image_tensor)
    logits = logits.view(1,model.outputs,model.num_points,model.bins).detach().numpy()
    predict = np.argmax(logits,axis=3)
    #print("Predict: ", predict)
    goal = []
    for pt in range(model.num_points):
        point = []
        for coord in range(model.outputs):
            bin_size = (model.max[pt][coord] - model.min[pt][coord])/model.bins
            point.append(model.min[pt][coord] + bin_size*predict[0,coord,pt])
        goal.append(np.array(point))
    goal = np.array(goal)
    return goal

def run_gcop(x,tree,orange,t=0,tf=15,N=100,save_path=None):#TODO:Add in args to adjust more params
    epochs = 300
    stiffness = 500
    stiff_mult = 2.0
    q = (0.2,0.2,0,#rotation log
         0,0,0,#position
         1,1,1,#rotation rate
         20,20,20)#velocity
    qf = (10,10,10,#rotation log
         50,50,50,#position
         0.5,0.5,0.5,#rotation rate
         5,5,5)#velocity
    r = (.1,.1,.1,1)
    yaw_g = 10
    yawf = np.arctan2(tree[1]-orange[1],tree[0]-orange[0])
    if (len(x) is 2):
        cameraPos = tuple(x[0])
        R0 = R.from_dcm(x[1])
    elif (len(x) is 6):
        cameraPos = tuple(x[0:3])
        R0 = R.from_euler('zyx',x[3:6])
    else:
        print("Unsupported x format")
        return

    if save_path is not None:
        import pickle
        with open(save_path + 'metadata.pickle','wb') as f:
            metadata = {'N':N,'tf':tf,'epochs':epochs,'stiffness':stiffness,'stiff_mult':stiff_mult, 'x0':x0_i, 'yaw0':yaw0_i,'xf':orangePos_i,'yawf':yawf,
                    'cyl_o':treePos_i,'cyl_r':orangeR, 'h':treeHeight, 'q':q, 'qf':qf, 'r':r, 'yaw_gain':yaw_g, 'rp_gain':rp_g, 'dir_gain':direction_gain}
        pickle.dump(metadata,f,pickle.HIGHEST_PROTOCOL)

    ref_traj = gcophrotor.trajgen_R(N,tf,epochs,cameraPos,tuple(R0.as_matrix().flatten()),
            tuple(orange), yawf, tuple(tree[0:3]),0.6,1.6,tuple(q),tuple(qf),tuple(r),yaw_g,0,0,
            stiffness,stiff_mult)
    return ref_traj

def sys_f_linear(x,goal,dt,goal_time=1,plot_flag=False):
    time_frac = dt/goal_time
    #goal_pt = goal[0]
    goal_pts = []
    if len(goal[0]) is 2: #gcop syntax
        for g in goal:
            goal_pts.append(g[0])
    else:
        r = R.from_euler('zyx',x[3:6])
        for g in goal:
            goal_pts.append(r.apply(g[0:3]) + x[0:3])
    dx = goal_pts[0] - x[0:3]
    goal_pt = goal_pts[0]
    if len(goal_pt) is 2:
        rot1 = R.from_dcm(goal_pt[1])
        rot1 = R.from_euler('zyx',rot1.as_euler('zyx')*time_frac)
        rot2 = R.from_euler('zyx',x[3:6])
        (yaw,pitch,roll) = (rot1*rot2).as_euler('zyx')
    elif (goal[0].size is 4):
        yaw = x[3] + time_frac*(goal_pt[3] - x[3])
        roll = np.arcsin(dx[0]*np.sin(yaw) - dx[1]*np.cos(yaw))
        pitch = np.arctan2(np.cos(roll)*(dx[0]*np.cos(yaw) - dx[1]*np.sin(yaw)),np.cos(roll)*(dx[2]+9.8))
    elif (goal[0].size is 3):
        yaw = np.arctan2(dx[1],dx[0])
        roll = np.arcsin(dx[0]*np.sin(yaw) - dx[1]*np.cos(yaw))
        pitch = np.arctan2(np.cos(roll)*(dx[0]*np.cos(yaw) - dx[1]*np.sin(yaw)),np.cos(roll)*(dx[2]+9.8))
    elif (goal[0].size is 6):
        rot1 = R.from_euler('zyx',goal[0][3:6]*time_frac)
        rot2 = R.from_euler('zyx',x[3:6])
        (yaw,pitch,roll) = (rot1*rot2).as_euler('zyx')
    new_pos = x[0:3] + time_frac*dx
    new_x = np.hstack((new_pos,yaw,pitch,roll))
    if plot_flag:# or np.linalg.norm(dx) < 1e-3:
        print(goal)
        make_step_plot(goal_pts,[x,new_x])
    return new_x

def run_sim(sys_f,env,model,eps=0.1, max_steps=99,dt=0.1,save_path = None,
            plot_step_flag = False,mean_image = None):
    x_list = []
    (x,camName,orange,tree) = shuffleEnv(env)#,x0,orange,tree)
    if camName is "":
        print("camName not set")
        return 2
    ref_traj = run_gcop(x,tree,orange)
    for step in range(max_steps):
        x_list.append(x)
        #print("State: ",x)
        dist = np.linalg.norm(x[0:3] - orange)
        if dist < eps:#TODO:Change to > when using DAgger
            if save_path is not None:
                ts = np.linspace(0,dt*(len(x_list)-1),len(x_list))
                make_full_plots(ts,x_list,orange,tree,saveFolder=save_path,truth=ref_traj)
            return 0
        #Get Image
        camAct = makeCamAct(x)
        image_arr = unity_image(env,camAct,camName)
        #Optionally save image
        if save_path is not None:
            save_image_array(image_arr,save_path,"sim_image"+str(step))
        goal = run_model(model,image_arr,mean_image)
        x = sys_f(x,goal,dt,plot_flag=plot_step_flag)
    #Ran out of time
    if save_path is not None:
        ts = np.linspace(0,dt*(len(x_list)-1),len(x_list))
        print("Saving at ",save_path)
        make_full_plots(ts,x_list,orange,tree,saveFolder=save_path,truth=ref_traj)
    return 1

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
  parser.add_argument('--env', type=str, default="unity/env_v5", help='unity filename')
  parser.add_argument('--plot_step', type=bool, default=False, help='PLot each step')
  parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run18.npy', help='Mean Image')
  args = parser.parse_args()
  #args.min = [(0,-0.5,-0.1),(0,-1,-0.15),(0,-1.5,-0.2),(0,-2,-0.3),(0,-3,-0.5)]
  #args.max = [(1,0.5,0.1),(2,1,0.15),(4,1.5,0.2),(6,2,0.3),(7,0.3,0.5)]
  args.min = [(0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0,-1,-0.15,-np.pi,-np.pi/2,-np.pi),(0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0,-2,-0.3,-np.pi,-np.pi/2,-np.pi),(0,-3,-0.5,-np.pi,-np.pi/2,-np.pi)]
  args.max = [(1,0.5,0.1,np.pi,np.pi/2,np.pi),(2,1,0.15,np.pi,np.pi/2,np.pi),(4,1.5,0.2,np.pi,np.pi/2,np.pi),(6,2,0.3,np.pi,np.pi/2,np.pi),(7,0.3,0.5,np.pi,np.pi/2,np.pi)]
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
      mean_image = np.load(args.mean_image)
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
    err_code = run_sim(sys_f_linear,env,model,plot_step_flag = args.plot_step,
                       max_steps=args.steps,dt=(1.0)/args.hz,
                       save_path = globalfolder+foldername,mean_image=mean_image)
    if err_code is not 0:
      print('Simulation did not converge.  code is: ', err_code) 
  env.close()

if __name__ == '__main__':
  main()
