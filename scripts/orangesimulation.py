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
    if (len(x) is 2) or (len(x) is 4):
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

def shuffleEnv(env_name,plot_only=False,future_version=False,trial_num=0,args=None):
    #Create environment
    print("Create env")
    if args is None:
        env = UnityEnvironment(file_name=env_name,worker_id=trial_num,seed=trial_num)
    else:
        env = UnityEnvironment(file_name=env_name,worker_id=args.worker_id+trial_num,seed=args.seed+trial_num)
    #Baseline Positions
    print("Init")
    x0 = np.array((-10,0,1))
    xmult = np.array((2,2,0.5))
    yaw0 = 0
    ymult = np.pi
    treePosYaw = np.array((0,0,0,0))
    treemult = np.array((0.5,0.5,0,180))
    treeHeight = 1.6
    orangePos = np.array((0,0,0.5))
    orangeR = 1.0 #0.6
    orangeRmult = 0.1 #0.3
    orangeHmult = 0.5
    #randomize environment
    xoffset = 2*xmult*(np.random.rand(3) - 0.5)
    x0_i = x0 + xoffset
    yaw0_i = yaw0 + 2*ymult*(np.random.rand(1) - 0.5)
    treeoffset = 2*treemult*(np.random.rand(4) - 0.5)
    treePos_i = treePosYaw + treeoffset
    theta = 2*np.pi*np.random.random_sample()
    orangeR_rand = orangeRmult*np.random.random_sample()
    orangeH_rand = orangeHmult*np.random.random_sample()
    R_i = orangeR + orangeR_rand
    orangeoffset = np.array((R_i*np.cos(theta), R_i*np.sin(theta),
                             orangePos[2] + orangeH_rand))
    orangePos_i = treePos_i[0:3] + orangeoffset
    x0_i = np.hstack((x0_i,yaw0_i,0,0))
    envAct = np.array((np.random.randint(6),np.random.randint(6),0))
    if not plot_only:
        (camName, envName) = setUpEnv(env,x0_i,treePos_i,orangePos_i,envAct, orangeColor = np.random.randint(9),future_version=future_version)
    else:
        camName = ""

    R_i = orangeR + orangeR_rand + 0.6
    orangeoffset = np.array((R_i*np.cos(theta), R_i*np.sin(theta),
                             orangePos[2] + orangeH_rand))
    orangePos_i = treePos_i[0:3] + orangeoffset

    return (env,x0_i, camName, envName, orangePos_i,treePos_i)

def setUpEnv(env, x0, treePos, orangePos, envAct=(0,1,0), treeScale = 0.125, orangeScale = 0.07,
             orangeColor = 0, future_version=False):
    print("Reset Env")
    env.reset()
    camAct = makeCamAct(x0)
    treeAct = np.array([np.hstack((gcopVecToUnity(treePos[0:3]),treePos[3],1))])
    orangeAct = np.array([np.hstack((gcopVecToUnity(orangePos),orangeColor,1))])
    envAct = np.array([np.hstack((envAct,1))])
    if not future_version: #future_version is master branch of mlagents as of 05/12/2020 
        print("Get names Env")
        names = env.get_behavior_names()
    else:
        print("Get names Env")
        names = env.behavior_specs

    camName = ""
    envName = ""
    for n in names:
        print(n)
        if "Tree" in n:
            #continue
            print(treeAct)
            print("Set tree Act Env")
            env.set_actions(n,treeAct)
            env.step()
            #print("tree set")
            continue
        if "Orange" in n:
            print(orangeAct)
            print("Set orange Act Env")
            env.set_actions(n,orangeAct)
            env.step()
            #print(orangeAct)
            continue
        if "Environment" in n:
            print(envAct)
            print("Set environment Act Env")
            env.set_actions(n,envAct)
            env.step()
            #print("Environment: ",envAct)
            envName = n
            continue
        if "Cam" in n:
            print(camAct)
            print("Set cam Act Env")
            env.set_actions(n,camAct)
            env.step()
            #print("Camera set")
            camName = n
            continue
    #print("Step Env")
    #env.step()
    #print("env step")
    #Fixing the first image bug
    #print("Cam Act Env")
    #env.set_actions(camName,camAct)
    #print("Step Env")
    #env.step()
    return (camName,envName)

def unity_image(env,act,cam_name,env_name=None):
    #print(act)
    #print("step called")
    obs = None
    if cam_name is not None:
        print("Cam Act Env")
        env.set_actions(cam_name,act)
        print("Step Env")
        env.step()
        print("Get Steps (cam) env")
        (ds,ts) = env.get_steps(cam_name)
        obs = ds.obs[0][0,:,:,:]
    #""" 
    #TODO ds.obs is empty list
    envobs = None
    if env_name is not None:
        print("EnvAct set env")
        env.set_actions(env_name,np.zeros((1,4)))
        print("Step env")
        env.step()
        print("Get Steps (env) env")
        (ds,ts) = env.get_steps(env_name)
        envobs = ds.obs[0][0,:,:,:]
    #"""
    if envobs is None:
        return obs
    if obs is None:
        return envobs
    return (obs,envobs)

def save_image_array(image_in,path,name):
  im_np = (image_in*255).astype('uint8')
  image = img.fromarray(im_np)
  if path is not None:
      image.save(path + name + '.png')
  else:
      image.show()

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
    logits = logits.cpu()
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
    print(goal[0,:])
    return goal

def trajCost(x_traj,u_traj,tree,orange,tf=15):
    N = len(x_traj)-1
    dt = float(tf)/N
    q = (3,3,2.5,#rotation log
         0,0,0,#position
         20,20,20,#rotation rate
         10,10,10)#velocity
    qf = (25,25,25,#rotation log
         50,50,50,#position
         5,5,5,#rotation rate
         5,5,5)#velocity
    r = (.1,.1,.1,1)
    cyl_r = 1.0 + 0.8 #0.6
    cyl_h = 1.6 + 0.2
    stiffness=500
    yaw_g = 10
    yawf = np.arctan2(tree[1]-orange[1],tree[0]-orange[0])
    orange = np.array(orange)
    rotf = R.from_euler('zyx',(yawf,0,0))
    mass = 0.5
    def yawCost(p,rot):
        df = np.array([orange[0] - p[0],orange[1] - p[1],0])
        dfnorm = np.linalg.norm(df)
        if dfnorm < 1e-10:
            #print("yawCost: 0")
            return 0
        df = df/np.linalg.norm(df)
        rot_arr = rot.as_matrix()
        value =  dt*yaw_g*(1 - np.dot(df,rot_arr[:,0]))
        #print("yawCost: ", value)
        return value
    def cylCost(p):
        cyl_o = np.array(tree)[0:3]
        v = np.array(p) - cyl_o
        if (p[2] < (cyl_o[2] + cyl_h)) and (p[2] > cyl_o[2]):
            v[2] = 0
        d = np.linalg.norm(v)
        if (cyl_r < d):
            #print("cylCost: 0")
            return 0
        value = dt*(stiffness/2)*(cyl_r - d)*(cyl_r-d)
        #print("cylCost: ", value)
        return value
    def groundCost(p):
        if (p[2] > 0):
            #print("groundCost: 0")
            return 0
        value = dt*(stiffness/2)*p[2]*p[2]
        #print("groundCost: ", value)
        return value
    def logR(rot):
        rot_m = rot.as_matrix()
        arg = (rot_m[0,0] + rot_m[1,1] + rot_m[2,2] - 1)/2
        if (arg >= 1):
            phi = 0
        elif (arg <= -1):
            phi = np.pi
        else:
            phi = np.arccos(arg)
        sphi = np.sin(phi)
        if (sphi < 1e-10):
            return (0,0,0)
        temp_m = (phi/(2*sphi))*(rot_m - rot_m.transpose())
        return np.array((temp_m[2,1],temp_m[0,2],temp_m[1,0]))
    def L(p,rot,v,w,u):
        #rot = R.from_dcm(rot)
        dR = logR(rot*rotf.inv())
        p = np.array(p)
        dp = p - orange
        w = logR(rot*rot_last.inv())/dt
        v = (p - np.array(p_last))/dt
        dx = np.hstack((dR,dp,w,v))
        #print("dx: ",dx)
        x_cost = dt*np.dot(q,dx*dx)
        #print("x cost: ", x_cost)
        du = np.array(u) - np.array([0,0,0,mass*9.81])
        #print("du: ",du)
        u_cost = dt*np.dot(r,du*du)
        #print("u cost: ", u_cost)
        value =  x_cost + u_cost + yawCost(p,rot) + cylCost(p) + groundCost(p)
        #print("L cost: ", value)
        return value
    def Lf(p,rot,v,w):
        #rot = R.from_dcm(rot)
        dR = logR(rot*rotf.inv())
        p = np.array(p)
        dp = p - orange
        dx = np.hstack((dR,dp,w,v))
        #print("dx: ",dx)
        x_cost = dt*np.dot(qf,dx*dx)
        #print("x cost: ", x_cost)
        value = x_cost + yawCost(p,rot) + cylCost(p) + groundCost(p)
        #print("Lf: ", value)
        return value
    cost = 0
    if (len(x_traj[0]) is 2):#gcop format
        x_last = x_traj[0][0]
        rot_last = R.from_dcm(x_traj[0][1])
        for i in range(N):
            rot = R.from_dcm(x_traj[i][1])
            w = logR(rot*rot_last.inv())/dt
            v = (p - np.array(p_last))/dt
            cost += L(x_traj[i][0],rot,v,w,u_traj[i])
            #print("Running Cost: ", cost)
            x_last = x_traj[i][0]
            rot_last = rot
        rot = R.from_dcm(x_traj[N][1])
        w = logR(rot*rot_last.inv())/dt
        v = (p - np.array(p_last))/dt
        cost += Lf(x_traj[N][0],rot,v,w)
        #print("Running Cost: ", cost)
    elif (len(x_traj[0]) is 4):
        for i in range(N):
            rot = R.from_dcm(x_traj[i][1])
            cost += L(x_traj[i][0],rot,x_traj[i][2],x_traj[i][3])
        rot = R.from_dcm(x_traj[N][1])
        cost += Lf(x_traj[N][0],rot,x_traj[N][2],x_traj[N][3])
    elif (len(x_traj[0]) is 6):
        x_last = x_traj[0][0:3]
        rot_last = R.from_euler(x_traj[0][3:6])#.as_matrix()
        for i in range(N):
            rot = R.from_euler(x_traj[i][3:6])#.as_matrix()
            w = logR(rot*rot_last.inv())/dt
            v = (p - np.array(p_last))/dt
            cost += L(x_traj[i][0:3],rot,v,w,u_traj[i])
            #print("Running Cost: ", cost)
            x_last = x_traj[i][0:3]
            rot_last = rot
        rot = R.from_euler(x_traj[N][3:6])#.as_matrix()
        w = logR(rot*rot_last.inv())/dt
        v = (p - np.array(p_last))/dt
        cost += Lf(x_traj[N][0:3],rot,v,w)
        #print("Running Cost: ", cost)
    else:
        print("Unrecognized x format: ",len(x))
    return cost

def run_gcop(x,tree,orange,t=0,tf=15,N=100,save_path=None):#TODO:Add in args to adjust more params
    epochs = 300
    stiffness = 500
    stiff_mult = 2.0
    q = (3,3,2.5,#rotation log
         0,0,0,#position
         20,20,20,#rotation rate
         10,10,10)#velocity
    qf = (25,25,25,#rotation log
         50,50,50,#position
         5,5,5,#rotation rate
         5,5,5)#velocity
    r = (.1,.1,.1,1)
    cyl_r = 1.0 + 0.8 #0.6
    cyl_h = 1.6 + 0.2
    yaw_g = 10
    yawf = np.arctan2(tree[1]-orange[1],tree[0]-orange[0])
    if (len(x) is 2):
        cameraPos = tuple(x[0])
        R0 = R.from_dcm(x[1])
        v0 = (0,0,0)
        w0 = (0,0,0)
    elif (len(x) is 4):
        cameraPos = tuple(x[0])
        R0 = R.from_dcm(x[1])
        v0 = x[2]
        w0 = x[3]
    elif (len(x) is 6):
        cameraPos = tuple(x[0:3])
        R0 = R.from_euler('zyx',x[3:6])
        v0 = (0,0,0)
        w0 = (0,0,0)
    else:
        print("Unsupported x format")
        return

    if save_path is not None:
        import pickle
        with open(save_path + 'metadata.pickle','wb') as f:
            metadata = {'N':N,'tf':tf,'epochs':epochs,'stiffness':stiffness,'stiff_mult':stiff_mult, 'x':x, 'xf':orange, 'yawf':yawf,
                    'cyl_o':tree , 'q':q, 'qf':qf, 'r':r, 'yaw_gain':yaw_g}
            pickle.dump(metadata,f,pickle.HIGHEST_PROTOCOL)

    cyl_r = 1.0 + 0.8 #0.6
    cyl_h = 1.6 + 0.2

    ref_traj = gcophrotor.trajgen_R(N,tf,epochs,cameraPos,tuple(R0.as_matrix().flatten()),v0,w0,
            tuple(orange), yawf, tuple(tree[0:3]),cyl_r,cyl_h,tuple(q),tuple(qf),tuple(r),yaw_g,0,0,
            stiffness,stiff_mult)
    return ref_traj

def sys_f_gcop(x,goal,dt,goal_time=3,hz=50,plot_flag=False):
    N = goal_time*hz
    epochs = 10
    q = (0,0,0,#rotation log#(0,0,0,0,0,0,0,0,0,0,0,0)
         0,0,0,#position
         10,10,10,#rotation rate
         10,10,10)#velocity
    qf = (10,10,10,#rotation log
         10,10,10,#position
         0,0,0,#rotation rate
         0,0,0)#velocity
    r = (.01,.01,.01,0.001)
    if (len(x) is 4):
        x0 = (tuple(x[0]),tuple(np.array(x[1]).flatten()),tuple(x[2]),tuple(x[3]))
    elif (len(x) is 2):
        x0 = (tuple(x[0]),tuple(np.array(x[1]).flatten()),(0,0,0),(0,0,0))
    elif (len(x) is 3):
        x0 = (tuple(x),(1,0,0,0,1,0,0,0,1),(0,0,0),(0,0,0))
    elif (len(x) is 4):
        x0 = (tuple(x[0:3]),tuple(R.from_euler('zyx',(x[3],0,0)).as_matrix().flatten()),(0,0,0),(0,0,0))
    elif (len(x) is 6):
        x0 = (tuple(x[0:3]),tuple(R.from_euler('zyx',x[3:6]).as_matrix().flatten()),(0,0,0),(0,0,0))
    else:
        print("Unrecognized x length: ",len(x))
    goal_list = []
    if (len(goal[0]) is 6):
        for g in goal:
            rot_local = R.from_euler('zyx',g[3:6])
            rot_x = R.from_dcm(np.reshape(x0[1],(3,3)))
            tot_rot = tuple((rot_local*rot_x).as_matrix().flatten())
            pos = rot_x.apply(g[0:3]) + np.array(x0[0])
            goal_list.append((tuple(pos),tot_rot))
    else:
        print("Unrecognized goal length: ",len(goal[0]))
    gcop_out = gcophrotor.trajgen_goal(N,goal_time,epochs,x0,goal_list[0],goal_list[1],goal_list[2],q,qf,r,0,0,0)
    ref_traj = gcop_out[0]
    print(dt)
    final_idx = int(dt*hz)
    print(final_idx)
    print(len(ref_traj))
    print(np.array(ref_traj[final_idx][0])-np.array(ref_traj[0][0]))
    if plot_flag:# or np.linalg.norm(dx) < 1e-3:
        print(goal)
        make_step_plot(goal_list,ref_traj[0:final_idx])
    return ref_traj[0:final_idx]

def sys_f_linear(x,goal,dt,goal_time=1,plot_flag=False):
    time_frac = dt/goal_time
    #goal_pt = goal[0]
    print(time_frac)
    print(x)
    print(goal)
    #exit()
    goal_pts = []
    if ((len(goal[0]) is 2) or (len(goal[0]) is 4)): #gcop syntax
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
    elif len(goal_pt) is 4:
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
    print(new_x)
    #exit(0)
    return new_x

def run_sim(args,sys_f,env_name,model,eps=0.1, max_steps=99,dt=0.1,save_path = None,
            plot_step_flag = False,mean_image = None,trial_num=0):
    x_list = []
    (env,x,camName,envName,orange,tree) = shuffleEnv(env_name,trial_num=trial_num,args=args)#,x0,orange,tree)
    #print(x.shape)
    if camName is "":
        print("camName not set")
        print("Close Env")
        env.close()
        return 2
    ref_traj = run_gcop(x,tree,orange)[0]#Take xs only
    #print(x.shape)
    image_spacing = 1#number of timesteps between images in multi-image networks
    for step in range(max_steps):
        x_list.append(x)
        print("State: ",x)
        if "sys_f_linear" in str(sys_f):
            dist = eps + 1
        else:
            dist = np.linalg.norm(x[0] - orange)
        if dist < eps:#TODO:Change to > when using DAgger
            print("exit if")
            if save_path is not None:
                ts = np.linspace(0,dt*(len(x_list)-1),len(x_list))
                make_full_plots(ts,x_list,orange,tree,saveFolder=save_path,truth=ref_traj)
            print("Close Env")
            env.close()
            return 0
        #Get Image
        image_arr = None
        for ii in range(model.num_images):
            temp_idx = max(0,int(len(x_list) - 1 - ii*image_spacing))
            camAct = makeCamAct(x_list[temp_idx])
            if image_arr is None:
                (image_arr,ext_image_arr) = unity_image(env,camAct,camName,envName)
                #image_arr = unity_image(env,camAct,camName)
                print(image_arr.shape)
            else:
                image_arr = np.concatenate((image_arr,unity_image(env,camAct,camName,None)),axis=2)#TODO: check axis number
                print(image_arr.shape)
        #Optionally save image
        if save_path is not None:
            save_image_array(image_arr[:,:,0:3],save_path,"sim_image"+str(step)) #TODO: Use concat axis from above
            save_image_array(ext_image_arr,save_path,"ext_image"+str(step)) #TODO
        goal = run_model(model,image_arr,mean_image)
        if "sys_f_linear" in str(sys_f):
            if type(x).__module__ == np.__name__:
                pass
            elif len(x) == 6:
                x_temp = list(x[0])
                x_temp.extend(R.from_matrix(x[1]).as_euler('zyx'))
                x = x_temp

        x = sys_f(x,goal,dt,plot_flag=plot_step_flag)
        if not "sys_f_linear" in str(sys_f):
            x = x[len(x)-1]
            print("gcop")
    #Ran out of time
    if save_path is not None:
        ts = np.linspace(0,dt*(len(x_list)-1),len(x_list))
        print("Saving at ",save_path)
        #make_full_plots(ts,x_list,orange,tree,saveFolder=save_path,truth=ref_traj) #TODO
    print("Close Env")
    env.close()
    os.system("python3 scripts/generate_gifs.py " + save_path + " --loc /home/gabe/ws/ros_ws/src/orange_picking/ &")
    return 1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('load', help='model to load')
  parser.add_argument('--gpu', help='gpu to use')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--worker_id', type=int, default=0, help='worker id')
  parser.add_argument('--name', type=int, default=0, help="model name")
  parser.add_argument('--debug', type=int, default=1, help="debug prints")
  #Model Options
  parser.add_argument('--num_images', type=int, default=1, help='number of input images')
  parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
  parser.add_argument('--capacity', type=float, default=1, help='network capacity')
  parser.add_argument('--bins', type=int, default=30, help='number of bins per coordinate')
  parser.add_argument('--outputs', type=int, default=3, help='number of coordinates')
  parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
  parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
  parser.add_argument('--resnet18', type=int, default=0, help='ResNet18')
  #Simulation Options
  parser.add_argument('--iters', type=int, default=5, help='number of simulations')
  parser.add_argument('--steps', type=int, default=500, help='Steps per simulation')
  parser.add_argument('--hz', type=float, default=10, help='Recalculation rate')
  parser.add_argument('--physics', type=float, default=50, help="Freq at which physics sim is performed")
  parser.add_argument('--env', type=str, default="unity/env_v6", help='unity filename')
  parser.add_argument('--plot_step', type=bool, default=False, help='PLot each step')
  parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run19.npy', help='Mean Image')
  args = parser.parse_args()
  #args.min = [(0,-0.5,-0.1),(0,-1,-0.15),(0,-1.5,-0.2),(0,-2,-0.3),(0,-3,-0.5)]
  #args.max = [(1,0.5,0.1),(2,1,0.15),(4,1.5,0.2),(6,2,0.3),(7,0.3,0.5)]
  args.min = [(0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0,-1,-0.15,-np.pi,-np.pi/2,-np.pi),(0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0,-2,-0.3,-np.pi,-np.pi/2,-np.pi),(0,-3,-0.5,-np.pi,-np.pi/2,-np.pi)]
  args.max = [(1,0.5,0.1,np.pi,np.pi/2,np.pi),(2,1,0.15,np.pi,np.pi/2,np.pi),(4,1.5,0.2,np.pi,np.pi/2,np.pi),(6,2,0.3,np.pi,np.pi/2,np.pi),(7,0.3,0.5,np.pi,np.pi/2,np.pi)]
  np.random.seed(args.seed)
  #Load model
  if not args.resnet18:
      model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=args.outputs)
  else:
      model = OrangeNet18(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=args.outputs)

  model.min = args.min
  model.max = args.max
  if os.path.isfile(args.load):
    print(args.load)
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
  #env.reset()
  #Iterate simulations
  run_num = args.name
  globalfolder = 'data/simulation/Sim' + str(run_num) + '_' + str(args.steps) + '_' + str(args.hz) + '_' + str(args.physics) + '/'
  while os.path.exists(globalfolder):
    run_num += 1
    globalfolder = 'data/simulation/Sim' + str(run_num) + '_' + str(args.steps) + '_' + str(args.hz) + '_' + str(args.physics) + '/'
  trial_num = -1
  while trial_num < (args.iters-1) or args.iters is -1:
    trial_num += 1
    print('Trial Number ',trial_num)
    #Filenames
    foldername = "trial" + str(trial_num) + "/"
    os.makedirs(globalfolder + foldername)
    err_code = run_sim(args,sys_f_linear,args.env,model,plot_step_flag = args.plot_step,
                       max_steps=args.steps,dt=3*(1.0)/args.hz,
                       save_path = globalfolder+foldername,mean_image=mean_image,trial_num=trial_num)
    if err_code is not 0:
      print('Simulation did not converge.  code is: ', err_code)

if __name__ == '__main__':
  main()
