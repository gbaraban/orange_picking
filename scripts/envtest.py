from orangesimulation import *
import numpy as np
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img
from scipy.spatial.transform import Rotation as R
import os
 
env = UnityEnvironment(file_name=None,seed=0)#"unity/env_v3",seed=0)
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
while trial_num < 10:
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
    camName = setUpEnv(env,x0_i,treePos_i,orangePos_i)
    camAct = makeCamAct(x0_i)
    im_arr = unity_image(env,camAct,camName)
    save_image_array(im_arr,globalfolder+foldername,"sim_image"+str(trial_num))
env.close()
