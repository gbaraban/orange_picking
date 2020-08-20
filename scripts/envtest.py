from orangesimulation import *
import numpy as np
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img
from scipy.spatial.transform import Rotation as R
import os
 
#env = UnityEnvironment(file_name=None)#'unity/env_v5',seed=0)
#Iterate simulations
run_num = 0
globalfolder = 'data/Sim' + str(run_num) + '/'
while os.path.exists(globalfolder):
    run_num += 1
    globalfolder = 'data/Sim' + str(run_num) + '/'
trial_num = -1
os.makedirs(globalfolder)
os.makedirs(globalfolder+"/images/")
os.makedirs(globalfolder+"/external/")
while trial_num < 0:
    trial_num += 1
    print('Trial Number ',trial_num)
    #Filenames
    #foldername = "trial" + str(trial_num) + "/"
    #os.makedirs(globalfolder + foldername)
    env_name = 'unity/env_v7'
    env = UnityEnvironment(file_name=env_name,worker_id=trial_num,seed=0)
    x = np.array([-1.5,0,1,0,0,0])
    tree = np.array([0,0,0,0])
    tree[3] = 360*np.random.random_sample()
    orange = np.array([-1,0,1])
    for ii in range(6):
        cR = 2 #1e-1*ii
        print("CollisionR: ",cR)
        (camName,envName,orange,occ_frac) = setUpEnv(env,x,tree,orange,collisionR = cR)
        print("Occlusion: ",occ_frac)
    #cR = 1e-1*trial_num
    #(camName,envName) = setUpEnv(env,x,tree,orange,collisionR = cR)
    #(env,x,camName,envName, orange, tree) = shuffleEnv(env_name,trial_num=trial_num,cR = -0.01*trial_num)#setUpEnv(env,x0_i,treePos_i,orangePos_i)
    #x = np.array([0,0,0,0,0,0])
    #camAct = makeCamAct(x)
    #(im_arr,ext_arr) = unity_image(env,camAct,camName,envName)
    #ext_arr = unity_image(env,camAct,None,envName)
    #print("saving to folder: ", globalfolder)
    #save_image_array(im_arr,globalfolder+"/images/","sim_image"+str(trial_num))
    #save_image_array(ext_arr,None,None)#globalfolder+"/external/","ext_image"+str(trial_num))
    #print("Close Env")
    env.close()
