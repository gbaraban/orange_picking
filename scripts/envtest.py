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
while trial_num < 3:
    trial_num += 1
    print('Trial Number ',trial_num)
    #Filenames
    #foldername = "trial" + str(trial_num) + "/"
    #os.makedirs(globalfolder + foldername)
    env_name = 'unity/env_v7'
    (env,x,camName,envName, orange, tree) = shuffleEnv(env_name,trial_num=trial_num,cR = -0.01*trial_num)#setUpEnv(env,x0_i,treePos_i,orangePos_i)
    #x = np.array([0,0,0,0,0,0])
    camAct = makeCamAct(x)
    #(im_arr,ext_arr) = unity_image(env,camAct,camName,envName)
    ext_arr = unity_image(env,camAct,None,envName)
    print("saving to folder: ", globalfolder)
    #save_image_array(im_arr,None,None)#globalfolder+"/images/","sim_image"+str(trial_num))
    save_image_array(ext_arr,None,None)#globalfolder+"/external/","ext_image"+str(trial_num))
    print("Close Env")
    env.close()
