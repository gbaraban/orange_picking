from orangesimulation import *
import numpy as np
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img
from scipy.spatial.transform import Rotation as R
import os
 
env = UnityEnvironment(file_name=None,seed=0)#"unity/env_v3",seed=0)
#Iterate simulations
run_num = 0
globalfolder = 'data/Sim' + str(run_num) + '/'
while os.path.exists(globalfolder):
    run_num += 1
    globalfolder = 'data/Run' + str(run_num) + '/'
trial_num = -1
os.makedirs(globalfolder)
while trial_num < 10:
    trial_num += 1
    print('Trial Number ',trial_num)
    #Filenames
    #foldername = "trial" + str(trial_num) + "/"
    #os.makedirs(globalfolder + foldername)
    (x,camName) = shuffleEnv(env)#setUpEnv(env,x0_i,treePos_i,orangePos_i)
    camAct = makeCamAct(x)
    im_arr = unity_image(env,camAct,camName)
    save_image_array(im_arr,globalfolder,"sim_image"+str(trial_num))
env.close()
