import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pickle
from orangesimulation import *
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_dir', nargs='+', help='pickle folder')
    args = parser.parse_args()
    for folder_name in args.trial_dir:
        sim_img_list = []
        ext_img_list = []
        if os.path.isfile(folder_name + "/metadata.pickle"):
            metadata = pickle.load(open(folder_name+"/metadata.pickle", "rb"))
        else:
            metadata = {}
            metadata['tree'] = (-0.2,0,0,26)
            metadata['orangeTrue'] = (-0.5,-0,1.14)
        trajdata = pickle.load(open(folder_name+"/traj_data.pickle","rb"))
        env_name = None#'unity/env_v9'
        env = UnityEnvironment(file_name=env_name,worker_id=0,seed=0)
        tree = np.array(metadata['tree'])
        orange = np.array(metadata['orangeTrue'])
        camName = None
        print(len(trajdata))
        for ii,state in enumerate(trajdata):
            #pos = np.array(state[0])
            #rot = R.from_dcm(state[1]).as_euler('zyz')
            #x = np.hstack((pos,rot))
            if camName is None:
                (camName,envName,orange,occ_frac,dummy) = setUpEnv(env,state,tree,orange)
            camAct = makeCamAct(state)
            (im_arr,ext_arr) = unity_image(env,camAct,camName,envName)
            im_arr = img.fromarray((im_arr*255).astype('uint8'))
            im_arr.save(folder_name + 'sim_image' + str(ii) +  '.png')
            ext_arr = img.fromarray((ext_arr*255).astype('uint8'))
            ext_arr.save(folder_name + 'ext_image' + str(ii) +  '.png')
            sim_img_list.append(im_arr)
            ext_img_list.append(ext_arr)
            #print(ii)
        #print(state)
        env.close()
        if len(sim_img_list) > 0:
            sim_img_list[0].save(folder_name + "/sim.gif",save_all=True,append_images=sim_img_list[1:],duration=10, loop=0,optimize=True,quality=100)
        if len(ext_img_list) > 0:
            ext_img_list[0].save(folder_name + "/ext.gif",save_all=True,append_images=ext_img_list[1:],duration=10, loop=0,optimize=True,quality=100)

if __name__ == "__main__":
    main()
