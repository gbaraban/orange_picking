from orangesimulation import *
from orangeimages import *
import numpy as np
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as Image
from scipy.spatial.transform import Rotation as R
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('traj', type=int, help='model to load')
args = parser.parse_args()

traj = args.traj #7 # 0 - 7
traj_folder = ["Sim1_100_3.0", "Sim1_150_3.0", "Sim3_150_3.0",
               "Sim3_150_3.0", "Sim3_150_3.0", "Sim4_750_15.0",
               "Sim3_200_3.0", "Sim1_250_3.0_no_orange"]
trial_n = [3, 2, 1, 2, 9, 6, 0, 11]
steps_n = [100, 150, 150, 150, 150, 750, 200, 250]
norange = [False, False, False, False, False, False, False, True]

print(len(traj_folder), len(trial_n), len(steps_n), len(norange))

loc = "./animation/" + traj_folder[traj] + "/trial" + str(trial_n[traj]) + "/"
trial = trial_n[traj]
steps = steps_n[traj]
no_orange = norange[traj]
#env = UnityEnvironment(file_name=None)#'unity/env_v5',seed=0)
#Iterate simulations
run_num = 0
globalfolder = 'data/animation/Sim' + str(run_num) + '/'
while os.path.exists(globalfolder):
    run_num += 1
    globalfolder = 'data/animation/Sim' + str(run_num) + '/'

os.makedirs(globalfolder)
os.makedirs(globalfolder+"/images/")
os.makedirs(globalfolder+"/external/")
if not os.path.exists("data/animation/gifs"):
    os.makedirs("data/animation/gifs/")

#meta = pickle.load(open(loc+"metadata.pickle", "rb"))
#tree = meta["tree"]
#orange = meta["orangeTrue"]
metadata = pickle.load(open("./meta/trial" + str(trial) + ".pickle", "rb"))
envAct = metadata["env"]
treeAct = metadata["tree"]
orangeAct = metadata["orange"]
if no_orange:
    orangeAct[0,0] = 0
    orangeAct[0,1] = -1.5
    orangeAct[0,2] = 0

print(orangeAct)
camAct = metadata["cam"]

x_traj = pickle.load(open(loc + "traj_data.pickle", "rb"))
print("loaded")
env = UnityEnvironment(None)
print("env loaded")
env.reset()
print("reset")
#treeAct = np.array([np.hstack((gcopVecToUnity(treePos[0:3]),treePos[3],1))])
#envAct = np.array([np.hstack((envAct,1))])
names = env.get_behavior_names()

camName = ""
envName = ""
treeName = ""
orangeName = ""

for n in names:
    print(n)
    if "Tree" in n:
        treeName = n
        continue
    if "Orange" in n:
        orangeName = n
        continue
    if "Environment" in n:
        envName = n
        continue
    if "Cam" in n:
        camName = n
        continue

env.set_actions(envName,envAct)
env.step()
env.set_actions(treeName,treeAct)
env.step()
env.set_actions(orangeName,orangeAct)
env.step()
#camAct = makeCamAct(x0)
env.set_actions(camName,camAct)
env.step()

save_path = globalfolder

step_size = int(len(x_traj)/steps)
l = 0
for i in range(0, len(x_traj), step_size):
    #print(i)
    x = x_traj[i]
    camAct = makeCamAct(x)
    (image_arr,ext_image_arr) = unity_image(env,camAct,camName,envName)
    save_image_array(image_arr[:,:,0:3],save_path,"sim_image"+str(l)) #TODO: Use concat a$
    save_image_array(ext_image_arr,save_path,"ext_image"+str(l)) #TODO
    l += 1

env.close()

img_dir = globalfolder
im = []
for iname in range(0,1000):
    #print(img_dir + "sim_image" + str(iname) + ".png")
    #print(os.path.isfile(img_dir + "sim_image" + str(iname) + ".png"))
    if os.path.isfile(img_dir + "sim_image" + str(iname) + ".png"):
        img = Image.open(img_dir + "sim_image" + str(iname) + ".png")
        img = img.resize((336,200))
        im.append(img)
    else:
        break

if len(im) > 0:
    temp_name = "data/animation/gifs/" + loc.strip(".").strip("/").replace("/","_")
    im[0].save(temp_name + '_sim.gif',save_all=True, append_images=im[1:], duration=10, loop=0,optimize=True, quality=100)

print(len(im))
im = []
for iname in range(0,1000):
    if os.path.isfile(img_dir + "ext_image" + str(iname) + ".png"):
        img = Image.open(img_dir + "ext_image" + str(iname) + ".png")
        img = img.resize((336,200))
        im.append(img)
    else:
        break

if len(im) > 0:
    temp_name ="data/animation/gifs/" + loc.strip(".").strip("/").replace("/","_") 
    im[0].save(temp_name + '_ext.gif',save_all=True, append_images=im[1:], duration=10, loop=0,optimize=True, quality=100)

