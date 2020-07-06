import gcophrotor
import matplotlib.pyplot as plt
import PIL.Image as img
import numpy as np
import sys
import os
import pickle
from mlagents_envs.environment import UnityEnvironment
from scipy.spatial.transform import Rotation as R
import argparse
from plotting.parsetrajfile import *
from orangesimulation import *

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', help='env file')
	parser.add_argument('--loop', type=bool, help='make infinite data')
	parser.add_argument('--worker_id', type=int, default=0, help='worker ID, diff if you want multiple Unity envs')
	parser.add_argument('--seed', type=int, default=0, help='seed vals')
	#parser.add_argument('--plotonly', type=bool, help='skip image generation') TODO: maybe add plot only mode
	args = parser.parse_args()

	env_name = args.env + 'unity/env_v5' #v4 / v5 for lambda
	#train_mode = True
	env = UnityEnvironment(file_name=env_name, worker_id=args.worker_id, seed=args.seed)
	env.reset()

	run_num = 0
	base_folder = './data/data_collection/'
	if not os.path.exists(base_folder):
		os.makedirs(base_folder)
	data_folder = base_folder + 'Run' + str(run_num) + "/"
	while os.path.exists(data_folder):
		run_num += 1
		data_folder = base_folder + "Run" + str(run_num) + "/"

	exp = 0
	trials = 20
	while (exp <= trials) or args.loop:
		fname = "trial" + str(exp) + "/"
		os.makedirs(data_folder + fname)

		overviewname = "traj_plot" + str(exp)
		picturename = "image"
		suffix = ".png"

		(x, camName, orange,tree) = shuffleEnv(env,future_version=True) #add plotonly

		N = 500
		tf = 15
		ref_traj = run_gcop(x, tree, orange, tf=tf ,N=N, save_path=data_folder + fname)

		ts = np.linspace(0,tf,N+1)
		make_full_plots(ts,ref_traj,orange,tree,saveFolder=data_folder + fname)

		with open(data_folder + fname + 'trajdata.pickle','wb') as f:
			pickle.dump(ref_traj,f,pickle.HIGHEST_PROTOCOL)

		ctr = 0
		for state in ref_traj:
			#print('State ' + str(ctr))
			#Unpack state
			#pos = state[0]
			#rot = np.array(state[1])
			#rot = rot.T
			#print(pos)
			camAct = makeCamAct(state)
			image = unity_image(env, camAct, camName)
			image = img.fromarray(np.uint8(image*255))
			image.save(data_folder + fname + picturename + str(ctr) + suffix)
			ctr += 1

		exp += 1
