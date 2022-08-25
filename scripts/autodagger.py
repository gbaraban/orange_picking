from orangesimulation import *
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
import itertools
import random
from orangedagger import main_function

"""
	Takes manual inputs of different dagger configs and tries their permutation combination to find the best combination

	read_csv: reads best NN model config
	get_variety/get_selective_variety: determines simulation config combinations
"""

def read_csv(location):
	"""
		Reads config of the best models used for running the NN model
	"""
	fopen = open(location, "r")
	cols = []
	data = {}
	for i,line in enumerate(fopen):
		line = line.strip("\n").strip()
		line = line.split(",")
		if i == 0:
			for l in line:
				cols.append(l.strip().strip("\t"))
		else:
			fname = ""
			temp_data = {}
			for j, l in enumerate(line):
				if j == 0:
					fname = l
				else:
					temp_data[cols[j]] = l

			data[fname] = temp_data

	return data


def generate_gif(fname, loc="/home/gabe/ws/ros_ws/src/orange_picking/data/simulation/", gifs="/home/gabe/ws/ros_ws/src/orange_picking/gifs/"):
	"""
		Function to generate gifs for a run of simulation
	"""
	for dir in os.listdir(location):
		for tname in os.listdir(loc + fname):
			if tname.startswith("trial"):
				img_dir = loc + fname + "/" + tname + "/"
				im = []
				for iname in range(0,500):
					if os.path.exists(img_dir + iname):
						img = Image.open(img_dir + "sim_image" + str(iname) + ".png")
						img = img.resize((336,200))
						im.append(img)
					else:
						break

				if len(im) > 0:
					im[0].save(gifs + str(fname)  + '_' + str(tname) + '.gif',save_all=True, append_images=im[1:], duration=10, loop=0,optimize=True, quality=100)

def get_variety():
	"""
		Function to find different combinations of configs to run dagger. Variable "labels" is the set of all configs
	"""
	var = {}
	var["iters"] = [10]
	var["outputs"] = [6]
	var["steps"] = [100, 250, 500]
	var["hz"] = [1,3,5,10,12,15,20,25,30]
	#var["physics"] = [10,50,100]
	labels = []
	s = []
	num = 1
	for key in var.keys():
		print(key)
		labels.append(key)
		num *= len(var[key])
		s.append(var[key])

	variety = list(itertools.product(*s))
	print(num)
	print(labels)
	return labels, variety

def get_selective_variety():
	"""
		Function to input manual configs for given set of labels to run dagger. Variable "labels" is the set of all configs
	"""
	variety = [[10, 6, 100, 3], [10, 6, 100, 5], [10, 6, 150, 5], [10, 6, 100, 10], [10, 6, 150, 10], [10, 6, 250, 10], [10, 6, 250, 12], [10, 6, 250, 15], [10, 6, 350, 15], [10, 6, 250, 20], [10, 6, 350, 20], [10, 6, 500, 15], [10, 6, 500, 20], [10, 6, 500, 25]]
	labels = ["iters", "outputs", "steps", "hz"]
	return labels, variety

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, default="/home/gabe/ws/ros_ws/src/orange_picking/useful_models.csv", help='random seed')
    parser.add_argument('--times', type=int, default=5, help='approx number of unity threads running')
    parser.add_argument('--explore', type=int, default=0, help="selective variety or explorative (i.e. get_variety or get_selective_variety)")
    args = parser.parse_args()

    data = read_csv(args.loc)
    i = 1
    wid = 0
    if args.explore == 1:
        labels, vars = get_variety()
    else:
        labels, vars = get_selective_variety()
    #print(data)
    #exit()
    gpu = 0
    x = 0
    for k in data.keys(): # for each runnable NN config
        d = data[k]
        model_loc = d["model_loc"] # model location, i.e. NN weights
        model_name = d["model"] 
        num_img = int(d["num_images"])
        resnet18 = False
        if d["resnet"] == "18":
            resnet18 = True
        for i in range(args.times): # number of times dagger is run with a particular config
            random.shuffle(vars)
            #proc = "python3 scripts/orangesimulation.py --gpu 0 " + str(d["model_loc"]) + str(d["model"]) + " --name " + str(k) + " --num_images " + str(d["num_images"]) + "  "
            #if d["resnet"] == "18":
            #    proc +=  " --resnet18 1 "
            for var in vars:
                #t_proc = proc
                #for n, v in enumerate(var):
                #    t_proc += " --" + str(labels[n]) + " " + str(v) + " "
                #    if labels[n] == "iters":
                #        wid += int(v)
                
				# reading simulation config
				outputs = vars[1]
                steps = vars[2]
                hz = vars[3]

				# calling main_function to run dagger
                new_model = main_function(model_loc+"/"+model_name,gpu,x,num_img,resnet18=resnet18,steps=steps,hz=hz,physics=hz,worker_id=x,batch=128)
                x += 1
                #t_proc += " --worker_id " + str(int(wid))
                #print(t_proc)
                #exit(0)
                #if i%args.j == 0:
                #    #pass
                #    os.system(t_proc)
                #else:
                #    #pass
                #    os.system(t_proc + " &" )
                #exit(0)
                #i += 1


if __name__ == '__main__':
  main()

