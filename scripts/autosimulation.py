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


def read_csv(location):
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
	var = {}
	var["iters"] = [10]
	var["outputs"] = [6]
	var["steps"] = [500] #[100, 250, 500]
	var["hz"] = [25, 30] #[1,3,5,10,12,15,20]
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
	return labels, variety

def get_selective_variety():
	variety = [{10, 6, 100, 3}]
	labels = ["iters", "outputs", "steps", "hz"]
	return variety, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, default="/home/gabe/ws/ros_ws/src/orange_picking/useful_models.csv", help='random seed')
    parser.add_argument('--j', type=int, default=4, help='approx number of unity threads running')
    parser.add_argument('--explore', type=int, default=1, help="selective variety or explorative")
    args = parser.parse_args()

    data = read_csv(args.loc)
    i = 1
    wid = 0
    if args.explore == 1:
        labels, vars = get_variety()
    else:
        labels, vars = get_selective_variety()
    #print(data)
    for k in data.keys():
        d = data[k]
        print(d)
        proc = "python3 scripts/orangesimulation.py --gpu 0 " + str(d["model_loc"]) + str(d["model"]) + " --name " + str(k) + " --num_images " + str(d["num_images"]) + "  "
        if d["resnet"] == "18":
            proc +=  " --resnet18 1 "

        for var in vars:
            t_proc = proc
            for n, v in enumerate(var):
                t_proc += " --" + str(labels[n]) + " " + str(v) + " "
                if labels[n] == "iters":
                    wid += int(v)

            t_proc += " --worker_id " + str(int(wid))
            print(t_proc)

            if i%args.j == 0:
                os.system(t_proc)
            else:
                os.system(t_proc + " &" )
            #exit(0)
            i += 1


if __name__ == '__main__':
  main()

