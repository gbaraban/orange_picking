import os
import numpy as np

np.random.seed(0)

test_perc = 0.1

base = "/home/gabe/ws/ros_ws/src/orange_picking/data/"
src = base + "/data_collection_sim/data/"
dest = base + "/Run24/"
test = base + "/Run24_test/"

os.makedirs(dest)
os.makedirs(test)

ctr = 0
test_ctr = 0

dirs = os.listdir(src)
dirs.sort()
for dir in dirs:
	l = len(os.listdir(src+dir))
	data = [""] * l
	for trial in os.listdir(src+dir):
		loc = int(trial.split("_")[0].strip(" ").strip("trial"))
		occ = float(trial.split("_")[1].strip(" "))
		data[loc] = occ

	for i, d in enumerate(data):
		if d == "":
			print("Problem in ", dir, data)

		rand = np.random.random()

		if rand > test_perc:
			os.system("ln -s " + src + dir + "/trial" + str(i) + "_" + str(d) + "   " + dest + "trial" + str(ctr) + "_" + str(d) + "" )
			ctr += 1
		else:
			os.system("ln -s " + src + dir + "/trial" + str(i) + "_" + str(d) + "   " + test + "trial" + str(test_ctr) + "_" + str(d) + "" )
			test_ctr += 1
