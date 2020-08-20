import os
import numpy as np
import random
import shutil


def main(loc):
	trial_dirs = os.listdir(loc)
	print(len(trial_dirs))
	test_perc = 0.15

	random.seed(76)

	test_samples = int(test_perc * len(trial_dirs))

	trial_test = random.sample(trial_dirs, k=test_samples)
	print(len(trial_test))

	trial_train = []
	for t_dir in trial_dirs:
		if t_dir not in trial_test:
			trial_train.append(t_dir)

	print(len(trial_train))


	dest_train = "./Run23/"
	dest_test = "./Run23_test/"

	ctr = 0
	for t_dir in trial_train:
		_, occ = t_dir.split("_")
		shutil.move(loc + t_dir,dest_train+ "trial" + str(ctr) + "_" + occ)
		ctr += 1

	ctr = 0
	for t_dir in trial_test:
		_, occ = t_dir.split("_")
		shutil.move(loc + t_dir, dest_test + "trial" + str(ctr) + "_" + occ)
		ctr += 1


if __name__ == "__main__":
	loc = "./Run23_final/"
	main(loc)
