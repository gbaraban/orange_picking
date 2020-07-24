import os
import numpy as np

if __name__ == "__main__":
	loc = "./"
	run_dir = os.listdir(loc)
	hist = np.zeros(10)
	for run in run_dir:
		if run.startswith("Run"):
			trial_dir = os.listdir(loc + "/" + run + "/")
			#hist = np.zeros(10)
			for trial in trial_dir:
				if trial.startswith("trial"):
					_,occ = trial.split("_")
					bin = int(float(occ)*10)
					if bin == 10:
						bin = 9

					hist[bin] += 1

	print(hist)
