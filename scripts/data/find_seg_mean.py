import os
import numpy as np

mean = np.zeros((480, 640))
count = 0

loc = "./seg_mask/"
for file in os.listdir(loc):
	for mask in os.listdir(loc+file):
		ma = np.load(loc + file + "/" + mask)
		count += 1
		mean = mean + ((ma-mean)/count)

np.save("mean_seg.npy", mean)
