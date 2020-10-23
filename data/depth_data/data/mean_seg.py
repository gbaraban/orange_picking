import os
import numpy as np

mean = np.load("mean_seg.npy")
count = 0

loc = "./seg_mask/"
floc = "./seg_mask_np/"
for file in os.listdir(loc):
        os.makedirs(floc+file)
	for mask in os.listdir(loc+file):
		ma = np.load(loc + file + "/" + mask)
		mean_sub = ma - mean
		np.save(floc+file+"/"+mask, mean_sub)
