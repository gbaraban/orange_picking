import numpy as np
import os

i = 0

mean_img = None

base_dir = "bag_v3_np"
for dirs in os.listdir(base_dir):
	print(dirs)
	for file in os.listdir(base_dir + "/" + dirs):
		if file.endswith(".npy"):
			img = np.load(base_dir + "/" + dirs + "/" + file)
			if mean_img is None:
				mean_img = img
			else:
				mean_img = np.multiply(mean_img, float(i)/float(i+1)) + np.multiply(img, 1/float(i+1))

			i += 1

np.save("mean_img_bag_v3_np.npy", mean_img)

