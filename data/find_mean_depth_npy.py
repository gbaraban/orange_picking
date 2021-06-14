import cv2
import numpy as np
import os

bag_dirs = None
def recursive_bag_search(loc):
	added = False
	for file in os.listdir(loc):
		if file.startswith("depth_image") and file.endswith("npy"):
			if not added:
				added = True
				global bag_dirs
				if bag_dirs is None:
					bag_dirs = {loc}
				else:
					bag_dirs.add(loc)
		elif os.path.isdir(loc + "/" + file):
			recursive_bag_search(loc + "/" + file)


i = 0

mean_img = None

base_dir = "orange_tracking_data/real_world_traj_bag/"


recursive_bag_search(base_dir)
print(len(bag_dirs))
min_val, max_val = np.inf, -np.inf

for x, dir in enumerate(bag_dirs):
	print(x)
	for file in os.listdir(dir):
		if file.startswith("depth_image") and file.endswith(".npy"):
			img = np.load(dir + "/" + file)
			min_val = np.min((min_val, np.min(img)))
			max_val = np.max((max_val, np.max(img)))
			if mean_img is None:
				mean_img = img
			else:
				mean_img = np.multiply(mean_img, float(i)/float(i+1)) + np.multiply(img, 1/float(i+1))
			i += 1

print(min_val, max_val)
np.save("mean_depth_imgv2.npy", mean_img)
cv2.imwrite("depth_mean.png", mean_img)
