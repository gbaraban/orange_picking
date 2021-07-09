import numpy as np
import os
from PIL import Image

total_procs = 16


fnum = 0

while os.path.exists("mean" + str(fnum)):
	fnum += 1

os.mkdir("mean" + str(fnum))

i = 0

mean_img = None
w = 640
h = 380


base_dir = "../Run24/"
all_dirs = os.listdir(base_dir)
all_dirs.sort()
each = int(len(all_dirs)/total_procs)

if fnum == total_procs - 1:
	proc_dir = all_dirs[each*fnum:]
else:
	proc_dir = all_dirs[each*fnum:each*(fnum+1)]

print(len(proc_dir))
for dirs in proc_dir:
	print(dirs)
	for file in os.listdir(base_dir + "/" + dirs):
		if file.endswith(".png") and file.startswith("image"):
			img = Image.open(base_dir + "/" + dirs + "/" + file).resize((w,h))
			img = np.array(img.getdata()).reshape(img.size[1],img.size[0],3)
			img = img[:,:,0:3]/255.0 #Cut out alpha

			if mean_img is None:
				mean_img = img
			else:
				mean_img = np.multiply(mean_img, float(i)/float(i+1)) + np.multiply(img, 1/float(i+1))

			i += 1

np.save("mean"+str(fnum)+"/mean"+str(i) + ".npy", mean_img)

