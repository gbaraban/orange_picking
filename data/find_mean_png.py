import numpy as np
import os
from PIL import Image

i = 0

mean_img = None
w = 640
h = 380


base_dir = "Run19"
for dirs in os.listdir(base_dir):
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

np.save("mean_imgv2_data_Run19.npy", mean_img)

