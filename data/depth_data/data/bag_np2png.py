import os
import PIL.Image as Image
import numpy as np

floc = "seg_mask_real/"
tloc = "seg_mask_real_png/"

for bag in os.listdir(floc):
	os.makedirs(tloc + "/" +  bag)
	for img in os.listdir(floc + "/" + bag):
		im_np = 255 * np.load(floc + "/" + bag + "/" + img)
		im = Image.fromarray(im_np.astype(np.uint8))
		#print(img[:-4])
		#exit(0)
		im.save(tloc + "/" + bag + "/image" + img[:-4] + ".png")
