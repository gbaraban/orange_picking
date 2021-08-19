import os
import numpy as np
import PIL.Image as Image
import gzip

seg = True
depth = False

src = "../Run24/"
seg_dest = "../Run24_seg/"
depth_dest = "../Run24_depth/"

orange = [255, 0, 0]
tree = [0, 0, 0]

mean_seg = None

ctr = 0

if seg:
	os.makedirs(seg_dest)

if depth:
	os.makedirs(depth_dest)

for i, dir in enumerate(os.listdir(src)):
	if seg:
		os.makedirs(seg_dest+dir)
	if depth:
		os.makedirs(depth_dest+dir)

	for img in os.listdir(src+dir):
		if img.startswith("seg") and img.endswith("png") and seg:
			name = img.replace(".","").lstrip("seg_image").rstrip().rstrip("png")
			image = Image.open(src+dir+"/"+img)
			image = np.array(image)
			orange_loc = np.array(np.where(np.all(image == orange, axis=-1)))
			tree_loc = np.array(np.where(np.all(image == tree, axis=-1)))
			seg_image = np.zeros((image.shape[0], image.shape[1], 2))
			seg_image[orange_loc[0,:], orange_loc[1,:], 0] = 1.
			seg_image[tree_loc[0,:], tree_loc[1,:], 1] = 1.
			if mean_seg is None:
				mean_seg = seg_image.copy()
				ctr = 1
			else:
				mean_seg += (seg_image.copy() - mean_seg)/float(ctr)
				ctr += 1
			f = gzip.GzipFile(seg_dest+dir+"/image" + name + ".npy.gz", "w")
			np.save(file=f, arr=seg_image)
			#print(img, name, dir)
			f.close()

		if img.startswith("depth") and img.endswith("png") and depth:
			pass
	print(i, " out of ", len(os.listdir(src)))

np.save("mean.npy", mean_seg)
