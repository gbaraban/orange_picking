import os
import numpy as np

def main(loc):
	i = 0
	mean_img = None

	dirs = os.listdir(loc)
	for dir in dirs:
		if dir.startswith("mean"):
			np_file = os.listdir(loc + "/" + dir)[0]
			size = int(np_file.strip("mean").strip(".npy"))
			img = np.load(loc + "/" + dir + "/" + np_file)
			if mean_img is None:
				mean_img = img
			else:
				mean_img = np.multiply(mean_img, float(i)/float(i+size)) + np.multiply(img, size/float(i+size))

			i += size

	np.save("mean.npy", mean_img)

if __name__ == "__main__":
	loc = "./"
	main(loc)
