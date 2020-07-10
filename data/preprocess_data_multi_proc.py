import numpy as np
from PIL import Image as img
import os
import time
import sys

args = sys.argv

dirs = args[1:]

mean_img = np.load("mean_imgv2_data_Run20.npy")

data_loc = "Run20"
np_data_loc = data_loc.rstrip("/") + "_np"
if not os.path.exists(np_data_loc):
    os.mkdir(np_data_loc)


w = 640
h = 380
t = time.time()
#dirs = os.listdir(data_loc)
for d in dirs:#files:
    dir_path = data_loc + "/" + d
    if os.path.isdir(dir_path):
        npdir_path = np_data_loc + "/" + d
        os.mkdir(npdir_path)
        print(dir_path)
        tmid = time.time()
        for im in os.listdir(dir_path):
            if im.endswith(".png") and im.startswith("image"):
                im_name = im.rstrip(".png")

                temp_image = img.open(dir_path + "/" + im).resize((w,h))
                temp_image = np.array(temp_image.getdata()).reshape(temp_image.size[1],temp_image.size[0],3)
                temp_image = temp_image[:,:,0:3]/255.0

                temp_image = np.array(temp_image) - mean_img

                np.save(npdir_path + "/" + im_name + ".npy", temp_image)
        print("Time taken: ", time.time() - tmid)
print("Time taken: ", time.time()-t)


