import numpy as np
from PIL import Image as img
import os
import time
import sys

args = sys.argv

dirs = args[1:]

#mean_img = np.load("mean_imgv2_data_Run24.npy")
mean_img = np.load("v1_flights/mean_imgv2_data_real_world_traj_bag.npy")
depth_mean_img = None #np.load("depth_mean_imgv2_data_real_world_traj_bag.npy")

data_loc = "v1_flights/v1/real_world_traj_bag/"
np_data_loc = "/mnt/corsair/gabe/" + data_loc.rstrip("/") + "_np"
if not os.path.exists(np_data_loc):
    os.makedirs(np_data_loc)


w = 640
h = 480
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
            if im.endswith(".png") and im.startswith("depth") and False:
                im_name = im.rstrip(".png")

                temp_image = img.open(dir_path + "/" + im).resize((w,h))
                temp_image = np.array(temp_image.getdata()).reshape(temp_image.size[1],temp_image.size[0],1)
                temp_image = temp_image[:,:,0]/255.0

                temp_image = np.array(temp_image) - depth_mean_img

                np.save(npdir_path + "/" + im_name + ".npy", temp_image)
        print("Time taken: ", time.time() - tmid)
print("Time taken: ", time.time()-t)


