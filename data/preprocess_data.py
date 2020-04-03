import numpy as np
from PIL import Image as img
import os
import time

mean_img = np.load("mean_img_data_temp.npy")

data_loc = "temp"
np_data_loc = data_loc.rstrip("/") + "_np"
if not os.path.exists(np_data_loc):
    os.mkdir(np_data_loc)

#files = ['bag0', 'bag1',  'bag2',  'bag3',  'bag6']

#files = [["trial11", "trial110", "trial111",  "trial112",  "trial113"], ["trial114", "trial115", "trial116", "trial117", "trial118", "trial119"]]
#files = [['trial0', 'trial107', 'trial116', 'trial125', 'trial134', 'trial143', 'trial16', 'trial25', 'trial34', 'trial43', 'trial52', 'trial61', 'trial70', 'trial8', 'trial89', 'trial98'], 
#    ['trial1', 'trial108', 'trial117', 'trial126', 'trial135', 'trial144', 'trial17', 'trial26', 'trial35', 'trial44', 'trial53', 'trial62', 'trial71', 'trial80', 'trial9', 'trial99'], 
#    ['trial10', 'trial109', 'trial118', 'trial127', 'trial136', 'trial145', 'trial18', 'trial27', 'trial36', 'trial45', 'trial54', 'trial63', 'trial72', 'trial81', 'trial90'], 
#    ['trial100', 'trial11', 'trial119', 'trial128', 'trial137', 'trial146', 'trial19', 'trial28', 'trial37', 'trial46', 'trial55', 'trial64', 'trial73', 'trial82', 'trial91'], 
#    ['trial101', 'trial110', 'trial12', 'trial129', 'trial138', 'trial147', 'trial2', 'trial29', 'trial38', 'trial47', 'trial56', 'trial65', 'trial74', 'trial83', 'trial92'], 
#    ['trial102', 'trial111', 'trial120', 'trial13', 'trial139', 'trial148', 'trial20', 'trial3', 'trial39', 'trial48', 'trial57', 'trial66', 'trial75', 'trial84', 'trial93'], 
#    ['trial103', 'trial112', 'trial121', 'trial130', 'trial14', 'trial149', 'trial21', 'trial30', 'trial4', 'trial49', 'trial58', 'trial67', 'trial76', 'trial85', 'trial94'], 
#    ['trial104', 'trial113', 'trial122', 'trial131', 'trial140', 'trial15', 'trial22', 'trial31', 'trial40', 'trial5', 'trial59', 'trial68', 'trial77', 'trial86', 'trial95'], 
#    ['trial105', 'trial114', 'trial123', 'trial132', 'trial141', 'trial150', 'trial23', 'trial32', 'trial41', 'trial50', 'trial6', 'trial69', 'trial78', 'trial87', 'trial96'], 
#    ['trial106', 'trial115', 'trial124', 'trial133', 'trial142', 'trial151', 'trial24', 'trial33', 'trial42', 'trial51', 'trial60', 'trial7', 'trial79', 'trial88', 'trial97']]

#f = 9
w = 640
h = 380
t = time.time()
dirs = os.listdir(data_loc)
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


