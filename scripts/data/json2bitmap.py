import os
import numpy as np
from PIL import Image
import json
import cv2
import zlib
import base64
import shutil
import time

w, h = 640, 480
asx, asy, asx2, asy2 = 0, 0, 0, 0
img_loc = "./real_world_traj_bag/"
seg_img_loc = "./real_world_traj_bag_seg/"

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def draw_figs(figs, debug, pNo, vidName):
    seg_image = None
    #print(len(figs))
    for i, fig in enumerate(figs):
        geometry = fig[u'geometry']
        bitmap = geometry[u'bitmap']
        data = bitmap[u'data']
        origin = bitmap[u'origin']
        mask = base64_2_mask(data).astype(np.uint8)
        #print(mask.shape)
        #mask = 255*mask
        #print(mask)
        #print(origin)
        #print(origin[1],origin[1]+mask.shape[1], origin[0],origin[0]+mask.shape[0])
        seg_image = np.zeros((w,h), dtype=np.uint8)
        #print(seg_image[origin[1]:origin[1]+mask.shape[1], origin[0]:origin[0]+mask.shape[0]])
        mask = np.transpose(mask)
        #print(mask_t.shape)
        #max_x = int(min((origin[1]+mask.shape[1]), h))
        #max_y = int(min((origin[0]+mask.shape[0]), w))

        global asx, asy, asx2, asy2
        asx = max(asx, origin[0]+mask.shape[0])
        asy = max(asy, origin[1]+mask.shape[1])
        asx2 = max(asx2, origin[0]+mask.shape[1])
        asy2 = max(asy2, origin[1]+mask.shape[0])

        #print(mask.shape[1], max_x, h, mask.shape[0], max_y, w)
        seg_image[origin[0]:origin[0]+mask.shape[0], origin[1]:origin[1]+mask.shape[1]] = mask
        #end = seg_image[origin[1]:max_x, origin[0]:max_y]
        #seg_image[origin[1]:max_x, origin[0]:max_y] = mask_t[:end.shape[0], :end.shape[1]]
        seg_image = np.transpose(seg_image)
        #cv2.imwrite("./image.png", seg_image)
        if debug:
            #cv2.imshow('image',seg_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            iloc = img_loc + vidName.strip("_mp4") + "/image" + str(pNo) + ".png"
            iimg = cv2.imread(iloc)
            iimg = cv2.cvtColor(iimg, cv2.COLOR_BGR2RGB)
            wloc = seg_img_loc + vidName.strip("4").strip("mp")
            if not os.path.isdir(wloc):
                os.makedirs(wloc)
            seg_image_copy = 255*seg_image.copy() 
            seg_image_copy = cv2.cvtColor(seg_image_copy, cv2.COLOR_GRAY2RGB)
            final = cv2.bitwise_or(seg_image_copy, iimg)
            cv2.imwrite(wloc + "/image" + str(pNo) + ".png" , final)
        #exit(0)
        if i != 0:
            print("ISSUUEESSS!!!", pNo)
            #exit(0)

    if seg_image is None:
        print("ISSUESS NO SEG_IMAGE!")
        exit(0)

    return seg_image

def infer_frames(frames, vid, vidName):
    for frame in frames:
        #print(frame)
        pNo = frame[u'index']
        #print(pNo)
        debug = False
        figs = frame[u'figures']
        seg_mask = draw_figs(figs, debug, pNo, vidName)
        np.save(vid + str(pNo) + ".npy", seg_mask)


def read_json(fname, loc):
    f = open(fname, "r")
    data = json.loads(f.read())
    print( data[u'videoName'])
    vidName = data[u'videoName'].replace(".", "")
    vidName = vidName.strip("4").strip("mp")
    vid = str(loc + vidName + "/")
    print(fname, vid)
    os.makedirs(vid)
    #exit(0)
    frames = data[u'frames']
    infer_frames(frames, vid, vidName)
    f.close()


def parse_videos(fnames, loc):
    for fname in fnames:
        read_json(fname, loc)


def get_json_vid_name(path):
    fnames = []
    files = os.listdir(path)
    for file in files:
        if file.endswith("json"):
            fnames.append(path + file)

    return fnames

if __name__ == "__main__":
    path = "./json/"
    loc = "./seg_mask/"
    if os.path.isdir(loc):
        shutil.rmtree(loc)
    os.makedirs(loc)
    fnames = get_json_vid_name(path)
    parse_videos(fnames, loc)
    #read_json("json/real_world_depth_dataset_orange_bag1.json")
    print(asx, asy)
    print(asx2, asy2)
