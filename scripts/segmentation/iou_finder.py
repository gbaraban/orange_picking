import numpy as np
import torch
import time, os
import cv2
from segmentnetarch import *
import PIL.Image as Image

gpu = 0

def IoU(actual, predicted):
    return np.sum(actual*predicted).astype(np.float32)/np.sum(actual+predicted).astype(np.float32)

seg_loc = "data/combined_data4/seg_mask/"
img_loc = "data/combined_data4/real_world_traj_bag_np/"

segload =  "/home/gabe/ws/ros_ws/src/orange_picking/model/segmentation/logs/variable_log/2021-01-31_13-25-31/model_seg145.pth.tar" 
segmodel = SegmentationNet()

if os.path.isfile(segload):
        if not gpu is None:
                checkpoint = torch.load(segload,map_location=torch.device('cuda'))
        else:
                checkpoint = torch.load(segload)
        segmodel.load_state_dict(checkpoint)
        segmodel.eval()
        print("Loaded Model: ", segload)
else:
        print("No checkpoint found at: ", segload)
        exit(0)


if not gpu is None:
	device = torch.device('cuda:'+str(gpu))
	segmodel = segmodel.to(device)
else:
	device = torch.device('cpu')
	segmodel = segmodel.to(device)


avg_iou = 0.0
ctr = 0
h = 480
w = 640

for bag in os.listdir(seg_loc):
    print(bag)
    dr = seg_loc+bag + "/"
    img_dr = img_loc + bag + "/"
    for img in os.listdir(dr):
        image = np.load(img_dr+"image" +img)
        seg = np.load(dr+img)
        image = image.transpose(2,0,1)
        image = image.reshape((1,3,h,w))
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.to(device,dtype=torch.float)
        seglogits = segmodel(image_tensor)
        seglogits = seglogits.view(-1,2,segmodel.h,segmodel.w)
        segimages = (torch.max(seglogits, 1).indices).to('cpu')
        seg_np = np.array(segimages[0,:,:])
        pub_np = (255 * seg_np).astype(np.uint8).reshape((h, w)).astype(np.bool)
        #pub_np = Image.fromarray(pub_np)
        #pub_np.save("test.png")
        pub_np2 = (255 * seg).astype(np.uint8).reshape((h, w)).astype(np.bool)
        #pub_np = Image.fromarray(pub_np)
        #pub_np.save("test2.png")
        
        iou = IoU(pub_np, pub_np2)
        ctr += 1
        avg_iou += (iou - avg_iou)/float(ctr)
    print(ctr, avg_iou)
print(ctr, avg_iou)