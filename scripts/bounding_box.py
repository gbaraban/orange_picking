import PIL.Image as Image
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from datetime import datetime
import argparse
import PIL.Image as img
import signal
import sys
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from customTransforms import *
#from orangenetarch import *
import pickle
#from customRealDatasets import *
#print("summ")
#from torch.utils.tensorboard import SummaryWriter 
#print("gc")
import gc
import random
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
from numpy.linalg import norm
#from customRealDatasetsOrientation import OrangeSimDataSet, SubSet
from customTransforms import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data folder')
    parser.add_argument('--batch_size', type=int, default=65, help='batch size')
    parser.add_argument('--num_images', type=int, default=1, help='number of input images')
    parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
    parser.add_argument('-j', type=int, default=4, help='number of loader workers')
    parser.add_argument('--custom', type=str, default="", help='custom parser: Run18/no_parse')
    parser.add_argument('--input_size',type=float,default=1,help='input size change')
    parser.add_argument('--depth',type=bool,default=False,help='use depth channel')
    parser.add_argument('--seg',type=bool,default=False,help='use segmentation channel')
    parser.add_argument('--temp_seg',type=bool,default=False,help='use segmentation channel')
    parser.add_argument('--seg_only',type=bool,default=False,help='use segmentation channel')
    parser.add_argument('--spherical',type=bool,default=False,help='use spherical coords')
    parser.add_argument('--real',type=bool,default=False,help='real world or simulation')
    args = parser.parse_args()

    args.min = [(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.75,-0.75,-0.5,-np.pi,-np.pi,-np.pi),(-1.0,-1.0,-0.75,-np.pi,-np.pi,-np.pi)]
    args.max = [(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.5,1.0,0.5,np.pi,np.pi,np.pi),(2.0,1.0,0.75,np.pi,np.pi,np.pi)]
    pt_trans = img_trans = None

    if args.spherical:
        print("Sphere")
        #pt_trans = transforms.Compose([xyzToSpherical()])
        #pt_trans = transforms.Compose([pointToBins(args.min, args.max, 100)]) #xyzToSpherical()])

    if args.real:
        print("Real")
        from customRealDatasetsOrientation import OrangeSimDataSet, SubSet
    else:
        from customDatasetsOrientation import OrangeSimDataSet, SubSet

    dataclass = OrangeSimDataSet(args.data, args.num_images, args.num_pts, pt_trans, img_trans, custom_dataset=args.custom, input=args.input_size, depth=args.depth, seg=args.seg, temp_seg=args.temp_seg, seg_only=args.seg_only, rel_pose = False)
    dataloader = DataLoader(dataclass, batch_size=args.batch_size, num_workers=args.j)

    x_list = []
    y_list = []
    z_list = []
    yaw_list = []
    pitch_list = []
    roll_list = []

    for data in dataloader:
        points = data['points']
        #print(points[0][0])
        #points = torch.tensor(points)
        #print(len(points), len(points[0]), len(points[0][0]))
        #exit()
        for batch_ii in range(len(points)):
            for point_ii in range(args.num_pts):
                #print(batch_ii, point_ii)
                x = points[batch_ii][point_ii][0]
                y = points[batch_ii][point_ii][1]
                z = points[batch_ii][point_ii][2]
                yaw = points[batch_ii][point_ii][3]
                pitch = points[batch_ii][point_ii][4]
                roll = points[batch_ii][point_ii][5]
                if (len(x_list) < args.num_pts):
                    x_list.append([x,x])
                    y_list.append([y,y])
                    z_list.append([z,z])
                    yaw_list.append([yaw,yaw])
                    pitch_list.append([pitch,pitch])
                    roll_list.append([roll,roll])
                    continue
                if (x < x_list[point_ii][0]):
                    x_list[point_ii][0] = x
                if (x > x_list[point_ii][1]):
                    x_list[point_ii][1] = x
                if (y < y_list[point_ii][0]):
                    y_list[point_ii][0] = y
                if (y > y_list[point_ii][1]):
                    y_list[point_ii][1] = y
                if (z < z_list[point_ii][0]):
                    z_list[point_ii][0] = z
                if (z > z_list[point_ii][1]):
                    z_list[point_ii][1] = z
                if (yaw < yaw_list[point_ii][0]):
                    yaw_list[point_ii][0] = yaw
                if (yaw > yaw_list[point_ii][1]):
                    yaw_list[point_ii][1] = yaw
                if (pitch < pitch_list[point_ii][0]):
                    pitch_list[point_ii][0] = pitch
                if (pitch > pitch_list[point_ii][1]):
                    pitch_list[point_ii][1] = pitch
                if (roll < roll_list[point_ii][0]):
                    roll_list[point_ii][0] = roll
                if (roll > roll_list[point_ii][1]):
                    roll_list[point_ii][1] = roll
    
    print("x list: ", x_list)
    print("y list: ", y_list)
    print("z list: ", z_list)
    print("yaw list: ", yaw_list)
    print("pitch list: ", pitch_list)
    print("roll list: ", roll_list)

if __name__ == '__main__':
    main()
