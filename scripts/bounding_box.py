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
    parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
    parser.add_argument('--pred_dt', type=float, default=1.0, help='time between output waypoints')
    parser.add_argument('-j', type=int, default=4, help='number of loader workers')
    parser.add_argument('--depth',type=bool,default=False,help='use depth channel')
    parser.add_argument('--resets',type=bool,default=False,help='use reset data')
    parser.add_argument('--spherical',type=bool,default=False,help='use spherical coords')
    parser.add_argument('--relative',type=bool,default=False,help='use relative pose')
    parser.add_argument('--reduce_N',type=bool,default=False,help='remove final window')
    args = parser.parse_args()

    #args.min = [(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.75,-0.75,-0.5,-np.pi,-np.pi,-np.pi),(-1.0,-1.0,-0.75,-np.pi,-np.pi,-np.pi)]
    #args.max = [(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.5,1.0,0.5,np.pi,np.pi,np.pi),(2.0,1.0,0.75,np.pi,np.pi,np.pi)]
    pt_trans = img_trans = None

    if args.spherical:
        print("Sphere")
        pt_trans = transforms.Compose([xyzToSpherical()])

    #if args.real:
    #    print("Real")
    #    from customRealDatasetsOrientation import OrangeSimDataSet, SubSet
    #else:
    #    from customDatasetsOrientation import OrangeSimDataSet, SubSet
    from customDatasetv1 import OrangeSimDataSet

    dataclass = OrangeSimDataSet(args.data, 1, args.num_pts, pt_trans, img_trans, img_dt=1,pred_dt=args.pred_dt, reduce_N = args.reduce_N,depth=args.depth, rel_pose = args.relative, gaussian_pts=False,use_resets=args.resets)
    dataloader = DataLoader(dataclass, batch_size=args.batch_size, num_workers=args.j)

    x_list = []
    y_list = []
    z_list = []
    yaw_list = []
    pitch_list = []
    roll_list = []
    nan_count = [0 for i in range(6)]
    for ii in range(args.num_pts):
        x_list.append([])
        y_list.append([])
        z_list.append([])
        yaw_list.append([])
        pitch_list.append([])
        roll_list.append([])

    for data in dataloader:
        points = data['points']
        #print(points[0][0])
        #points = torch.tensor(points)
        #print(len(points), len(points[0]), len(points[0][0]))
        #exit()
        for batch_ii in range(len(points)):
            for point_ii in range(args.num_pts):
                #print(batch_ii, point_ii)
                x = np.array(float(points[batch_ii][point_ii][0]))
                y = np.array(float(points[batch_ii][point_ii][1]))
                z = np.array(float(points[batch_ii][point_ii][2]))
                yaw = np.array(float(points[batch_ii][point_ii][3]))
                pitch = np.array(float(points[batch_ii][point_ii][4]))
                roll = np.array(float(points[batch_ii][point_ii][5]))
                #if (len(x_list) < args.num_pts):
                #x_list.append([x,x])
                #y_list.append([y,y])
                #z_list.append([z,z])
                #yaw_list.append([yaw,yaw])
                #pitch_list.append([pitch,pitch])
                #roll_list.append([roll,roll]) 
                try:
                    if not np.any(np.isnan(x)):
                        x_list[point_ii].append(float(x))
                        nan_count[0] += 1

                    if not np.any(np.isnan(y)):
                        y_list[point_ii].append(float(y))
                        nan_count[1] += 1
                        
                    if not np.any(np.isnan(z)):
                        z_list[point_ii].append(float(z))
                        nan_count[2] += 1
                    if not np.any(np.isnan(yaw)):
                        yaw_list[point_ii].append(float(yaw))
                        nan_count[3] += 1
                    if not np.any(np.isnan(pitch)):
                        pitch_list[point_ii].append(float(pitch))
                        nan_count[4] += 1
                    if not np.any(np.isnan(roll)):
                        roll_list[point_ii].append(float(roll))
                        nan_count[5] += 1
                except:
                    print(x, y, z, yaw, pitch, roll)

    print("X min max:", np.min(x_list), np.max(x_list))
    print("X mean var:", np.mean(x_list), np.var(x_list), np.median(x_list))
    print("x 1: ", percentList(x_list,1))
    print("x 10: ", percentList(x_list,10))
    print("x 90: ", percentList(x_list,90))
    print("x 99: ", percentList(x_list,99))

    print("Y min max:", np.min(y_list), np.max(y_list))
    print("Y mean var:", np.mean(y_list), np.var(y_list), np.median(y_list))
    print("y 1: ", percentList(y_list,1))
    print("y 10: ", percentList(y_list,10))
    print("y 90: ", percentList(y_list,90))
    print("y 99: ", percentList(y_list,99))

    print("Z min max:", np.min(z_list), np.max(z_list))
    print("Z mean var:", np.mean(z_list), np.var(z_list), np.median(z_list))
    print("z 1: ", percentList(z_list,1))
    print("z 10: ", percentList(z_list,10))
    print("z 90: ", percentList(z_list,90))
    print("z 99: ", percentList(z_list,99))

    print("Yaw min max:", np.min(yaw_list), np.max(yaw_list))
    print("Yaw mean var:", np.mean(yaw_list), np.var(yaw_list), np.median(yaw_list))
    print("yaw 1: ", percentList(yaw_list,1))
    print("yaw 10: ", percentList(yaw_list,10))
    print("yaw 90: ", percentList(yaw_list,90))
    print("yaw 99: ", percentList(yaw_list,99))
    
    print("pitch min max:", np.min(pitch_list), np.max(pitch_list))
    print("pitch mean var:", np.mean(pitch_list), np.var(pitch_list), np.median(pitch_list))
    print("pitch 1: ", percentList(pitch_list,1))
    print("pitch 10: ", percentList(pitch_list,10))
    print("pitch 90: ", percentList(pitch_list,90))
    print("pitch 99: ", percentList(pitch_list,99))

    print("Roll min max:", np.min(roll_list), np.max(roll_list))
    print("roll mean var:", np.mean(roll_list), np.var(roll_list), np.median(roll_list))
    print("roll 1: ", percentList(roll_list,1))
    print("roll 10: ", percentList(roll_list,10))
    print("roll 90: ", percentList(roll_list,90))
    print("roll 99: ", percentList(roll_list,99))

    print(nan_count)
   
#    print("x list: ", x_list)
#    print("y list: ", y_list)
#    print("z list: ", z_list)
#    print("yaw list: ", yaw_list)
#    print("pitch list: ", pitch_list)
#    print("roll list: ", roll_list)
def percentList(coord_list,perc):
    # print(coord_list)
    ret_val = []
    for pt_list in coord_list:
        r = np.percentile(pt_list,perc)
        ret_val.append(r)
    return ret_val

if __name__ == '__main__':
    main()
