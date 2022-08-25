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
import matplotlib.pyplot as plt
from flatDataset import FlatDataSet
from customDatasetv1 import OrangeSimDataSet

phase_name_dict = {0: "staging", 1: "final", 2: "reset", 3: "grip"}

def make_phase_lists(phase_name,num_pts):
    ret_val = {}
    ret_val["name"] = phase_name_dict[int(phase_name)]
    ret_val["num_points"] = 0
    ret_val["x_list"] = []
    ret_val["y_list"] = []
    ret_val["z_list"] = []
    ret_val["yaw_list"] = []
    ret_val["pitch_list"] = []
    ret_val["roll_list"] = []
    ret_val["nan_count"] = [0 for i in range(6)]
    ret_val["body_v_x"] = []
    ret_val["body_v_y"] = []
    ret_val["body_v_z"] = []
    ret_val["body_w_1"] = []
    ret_val["body_w_2"] = []
    ret_val["body_w_3"] = []
    ret_val["body_mag_x"] = []
    ret_val["body_mag_y"] = []
    ret_val["body_mag_z"] = []
    for ii in range(num_pts):
        ret_val["x_list"].append([])
        ret_val["y_list"].append([])
        ret_val["z_list"].append([])
        ret_val["yaw_list"].append([])
        ret_val["pitch_list"].append([])
        ret_val["roll_list"].append([])
    return ret_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data folder')
    parser.add_argument('--batch_size', type=int, default=65, help='batch size')
    parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
    parser.add_argument('--pred_dt', type=float, default=1.0, help='time between output waypoints')
    parser.add_argument('-j', type=int, default=5, help='number of loader workers')
    parser.add_argument('--depth',type=bool,default=False,help='use depth channel')
    parser.add_argument('--resets',type=bool,default=True,help='use reset data')
    parser.add_argument('--spherical',type=bool,default=False,help='use spherical coords')
    parser.add_argument('--relative',type=bool,default=True,help='use relative pose')
    parser.add_argument('--reduce_N',type=bool,default=True,help='remove final window')
    parser.add_argument('--extra_dt', nargs="+", type=float, default=[0.25,0.25,0], help='pred_dt extra phases')
    parser.add_argument('--remove_hover', nargs="+", type=float, default=None,help='Threshold to remove equilibrium staging points')
    parser.add_argument('--magnet', type=bool, default=False, help='Use magnet data')
#    parser.add_argument('--flat', type=bool, default=False, help='Use flat dataset')
    parser.add_argument('--phase_end', type=bool, default=False, help='Use flat dataset')
    parser.add_argument('--plot', type=bool, default=False, help='Plot results')
    args = parser.parse_args()
    dict_name = "bounding_box_"
    ignore_attributes = ["data","batch_size","j","depth","remove_hover","magnet","plot"]
    for key in dir(args):
        if ((not key.startswith('_')) and (key not in ignore_attributes)):
            dict_name += "_" + key + "_" + str(getattr(args,key))
    dict_name = args.data + dict_name + ".pickle"
    print(dict_name)

    if args.remove_hover is not None:
        args.remove_hover = tuple(args.remove_hover)

    #args.min = [(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.75,-0.75,-0.5,-np.pi,-np.pi,-np.pi),(-1.0,-1.0,-0.75,-np.pi,-np.pi,-np.pi)]
    #args.max = [(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.5,1.0,0.5,np.pi,np.pi,np.pi),(2.0,1.0,0.75,np.pi,np.pi,np.pi)]
    #args.min = [(-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1)]
    #args.max = [(0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1), (0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1), (0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1)]

    if "pickle" in args.data:
        with open(args.data,'rb') as f:
            phase_dict = pickle.load(f, encoding="latin1")
    else:
        #args.bins = 100
        pt_trans = []
        if args.spherical:
            pt_trans.append(xyzToSpherical())
        #pt_trans.append(pointToBins(args.min,args.max,args.bins))
        if len(pt_trans) > 0:
            pt_trans = transforms.Compose(pt_trans)
        else:
            pt_trans = None

        img_trans = None
        img_trans = transforms.Compose([RandomHorizontalTrajFlip(p=0.5)])


        if args.phase_end:
            args.num_pts = 1
#            dataclass = FlatDataSet(args.data, 1, pt_trans, img_trans, depth=args.depth, pred_dt = args.pred_dt, img_dt=1, reduce_N=args.reduce_N, use_resets=args.resets, extra_dt=args.extra_dt, remove_hover = args.remove_hover, use_magnet=args.magnet)
        dataclass = OrangeSimDataSet(args.data, 1, args.num_pts, pt_trans, img_trans, img_dt=1,pred_dt=args.pred_dt, reduce_N = args.reduce_N,depth=args.depth, rel_pose = args.relative, gaussian_pts=False,use_resets=args.resets,extra_dt = args.extra_dt, relabel=False, mix_labels=False, remove_hover=args.remove_hover, use_magnet=args.magnet,phase_end=args.phase_end)
        dataloader = DataLoader(dataclass, batch_size=args.batch_size, num_workers=args.j)

        phase_dict = {}
        print(len(dataclass))

        for data in dataloader:
            phase = data["phase"]
            points = data['points']
            vels = data["body_v"]
            if args.magnet:
                mags = data["magnet"]
            #print(points[0][0])
            #points = torch.tensor(points)
            #print(len(points), len(points[0]), len(points[0][0]))
            #exit()
            for batch_ii in range(len(points)):
                phase_ii = int(phase[batch_ii])
                if phase_ii not in phase_dict.keys():
                    phase_dict[phase_ii] = make_phase_lists(phase_ii,args.num_pts)
                    print("Adding Phase: " + phase_dict[phase_ii]["name"])
                phase_dict[phase_ii]["num_points"] += 1
                vel = np.array(vels[batch_ii])
                try:
                    if not np.any(np.isnan(vel)):
                        phase_dict[phase_ii]["body_v_x"].append(float(vel[0]))
                        phase_dict[phase_ii]["body_v_y"].append(float(vel[1]))
                        phase_dict[phase_ii]["body_v_z"].append(float(vel[2]))
                        phase_dict[phase_ii]["body_w_1"].append(float(vel[3]))
                        phase_dict[phase_ii]["body_w_2"].append(float(vel[4]))
                        phase_dict[phase_ii]["body_w_3"].append(float(vel[5]))
                    else:
                        print("NaNs in bodyV")
                        print(vel)
                except Exception as e:
                    print("Vel Exception Found")
                    print(e)
                    print(vel)
                if args.magnet:
                    mag = np.array(mags[batch_ii])
                else:
                    mag = None
                try:
                    if mag is not None:
                        if not np.any(np.isnan(mag)):
                            phase_dict[phase_ii]["body_mag_x"].append(float(mag[0]))
                            phase_dict[phase_ii]["body_mag_y"].append(float(mag[1]))
                            phase_dict[phase_ii]["body_mag_z"].append(float(mag[2]))
                        else:
                            print("NaNs in mag")
                            print(mag)
                except Exception as e:
                    print("Mag Exception Found")
                    print(e)
                    print(mag)
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
                                phase_dict[phase_ii]["x_list"][point_ii].append(float(x))
                            else:
                                phase_dict[phase_ii]["nan_count"][0] += 1
                            if not np.any(np.isnan(y)):
                                phase_dict[phase_ii]["y_list"][point_ii].append(float(y))
                            else:
                                phase_dict[phase_ii]["nan_count"][1] += 1 
                            if not np.any(np.isnan(z)):
                                phase_dict[phase_ii]["z_list"][point_ii].append(float(z))
                            else:
                                phase_dict[phase_ii]["nan_count"][2] += 1
                            if not np.any(np.isnan(yaw)):
                                phase_dict[phase_ii]["yaw_list"][point_ii].append(float(yaw))
                            else:
                                phase_dict[phase_ii]["nan_count"][3] += 1
                            if not np.any(np.isnan(pitch)):
                                phase_dict[phase_ii]["pitch_list"][point_ii].append(float(pitch))
                            else:
                                phase_dict[phase_ii]["nan_count"][4] += 1
                            if not np.any(np.isnan(roll)):
                                phase_dict[phase_ii]["roll_list"][point_ii].append(float(roll))
                            else:
                                phase_dict[phase_ii]["nan_count"][5] += 1
                        except Exception as e:
                            print("Exception Found")
                            print(e)
                            print(x, y, z, yaw, pitch, roll)

        with open(dict_name,'wb') as f:
            pickle.dump(phase_dict,f,pickle.HIGHEST_PROTOCOL)

    for phase in phase_dict:
        print("Values for data labelled: " + phase_dict[phase]["name"])
        print("Num Points: " + str(phase_dict[phase]["num_points"]))
        print("X min max:", np.min(phase_dict[phase]["x_list"]), np.max(phase_dict[phase]["x_list"]))
        print("X mean var:", np.mean(phase_dict[phase]["x_list"]), np.var(phase_dict[phase]["x_list"]), np.median(phase_dict[phase]["x_list"]))
        print("x 1: ", percentList(phase_dict[phase]["x_list"],1))
        print("x 10: ", percentList(phase_dict[phase]["x_list"],10))
        print("x 90: ", percentList(phase_dict[phase]["x_list"],90))
        print("x 99: ", percentList(phase_dict[phase]["x_list"],99))

        print("Y min max:", np.min(phase_dict[phase]["y_list"]), np.max(phase_dict[phase]["y_list"]))
        print("Y mean var:", np.mean(phase_dict[phase]["y_list"]), np.var(phase_dict[phase]["y_list"]), np.median(phase_dict[phase]["y_list"]))
        print("y 1: ", percentList(phase_dict[phase]["y_list"],1))
        print("y 10: ", percentList(phase_dict[phase]["y_list"],10))
        print("y 90: ", percentList(phase_dict[phase]["y_list"],90))
        print("y 99: ", percentList(phase_dict[phase]["y_list"],99))

        print("Z min max:", np.min(phase_dict[phase]["z_list"]), np.max(phase_dict[phase]["z_list"]))
        print("Z mean var:", np.mean(phase_dict[phase]["z_list"]), np.var(phase_dict[phase]["z_list"]), np.median(phase_dict[phase]["z_list"]))
        print("z 1: ", percentList(phase_dict[phase]["z_list"],1))
        print("z 10: ", percentList(phase_dict[phase]["z_list"],10))
        print("z 90: ", percentList(phase_dict[phase]["z_list"],90))
        print("z 99: ", percentList(phase_dict[phase]["z_list"],99))

        print("Yaw min max:", np.min(phase_dict[phase]["yaw_list"]), np.max(phase_dict[phase]["yaw_list"]))
        print("Yaw mean var:", np.mean(phase_dict[phase]["yaw_list"]), np.var(phase_dict[phase]["yaw_list"]), np.median(phase_dict[phase]["yaw_list"]))
        print("yaw 1: ", percentList(phase_dict[phase]["yaw_list"],1))
        print("yaw 10: ", percentList(phase_dict[phase]["yaw_list"],10))
        print("yaw 90: ", percentList(phase_dict[phase]["yaw_list"],90))
        print("yaw 99: ", percentList(phase_dict[phase]["yaw_list"],99))
        
        print("pitch min max:", np.min(phase_dict[phase]["pitch_list"]), np.max(phase_dict[phase]["pitch_list"]))
        print("pitch mean var:", np.mean(phase_dict[phase]["pitch_list"]), np.var(phase_dict[phase]["pitch_list"]), np.median(phase_dict[phase]["pitch_list"]))
        print("pitch 1: ", percentList(phase_dict[phase]["pitch_list"],1))
        print("pitch 10: ", percentList(phase_dict[phase]["pitch_list"],10))
        print("pitch 90: ", percentList(phase_dict[phase]["pitch_list"],90))
        print("pitch 99: ", percentList(phase_dict[phase]["pitch_list"],99))

        print("Roll min max:", np.min(phase_dict[phase]["roll_list"]), np.max(phase_dict[phase]["roll_list"]))
        print("roll mean var:", np.mean(phase_dict[phase]["roll_list"]), np.var(phase_dict[phase]["roll_list"]), np.median(phase_dict[phase]["roll_list"]))
        print("roll 1: ", percentList(phase_dict[phase]["roll_list"],1))
        print("roll 10: ", percentList(phase_dict[phase]["roll_list"],10))
        print("roll 90: ", percentList(phase_dict[phase]["roll_list"],90))
        print("roll 99: ", percentList(phase_dict[phase]["roll_list"],99))
        #axs[0][0].hist(phase_dict[phase]["body_v_x"],bins=None)
        #axs[0][1].hist(phase_dict[phase]["body_v_y"],bins=None)
        #axs[0][2].hist(phase_dict[phase]["body_v_z"],bins=None)
        #axs[1][0].hist(phase_dict[phase]["body_w_1"],bins=None)
        #axs[1][1].hist(phase_dict[phase]["body_w_2"],bins=None)
        #axs[1][2].hist(phase_dict[phase]["body_w_3"],bins=None)
        print("Min/max vx: " + str(min(phase_dict[phase]["body_v_x"])) + " " + str(max(phase_dict[phase]["body_v_x"])))
        print("Min/max vy: " + str(min(phase_dict[phase]["body_v_y"])) + " " + str(max(phase_dict[phase]["body_v_y"])))
        print("Min/max vz: " + str(min(phase_dict[phase]["body_v_z"])) + " " + str(max(phase_dict[phase]["body_v_z"])))
        print("Min/max w1: " + str(min(phase_dict[phase]["body_w_1"])) + " " + str(max(phase_dict[phase]["body_w_1"])))
        print("Min/max w2: " + str(min(phase_dict[phase]["body_w_2"])) + " " + str(max(phase_dict[phase]["body_w_2"])))
        print("Min/max w3: " + str(min(phase_dict[phase]["body_w_3"])) + " " + str(max(phase_dict[phase]["body_w_3"])))
        if args.magnet:
            print("Min/max/avg magx: " + str(min(phase_dict[phase]["body_mag_x"])) + " " + str(max(phase_dict[phase]["body_mag_x"])) + " " + str(np.mean(phase_dict[phase]["body_mag_x"])))
            print("Min/max/avg magy: " + str(min(phase_dict[phase]["body_mag_y"])) + " " + str(max(phase_dict[phase]["body_mag_y"])) + " " + str(np.mean(phase_dict[phase]["body_mag_y"])))
            print("Min/max/avg magz: " + str(min(phase_dict[phase]["body_mag_z"])) + " " + str(max(phase_dict[phase]["body_mag_z"])) + " " + str(np.mean(phase_dict[phase]["body_mag_z"])))
        print(phase_dict[phase]["nan_count"])
        if args.plot:
            fig, axs = plt.subplots(2,3)
            strings = ["body_v_x","body_v_y","body_v_z","body_w_1","body_w_2","body_w_3"]
            for ii in range(2):
                for jj in range(3):
                    temp = phase_dict[phase][strings[ii*3+jj]]
                    lb,ub= np.percentile(temp,(1,99))
                    axs[ii][jj].hist(temp,range=(lb,ub))
    if args.plot:
        plt.show()
       
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
