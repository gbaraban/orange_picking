import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
import PIL.Image as img
import sys
from torch.utils.data import Dataset, DataLoader
from customTransforms import *
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
from scipy.spatial.transform import Slerp

# import rosbag
# import rospy
from numpy import linalg as LA
import cv2
# from sensor_msgs.msg import Image, CompressedImage
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Point, Quaternion
# from std_msgs.msg import Bool, String
from scipy.linalg import logm
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pytransform3d.rotations import *
import shutil
# from visual_servoing_pose_estimate import BaselineOrangeFinder

#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#def make_step_plot(states,saveFolder=None,name=None):
#    #Plot Environment
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    alpha = 0.5
#    colors = ['red','blue','green']
#    ax.plot3D((0,alpha),(0,0),(0,0),colors[0])
#    ax.plot3D((0,0),(0,alpha),(0,0),colors[1])
#    ax.plot3D((0,0),(0,0),(0,alpha),colors[2])
#    #Plot states
#    for state in states:
#        if len(state) is 6:
#            p = state[0:3]
#            rot_mat = R.from_euler('ZYX',state[3:6]).as_matrix()
#        elif len(state) is 2:
#            p = state[0]
#            rot_mat = np.array(state[1])
#        for ii in [0,1,2]:
#            v = tuple(rot_mat[:,ii])
#            ax.plot3D((p[0],p[0] + alpha*v[0]),(p[1],p[1]+alpha*v[1]),(p[2],p[2]+alpha*v[2]),colors[ii])
#    if (saveFolder is None) or (name is None):
#        plt.show()
##    else:
#        fig.savefig(saveFolder + 'step_plot' + name + '.png')
#        fig.clf()
#        plt.close('all')
#



class SubSet(Dataset):
    def __init__(self,dataset,idx):
        self.ds = dataset
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self,i):
        return self.ds[self.idx[i]]

class OrangeSimDataSet(Dataset):
    def __init__(self, root_dir, num_images, num_pts, pt_trans, img_trans, img_dt=1.0, pred_dt=1.0, reduce_N = False, depth=False, rel_pose=False, gaussian_pts=False, use_resets = False):
        self.run_dir = root_dir
        self.num_images = num_images
        self.num_pts = num_pts
        self.point_transform = pt_trans
        self.image_transform = img_trans
        self.img_dt = img_dt
        self.pred_dt = pred_dt
        self.reduce_N = reduce_N
        self.depth = depth
        self.mean_seg = np.load("data/mean_imgv2_data_seg_real_world_traj_bag.npy")
        self.mean_color_image = np.load("data/orange_tracking_data/mean_imgv2_data_orange_tracking_data_real_world_traj_bag.npy")
        self.mean_depth_image = np.load("data/mean_depth_imgv2.npy")/10000.0 #10000 is the max of depth
        self.relative_pose = rel_pose
        self.gaussian_pts = gaussian_pts
        self.gaussian_var = [0.012, 0.012, 0.012, 0.04, 0.02, 0.02]
        self.gaussian_limit = [0.025, 0.025, 0.025, 0.08, 0.04, 0.04]
        self.resets = use_resets
        self.coeff_exists = False
        self.phase_dict = {"staging": 0, "final": 1, "reset": 2}

        self.np_dir = root_dir.rstrip("/") + "_np/"
        #if seg or seg_only: #temp_seg
        #    self.trial_list = os.listdir(root_dir + "../seg_mask_real_np/")
        if reduce_N:
            time_window = num_pts*img_dt
        else:
            time_window = 0

        bag_list = os.listdir(self.run_dir)
        trial_list = []
        for bag_folder in bag_list:
            if os.path.isdir(self.run_dir + bag_folder):
                trial_list = trial_list + [bag_folder+"/"+temp for temp in os.listdir(self.run_dir + bag_folder)]
        self.event_list = []
        self.traj_count = 0
        self.num_samples_dir = {}
        for trial in trial_list:
            trial_subfolders = os.listdir(self.run_dir + "/" + trial)
            if os.path.exists(self.run_dir + "/" + trial + "/data.pickle_no_parse"):
                ret_val = self.parseSubFolder(self.run_dir+"/"+trial+"/")
                if ret_val:
                    temp_dict = {}
                    temp_dict["start"] = len(self.event_list)
                    self.event_list = self.event_list+ret_val
                    temp_dict["end"] = len(self.event_list)
                    self.num_samples_dir[self.traj_count] = temp_dict
                    self.traj_count += 1
            else:
                for subfolder in trial_subfolders:
                    temp_dict = {}
                    temp_dict["start"] = len(self.event_list)
                    if ("staging" in subfolder) or ("final" in subfolder):
                        ret_val = self.parseSubFolder(self.run_dir+"/"+trial+"/"+subfolder+"/")
                        if ret_val:
                            self.event_list = self.event_list+ret_val
                            temp_dict["end"] = len(self.event_list)
                            self.num_samples_dir[self.traj_count] = temp_dict
                            self.traj_count += 1
                    if self.resets and ("reset" in subfolder):
                        ret_val = self.parseSubFolder(self.run_dir+"/"+trial+"/"+subfolder+"/")
                        if ret_val:
                            self.event_list = self.event_list+ret_val
                            temp_dict["end"] = len(self.event_list)
                            self.num_samples_dir[self.traj_count] = temp_dict
                            self.traj_count += 1


    def __len__(self):
        return len(self.event_list)

    def getTrajLen(self):
        pass

    def __getitem__(self,i):
        dict_i = self.event_list[i]
        points = np.array(dict_i["points"])

        if self.gaussian_pts:
            for pt_num in range(len(points)):
                for dof in range(len(points[pt_num])):
                    err = np.min((np.max((-np.random.normal(0.0, self.gaussian_var[dof]), -self.gaussian_limit[dof])), self.gaussian_limit[dof]))
                    points[pt_num][dof] += err
        image_loc_list = dict_i["image"]
        depth_loc_list = dict_i["depth"]
        orange_pose = None
        if "orange_pose" in dict_i:
            orange_pose = dict_i["orange_pose"].astype("float32")

        rp = dict_i["rp"]
        image = None
        for ii, image_loc in enumerate(image_loc_list):
            image_np_loc = image_loc.replace("real_world_traj_bag","real_world_traj_bag_np").replace("png","npy")
            if os.path.isfile(image_np_loc):
                temp_image = np.load(image_np_loc)
            else:
                temp_image_PIL = img.open(image_loc)
                temp_image = np.array(temp_image_PIL).astype(np.float32) #TODO: verify this works.  don't trust the internet
                temp_image /= 255.0
                temp_image -= self.mean_color_image

            temp_image = np.transpose(temp_image,[2,0,1])
            if self.depth:
                temp_depth = np.load(depth_loc_list[ii])/10000.0 # 10000 is max output of depth, found using data/find_mean_depth_img.py
                temp_depth -= self.mean_depth_image
                temp_depth = np.expand_dims(temp_depth,0)
                temp_image = np.concatenate((temp_image,temp_depth),axis=0)
            if image is None:
                image = temp_image
            else:
                image = np.concatenate((image,temp_image),axis=0)
        image = image.astype('float32')

        flipped = False
        # TODO Add augmentation
        point_list = []
        rot_list = []
        for i in range(self.num_pts):
            point_list.append(points[i, :3])
            rot_list.append(R.from_euler("ZYX", points[i, 3:6]).as_dcm())

        if self.image_transform:
             data = {}
             data["img"] = image
             data["pts"] = point_list
             data["rots"] = rot_list
             image, point_list, rot_list, flipped = self.image_transform(data)

        if flipped:
            temp_points = []
            for i in range(self.num_pts):
                temp = list(point_list[i])
                temp.extend(R.from_dcm(rot_list[i]).as_euler("ZYX"))
                temp_points.append(np.array(temp))
            if max([abs(pt[3]) for pt in temp_points]) > 1.3:
                #print("Input Points")
                #print(points)
                #print("New Points")
                #print(np.array(temp_points))
                pass
            points = np.array(temp_points)

        if self.point_transform:
            point_dict = {}
            point_dict['points'] = points
            point_dict['phase'] = dict_i["phase"]
            points = self.point_transform(point_dict)
        else:
            points = np.array(points).astype(np.float64)

        time_frac = dict_i["time_frac"]
        return_dict = {'image':image, 'points':points, "flipped":flipped, "time_frac": time_frac, "rp": rp.astype("float32")}

        if "phase" in dict_i:
            return_dict["phase"] = dict_i["phase"]

        if "bodyV" in dict_i and not self.coeff_exists:
            return_dict["body_v"] = dict_i["bodyV"].astype("float32")

        if orange_pose is not None and not self.coeff_exists:
            return_dict["orange_pose"] = orange_pose
        #print(return_dict)
        return return_dict

    def parseSubFolder(self,subfolder):
        if "final" in subfolder:
            phase = "final"
        elif "reset" in subfolder:
            phase = "reset"
        elif "staging" in subfolder:
            phase = "staging"
        else:
            phase = None
            print("phase not found in " + subfolder)
        dict_list = []
        with open(subfolder + "data.pickle_no_parse",'rb') as f:
            folder_data = pickle.load(f, encoding="latin1")
        if "coeff" in folder_data:
            self.coeff_exists = True
            #Hand carried data
            folder_odom = folder_data["data"]
            folder_coeff = folder_data["coeff"]
            folder_time = folder_data["time"]
            no_events = len(folder_coeff)
            image_hz = no_events/float(folder_time)
            image_offset = int(self.img_dt*image_hz)
            for ii, (odom, coeff) in enumerate(zip(folder_odom,folder_coeff)):
                dict_i = {}
                dict_i["image"] = []
                if self.depth:
                    dict_i["depth"] = []
                else:
                    dict_i["depth"] = None
                for image_ctr in range(self.num_images):
                    image_idx = max((ii - image_ctr*image_offset),0)
                    image_loc = subfolder + "image" + str(image_idx) + ".png"
                    dict_i["image"].append(image_loc)
                    if self.depth:
                        depth_loc = subfolder + "depth_image" + str(image_idx) + ".npy"
                        dict_i["depth"].append(depth_loc)
                dict_i["time_frac"] = float(ii)/no_events
                point_idx = ii
                temp_p0 = folder_odom[point_idx][0][0:3]
                temp_R0 = R.from_quat(folder_odom[point_idx][1][:4]).as_euler("ZYX")
                temp_pt = np.concatenate((temp_p0, temp_R0))
                folder_odom[point_idx] = temp_pt
                p0 = folder_odom[point_idx][0:3]
                #R0 = R.from_quat(folder_odom[point_idx][1][:4]).as_dcm()
                R0 = R.from_euler('ZYX', folder_odom[point_idx][3:6]).as_dcm()
                dict_i["rp"] = folder_odom[point_idx][4:6]
                time_list = [(temp+1)*self.pred_dt for temp in range(self.num_pts)]
                wp_list = self.coeffToWP(coeff,folder_odom[point_idx][0:4],time_list)
                point_list = []
                for pt_ctr in range(self.num_pts):
                    posyaw = wp_list[pt_ctr][0]
                    pi = posyaw[0:3]
                    Ri = R.from_euler('z', posyaw[3]).as_dcm()
                    pb = list(np.matmul(R0.T,pi-p0))
                    Rb = np.matmul(R0.T,Ri)
                    Rb_zyx = list(R.from_dcm(Rb).as_euler('ZYX'))
                    pb.extend(Rb_zyx)
                    point_list.append(pb)
                    if self.relative_pose:
                        p0 = pi
                        R0 = Ri
                dict_i["points"] = point_list
                if phase is not None:
                    dict_i["phase"] = self.phase_dict[phase]
#                forward_v = self.getBodyVelocity(folder_odom,point_idx,point_idx+1,point_hz)
#                backward_v = self.getBodyVelocity(folder_odom,point_idx-1,point_idx,point_hz)
#                if (forward_v is None) and (backward_v is None):
#                    print("Double None found")
#                    break
#                if (forward_v is None):
#                    dict_i["bodyV"] = backward_v
#                elif (backward_v is None):
#                    dict_i["bodyV"] = forward_v
#                else:
#                    dict_i["bodyV"] = 0.5*backward_v + 0.5*forward_v
                dict_list.append(dict_i)
        else:
            no_events = folder_data["nEvents"]
            no_points = folder_data["nOdom"]
            folder_time = folder_data["time_secs"] + (folder_data["time_nsecs"]/1e9)
            folder_odom = folder_data["data"]

            #print(no_events, no_points, folder_time, folder_data)
            folder_orange_pose = None
            if "orange_pose" in folder_data:
                folder_orange_pose = folder_data["orange_pose"] 

            if folder_time < 1.0:
                print(str(no_events) + " " + str(no_points) + " " + subfolder)        
                return None
            image_hz = no_events/float(folder_time)
            image_offset = int(self.img_dt*image_hz)
            point_hz = float(no_points)/folder_time
            point_offset = int(self.pred_dt*point_hz)
            for ii in range(no_events):
                dict_i = {}
                dict_i["image"] = []

                if folder_orange_pose is not None:
                    if folder_orange_pose[ii] is not None:
                        if np.any(np.isnan(folder_orange_pose[ii][0])) or np.any(np.isnan(folder_orange_pose[ii][1])):
                            dict_i["orange_pose"] = np.array([0., 0., 0., 0., 0., 0.])
                        else:
                            dict_i["orange_pose"] = np.concatenate((folder_orange_pose[ii][0], R.from_quat(folder_orange_pose[ii][1]).as_euler('ZYX'))) 
                    else:
                        dict_i["orange_pose"] = np.array([0., 0., 0., 0., 0., 0.])
                else:
                    dict_i["orange_pose"] = None

                if self.depth:
                    dict_i["depth"] = []
                else:
                    dict_i["depth"] = None
                for image_ctr in range(self.num_images):
                    image_idx = max((ii - image_ctr*image_offset),0)
                    image_loc = subfolder + "image" + str(image_idx) + ".png"
                    dict_i["image"].append(image_loc)
                    if self.depth:
                        depth_loc = subfolder + "depth_image" + str(image_idx) + ".npy"
                        dict_i["depth"].append(depth_loc)
                dict_i["time_frac"] = float(ii)/no_events
                point_idx = int(dict_i["time_frac"]*no_points)
                break_flag = False
                point_list = []
                if folder_odom[point_idx] is None: #TODO Ask gabe to check
                    continue
                p0 = folder_odom[point_idx][0:3]
                R0 = R.from_euler('ZYX', folder_odom[point_idx][3:6]).as_dcm()
                dict_i["rp"] = folder_odom[point_idx][4:6]
                #print("PIDX:", point_idx)
                for point_ctr in range(self.num_pts):
                    temp_idx = point_idx + (point_ctr+1)*point_offset
                    #print(temp_idx)
                    if self.reduce_N and temp_idx > no_points:
                        break_flag = True
                        break
                    temp_idx = min(temp_idx,len(folder_odom)-1)
                    if folder_odom[temp_idx] is None:
                        #print("interpolate")
                        odom = self.interpolateOdom(folder_odom,temp_idx)
                        if odom is None:
                            break_flag = True
                            break
                    else:
                        odom = folder_odom[temp_idx]
                    pi = odom[0:3]
                    Ri = R.from_euler('ZYX', odom[3:6]).as_dcm()
                    pb = list(np.matmul(R0.T,pi-p0))
                    Rb = np.matmul(R0.T,Ri)
                    Rb_zyx = list(R.from_dcm(Rb).as_euler('ZYX'))
                    pb.extend(Rb_zyx)
                    point_list.append(pb)
                    if self.relative_pose:
                        p0 = pi
                        R0 = Ri
                #print(point_idx, p0, R0, point_list)
                if break_flag:
                    continue
    #            if max(point_list[0]) > 1.5:
    #                print(subfolder)
    #                print(point_list)
    #                print("p0: ",p0)
    #                print("R0: ",R0)
    #                print("ii: ",ii)
    #                print("point_offset: ",point_offset)
    #                print("no_points: ",no_points)
    #                print("no_events: ",no_events)
    #                print("len(folder_odom) ",len(folder_odom))
    #                print("point_idx: ",point_idx)
    #                make_step_plot(folder_odom[point_idx:point_idx+point_offset])
    #                make_step_plot(point_list)
                dict_i["points"] = point_list
                dict_i["phase"] = self.phase_dict[phase]
                forward_v = self.getBodyVelocity(folder_odom,point_idx,point_idx+1,point_hz)
                backward_v = self.getBodyVelocity(folder_odom,point_idx-1,point_idx,point_hz)
                if (forward_v is None) and (backward_v is None):
                    print("Double None found")
                    continue
                if (forward_v is None):
                    dict_i["bodyV"] = backward_v
                elif (backward_v is None):
                    dict_i["bodyV"] = forward_v
                else:
                    dict_i["bodyV"] = 0.5*backward_v + 0.5*forward_v
                dict_list.append(dict_i)
        return dict_list
    
    def interpolateOdom(self,folder_odom,none_idx):
        back_idx = 0
        while (folder_odom[none_idx-back_idx] is None) and (none_idx-back_idx >= 0):
            back_idx += 1
        if none_idx-back_idx < 0:
            return None
        forward_idx = 0
        while (none_idx+forward_idx < len(folder_odom)) and (folder_odom[none_idx+forward_idx] is None):
            forward_idx += 1
        if none_idx+forward_idx >= len(folder_odom):
            return None
        odom_a = folder_odom[none_idx-back_idx]
        odom_b = folder_odom[none_idx+forward_idx]
        p_a = odom_a[0:3]
        p_b = odom_b[0:3]
        frac = float(back_idx)/(back_idx +forward_idx)
        p0 = p_a + frac*(p_b - p_a)
        Rs = R.from_euler('ZYX', (odom_a[3:6],odom_b[3:6]))
        sl = Slerp([none_idx-back_idx,none_idx+forward_idx],Rs)
        R_0 = sl([none_idx])
        return np.hstack((p0,R_0.as_euler('ZYX')[0]))

    def logR(self,R):
        arg = (np.trace(R)-1)/float(2)
        if arg >= 1:
            phi = 0.0
        elif arg <= -1:
            phi = np.pi
        else:
            phi = np.arccos(arg)
        sphi = np.sin(phi)
        temp = float(phi/(2.0*sphi))*(R-R.T)
        return np.array([temp[2,1],temp[0,2],temp[1,0]])

    def getBodyVelocity(self,odom_list,a,b,hz):
        if a < 0:
            a = 0
        if b >= len(odom_list):
            b = len(odom_list) - 1
        if (odom_list[a] is None) or (odom_list[b] is None):
            return None
        v = (odom_list[b][0:3] - odom_list[a][0:3])*hz
        R_b = R.from_euler('ZYX', odom_list[b][3:6]).as_dcm()
        R_a = R.from_euler('ZYX', odom_list[a][3:6]).as_dcm()
        w = (R_b.T).dot(self.logR(R_a.T.dot(R_b)))*hz
        return np.hstack((v,w))

    def basisMatrix(self,time):
        deg = 9
        dim = 4
        basis = np.zeros((dim+1,deg+1))
        coeff = np.ones(deg+1)
        t_exp = np.zeros(deg+1)
        t_exp[0] = 1
        for ii in range(1,deg):
            t_exp[ii] = t_exp[ii-1]*time
        for row in range(dim+1):
            for col in range(deg+1):
                col_row_diff = col - row
                if (row >= 1) and (col_row_diff >= 0):
                    coeff[col] = coeff[col]*(col_row_diff + 1)
                if (col_row_diff >= 0):
                    basis[row,col] = t_exp[col_row_diff]*coeff[col]
                else:
                    basis[row,col] = 0
        return basis

    def coeffToWP(self,coeff,x0,times):
        wp_list = []
        for time in times:
            basis = self.basisMatrix(time)
            out = np.matmul(basis,coeff)
            posyaw = x0 + out[0,:]
            velyawrate = out[1,:]
            wp_list.append((posyaw,velyawrate))
        return wp_list


