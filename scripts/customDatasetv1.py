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
    def __init__(self, root_dir, num_images, num_pts, pt_trans, img_trans, img_dt=1.0, pred_dt=1.0, reduce_N = False, depth=False, rel_pose=False, gaussian_pts=False, use_resets = False, extra_dt=None, relabel=False, mix_labels=True, body_v_dt = None, use_magnet=False, remove_hover=None, phase_end=False):
        self.run_dir = root_dir
        self.num_images = num_images
        self.num_pts = num_pts
        self.point_transform = pt_trans
        self.image_transform = img_trans
        self.img_dt = img_dt
        self.pred_dt = pred_dt
        self.extra_dt = extra_dt
        self.reduce_N = reduce_N
        self.depth = depth
        #self.mean_seg = np.load(root_dir + "/mean_seg.npy")
        self.mean_color_image = np.load(root_dir + "/mean_color_image.npy")
        self.mean_depth_image = np.load(root_dir + "/mean_depth_image.npy") #10000 is the max of depth
        self.relative_pose = rel_pose
        self.gaussian_pts = gaussian_pts
        self.gaussian_var = [0.012, 0.012, 0.012, 0.04, 0.02, 0.02]
        self.gaussian_limit = [0.025, 0.025, 0.025, 0.08, 0.04, 0.04]
        self.resets = use_resets
        self.coeff_exists = False
        self.phase_dict = {"staging": 0, "final": 1, "reset": 2, "grip": 3}
        self.final_thresh_x = float('-inf')
        self.final_thresh_r = float('-inf')
        self.relabel = relabel
        self.mix_labels = mix_labels
        self.final_reclassify_time = 2.5
        self.body_v_dt = body_v_dt
        self.use_magnet = use_magnet
        self.data_folder = "real_world_traj_bag"
        self.remove_hover = remove_hover
        if self.remove_hover and (type(self.remove_hover) is not type(tuple())):
            self.remove_hover = (self.remove_hover,self.remove_hover)
        if self.use_magnet:
            self.data_folder = "real_world_traj_mag_bag"
        if relabel or mix_labels:
            self.final_thresh_x = 1.2
            self.final_thresh_r = 0.6
        self.phase_end = phase_end
        
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
        phase_ctr = {}
        phase_time_list = {}
        phase_event_list = {}
        for trial in trial_list:
            if not os.path.isdir(self.run_dir + "/" + trial):
                continue
            trial_subfolders = os.listdir(self.run_dir + "/" + trial)
            if os.path.exists(self.run_dir + "/" + trial + "/data.pickle_no_parse"):
                ret_val, folder_info = self.parseSubFolder(self.run_dir+"/"+trial+"/")
                if ret_val:
                    temp_dict = {}
                    temp_dict["start"] = len(self.event_list)
                    self.event_list = self.event_list+ret_val
                    temp_dict["end"] = len(self.event_list)
                    self.num_samples_dir[self.traj_count] = temp_dict
                    self.traj_count += 1
                    if folder_info[0] not in phase_ctr.keys():
                        phase_ctr[folder_info[0]] = 0
                        phase_time_list[folder_info[0]] = []
                        phase_event_list[folder_info[0]] = []
                    phase_ctr[folder_info[0]] += 1
                    phase_time_list[folder_info[0]].append(folder_info[1])
                    phase_event_list[folder_info[0]].append(folder_info[2])
            else:
                for subfolder in trial_subfolders:
                    temp_dict = {}
                    temp_dict["start"] = len(self.event_list)
                    if ("staging" in subfolder) or ("final" in subfolder) or ("grip" in subfolder):
                        ret_val, folder_info = self.parseSubFolder(self.run_dir+"/"+trial+"/"+subfolder+"/")
                        if ret_val:
                            self.event_list = self.event_list+ret_val
                            temp_dict["end"] = len(self.event_list)
                            self.num_samples_dir[self.traj_count] = temp_dict
                            self.traj_count += 1
                            if folder_info[0] not in phase_ctr.keys():
                                phase_ctr[folder_info[0]] = 0
                                phase_time_list[folder_info[0]] = []
                                phase_event_list[folder_info[0]] = []
                            phase_ctr[folder_info[0]] += 1
                            phase_time_list[folder_info[0]].append(folder_info[1])
                            phase_event_list[folder_info[0]].append(folder_info[2])
                    if (self.resets > 0) and ("reset" in subfolder):
                        ret_val, folder_info = self.parseSubFolder(self.run_dir+"/"+trial+"/"+subfolder+"/")
                        if ret_val:
                            if (self.resets < 1):
                                #print("Sampling from Reset: ",len(ret_val)," ",folder_info[1]," ",folder_info[2])
                                temp_frac = min(self.resets,1)
                                rand_idx = np.random.permutation(len(ret_val))
                                final_idx = int(temp_frac*len(ret_val))
                                rand_idx = rand_idx[0:final_idx]
                                ret_val = [ret_val[_] for _ in rand_idx]
                                #folder_info[1] = folder_info[1]*temp_frac
                                #folder_info[2] = folder_info[2]*temp_frac
                                #print("After: ",len(ret_val)," ",folder_info[1]," ",folder_info[2])
                            self.event_list = self.event_list+ret_val
                            temp_dict["end"] = len(self.event_list)
                            self.num_samples_dir[self.traj_count] = temp_dict
                            self.traj_count += 1
                            if folder_info[0] not in phase_ctr.keys():
                                phase_ctr[folder_info[0]] = 0
                                phase_time_list[folder_info[0]] = []
                                phase_event_list[folder_info[0]] = []
                            phase_ctr[folder_info[0]] += 1
                            phase_time_list[folder_info[0]].append(folder_info[1])
                            phase_event_list[folder_info[0]].append(folder_info[2])
        for phase in phase_ctr.keys():
            print("Phase: " + str(phase) + " time data")
            print(str(phase_ctr[phase]) + " folders.  Average Length: " + str(np.mean(phase_time_list[phase])) + " Total Events: " + str(sum(phase_event_list[phase])))


    def __len__(self):
        return len(self.event_list)

    def getTrajLen(self):
        pass

    def __getitem__(self,arg_index):
        dict_i = self.event_list[arg_index]
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
        raw_image = None
        for ii, image_loc in enumerate(image_loc_list):
            image_np_loc = image_loc.replace(self.data_folder, self.data_folder + "_np").replace("png","npy")
            if os.path.isfile(image_np_loc):
                try:
                    temp_raw_image = np.load(image_np_loc)
                except:
                    print("Loading NP Image failed...")
                    print(image_np_loc)
                    exit(0)
            else:
                #print(image_loc)
                temp_raw_image_PIL = img.open(image_loc).resize((640,480))
                temp_raw_image = np.array(temp_raw_image_PIL.getdata())
                temp_raw_image = temp_raw_image.reshape(temp_raw_image_PIL.size[1],temp_raw_image_PIL.size[0],3)
                temp_raw_image = temp_raw_image[:,:,0:3]/255.0
                np.save(image_np_loc,temp_raw_image)

            temp_image = temp_raw_image - self.mean_color_image
            temp_raw_image = np.transpose(temp_raw_image,[2,0,1])
            temp_image = np.transpose(temp_image,[2,0,1])
            if self.depth:
                temp_depth = np.load(depth_loc_list[ii])/10000.0 # 10000 is max output of depth, found using data/find_mean_depth_img.py
                temp_depth -= self.mean_depth_image
                #temp_depth = np.expand_dims(temp_depth,0)
                temp_depth = np.transpose(temp_depth,[2,0,1])
                temp_image = np.concatenate((temp_image,temp_depth),axis=0)
            if image is None:
                image = temp_image
            else:
                image = np.concatenate((image,temp_image),axis=0)
            if raw_image is None:
                raw_image = temp_raw_image
            else:
                raw_image = np.concatenate((raw_image,temp_raw_image),axis=0)
        if image is not None:
            image = image.astype('float32')
            raw_image = raw_image.astype('float32')

        flipped = False
        rotated = False
        # TODO Add augmentation
        point_list = []
        rot_list = []
        for ii in range(self.num_pts):
            point_list.append(points[ii, :3])
            rot_list.append(R.from_euler("ZYX", points[ii, 3:6]).as_dcm())

        if self.image_transform:
            data = {}
            if image is not None:
                data["img"] = image
                data["raw_img"] = raw_image
            data["pts"] = point_list
            data["rots"] = rot_list
            data["orange_pose"] = orange_pose
            ret = self.image_transform(data)
            if image is not None:
                image = ret["img"]
                raw_image = ret["raw_img"]
            point_list = ret["pts"]
            rot_list = ret["rots"]
            if "flipped" in ret.keys():
                flipped = ret["flipped"]
            if "rotated" in ret.keys():
                rotated = ret["rotated"]
            orange_pose = ret["orange_pose"]
        if flipped or rotated:
            temp_points = []
            for i in range(self.num_pts):
                temp = list(point_list[i])
                temp.extend(R.from_dcm(rot_list[i]).as_euler("ZYX"))
                temp_points.append(np.array(temp))
            if max([abs(pt[3]) for pt in temp_points]) > 1.3:
                pass
            points = np.array(temp_points)
            rp = np.array((rp[0],-rp[1])) 
        phase = None
        if "phase" in dict_i:
            if type(dict_i["phase"]) is list and len(dict_i["phase"]) > 1:
                phase = np.random.choice(dict_i["phase"])
            else:
                phase = dict_i["phase"]

        if self.point_transform:
            point_dict = {}
            point_dict['points'] = points
            point_dict['phase'] = phase
            points = self.point_transform(point_dict)

        time_frac = dict_i["time_frac"]
        return_dict = {'points':points, "flipped":flipped, "time_frac": time_frac, "rp": rp.astype("float32")}
        if image is not None:
            return_dict['image'] = image
            return_dict['raw_image'] = raw_image

        if "phase" in dict_i:
            return_dict["phase"] = phase

        if "bodyV" in dict_i and not self.coeff_exists:
            bodyV = dict_i["bodyV"].astype("float32")
            if flipped:
                bodyV[1] = -bodyV[1]
                bodyV[3] = -bodyV[3]
                bodyV[5] = -bodyV[5]
            return_dict["body_v"] = dict_i["bodyV"].astype("float32")

        if "magnet" in dict_i:
            return_dict["magnet"] = (dict_i["magnet"]/4096.).astype(np.float32)

        if orange_pose is not None and not self.coeff_exists:
            return_dict["orange_pose"] = orange_pose.astype("float32")
        return return_dict

    def parseSubFolder(self,subfolder):
        if "final" in subfolder:
            phase = "final"
        elif "reset" in subfolder:
            phase = "reset"
        elif "staging" in subfolder:
            phase = "staging"
        elif "grip" in subfolder:
            phase = "grip"
        else:
            phase = None
            print("phase not found in " + subfolder)
        dict_list = []
        # print(subfolder)
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
            image_offset = round(self.img_dt*image_hz)
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
            no_devents = folder_data["nDEvents"]
            no_points = folder_data["nOdom"]
            folder_time = folder_data["time_secs"] + (folder_data["time_nsecs"]/1e9)
            folder_odom = folder_data["data"]

            folder_magnet = None
            no_magnets = None
            magnet_hz = None

            #print(no_events, no_points, folder_time, folder_data)
            folder_orange_pose = None
            if "orange_pose" in folder_data:
                folder_orange_pose = folder_data["orange_pose"] 

            if folder_time < 1.0:
                print("Skipping folder with: " + str(no_events) + " " + str(no_points) + " " + subfolder)        
                return None, None

            if ("mag" in folder_data) and (self.use_magnet):
                folder_magnet = folder_data["mag"]
                no_magnets = folder_data["nMag"]
                magnet_hz = float(no_magnets)/float(folder_time)

            image_hz = no_events/float(folder_time)
            dimage_hz = no_devents/float(folder_time)
            image_offset = round(self.img_dt*image_hz)
            point_hz = float(no_points)/folder_time
            if self.phase_dict[phase] == 0 or self.extra_dt is None: #TODO Fix time problem if mix_labels phase
                future_pred_dt = self.pred_dt
            else:
                future_pred_dt = self.extra_dt[self.phase_dict[phase]-1]
            point_offset = round(future_pred_dt*point_hz)
            for ii in range(no_events):
                dict_i = {}
                dict_i["image"] = []

                dict_i["phase"] = self.phase_dict[phase]
                
                if folder_orange_pose is not None:
                    orange_idx = int((float(ii)/no_events)*len(folder_orange_pose))
                    if folder_orange_pose[orange_idx] is not None:
                        if np.any(np.isnan(folder_orange_pose[orange_idx])):
                            dict_i["orange_pose"] = np.array([0., 0., 0., 0., 0., 0.])
                        else:
                            if len(folder_orange_pose[orange_idx]) is 2:
                                orange_p = folder_orange_pose[orange_idx][0]
                                orange_q = folder_orange_pose[orange_idx][1]
                                dict_i["orange_pose"] = np.concatenate((orange_p, R.from_quat(orange_q).as_euler('ZYX'))) 
                            if len(folder_orange_pose[orange_idx]) is 6:
                                orange_p = folder_orange_pose[orange_idx][0:3]
                                dict_i["orange_pose"] = np.array(folder_orange_pose[orange_idx])
                            if (phase is "staging") and (orange_p[0] < self.final_thresh_x) and (np.linalg.norm(orange_p[1:]) < self.final_thresh_r):
                                if self.relabel:
                                    dict_i["phase"] = self.phase_dict["final"]
                                elif self.mix_labels:
                                    dict_i["phase"] = [self.phase_dict["staging"], self.phase_dict["final"]]
                            if (phase is "final") and (ii <= image_hz*self.final_reclassify_time) and self.mix_labels:
                                dict_i["phase"] = [self.phase_dict["staging"], self.phase_dict["final"]]

                    else:
                        dict_i["orange_pose"] = np.array([0., 0., 0., 0., 0., 0.])
                else:
                    print("No orange_pose in:", subfolder)
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
                        dimage_idx = int((float(image_idx)/no_events)*no_devents)#TODO: try different strategies
                        depth_loc = subfolder + "depth_image" + str(dimage_idx) + ".npy"
                        dict_i["depth"].append(depth_loc)
                dict_i["time_frac"] = float(ii)/no_events
                point_idx = round(dict_i["time_frac"]*no_points)

                if folder_magnet is not None:
                    magnet_idx = round(dict_i["time_frac"]*no_magnets)
                    # print(len(folder_magnet), magnet_idx, dict_i["time_frac"])
                    if folder_magnet[magnet_idx] is not None:
                        dict_i["magnet"] = np.array(folder_magnet[magnet_idx])
                    else:
                        print("Issuee!! Magnet data not available")
                        dict_i["magnet"] = np.array([0., 0., 0.])

                break_flag = False
                point_list = []
                if folder_odom[point_idx] is None: #TODO Ask gabe to check
                    continue
                p0 = folder_odom[point_idx][0:3]
                R0 = R.from_euler('ZYX', folder_odom[point_idx][3:6]).as_dcm()
                dict_i["rp"] = folder_odom[point_idx][4:6]
                #print("PIDX:", point_idx)
                for point_ctr in range(self.num_pts):
                    if self.phase_end:
                        temp_idx = no_points - 1
                    else:
                        temp_idx = point_idx + (point_ctr+1)*point_offset
                        if self.reduce_N and temp_idx > no_points and (phase is not "grip"):
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
                    break#continue
                if (phase is not "grip") and self.remove_hover:
                    if np.all([(np.linalg.norm(point[0:3]) < self.remove_hover[0]) and (np.linalg.norm(point[3:]) < self.remove_hover[1]) for point in point_list]):
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
                if self.body_v_dt is None:
                    body_v_offset = 1
                else:
                    body_v_offset = round(self.body_v_dt*point_hz)
                forward_v = self.getBodyVelocity(folder_odom,point_idx,point_idx+body_v_offset,point_hz)
                backward_v = self.getBodyVelocity(folder_odom,point_idx-body_v_offset,point_idx,point_hz)
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
        return dict_list, (self.phase_dict[phase],folder_time, no_events)
    
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
        if abs(sphi) < 1e-30:
            return np.zeros(3)
        temp = float(phi/(2.0*sphi))*(R-R.T)
        if np.any(np.isnan(temp)):
            print("logR NaN")
            print(R)
            print(arg)
            print(phi)
            print(sphi)
            print(temp)
        return np.array([temp[2,1],temp[0,2],temp[1,0]])

    def getBodyVelocity(self,odom_list,a,b,hz):
        if a < 0:
            a = 0
        if b >= len(odom_list):
            b = len(odom_list) - 1
        if (odom_list[a] is None) or (odom_list[b] is None):
            return None
        dt = (a-b)*hz
        v = (odom_list[b][0:3] - odom_list[a][0:3])*dt
        R_b = R.from_euler('ZYX', odom_list[b][3:6]).as_dcm()
        R_a = R.from_euler('ZYX', odom_list[a][3:6]).as_dcm()
        v = (R_b.T).dot(v)
        w = (R_b.T).dot(self.logR(R_a.T.dot(R_b)))*dt
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


