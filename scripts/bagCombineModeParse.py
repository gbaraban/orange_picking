from __future__ import print_function
import rosbag
from numpy import linalg as LA
import PIL.Image as Img
import cv2
#from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Bool, String
import numpy as np
import sys
import os
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pytransform3d.rotations import *
import shutil
import subprocess
import time
from visual_servoing_pose_estimate import BaselineOrangeFinder
import glob
from computeMeanImage import create_mean_image
from tqdm import tqdm

bad_angles = []


status_table = [["Beginning"],["Manual","Hovering"],["ResetTrial"],["OrangeStaging"],["ResetOrangeTracking"],["OrangeTracking","FinalRise"],["GrippingState"],["Ending"]]

force_good_list = {"2022-07-25":{4:[1,0]},
        "2022-07-27":{1:[0,0],2:[-1],6:[0],16:[-1]},
        "2022-07-28":{6:[-1],7:[-1],18:[-1],19:[-1],20:[-1],21:[-1],18:[-1]},
        "2022-07-29":{7:[-1],8:[-1],11:[-1],23:[-1],29:[-1],31:[-1],33:[-1]},
        "2022-07-31":{13:[-1],23:[-1],24:[-1]},
        "2022-08-08":{30:[-1],37:[-1],38:[-1]},
        "2022-08-10":{9:[-1],10:[-1],13:[-1]},
        "2022-08-11":{2:[-1],3:[-1]},
        "2022-08-26":{5:[-1],8:[-1],9:[-1],14:[-1],16:[-1],18:[-1],23:[-1],25:[-1],30:[-1]},
        "2022-08-28":{13:[-1],14:[-1],16:[-1],20:[-1],22:[-1],30:[-1],31:[-1],32:[-1],33:[-1],34:[-1],37:[-1],38:[-1],39:[-1],40:[-1],43:[-1],45:[-1],46:[-1],47:[-1],48:[-1]},
        "2022-08-29":{10:[-1],11:[-1],12:[-1],13:[-1],14:[-1],21:[-1],26:[-1],27:[-1],28:[-1],29:[-1],30:[-1],34:[-1],35:[-1],36:[-1],37:[-1],43:[-1],52:[-1],58:[-1],60:[-1]},
        "2022-08-30":{1:[-1],6:[-1],7:[-1],8:[-1],10:[-1],12:[-1],13:[-1],14:[-1],15:[-1]}}

force_bad_list = ["_7_2022-07-27", 
        "_22_2022-07-28","_24_2022-07-28","_26_2022-07-28",
        "_5_2022-07-29","_27_2022-07-29",
        "_11_2022-07-31","_12_2022-07-31","_16_2022-07-31",
        "_2_2022-08-08","_5_2022-08-08","_6_2022-08-08","_9_2022-08-08","_35_2022-08-08","_44_2022-08-08",
        "_2_2022-08-10",
        "_27_2022-08-26","_29_2022-08-26",
        "_11_2022-08-28","_21_2022-08-28","_24_2022-08-28",
        "_19_2022-08-29","_24_2022-08-29"]

error_printout_file = ""
error_prefix = ""
summary_file = ""

overwrite = True


def np_from_compressed_image(comp_im):
    np_arr = np.fromstring(comp_im.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np

def depthnp_from_compressed_image(msg):
    # 'msg' as type CompressedImage
    depth_fmt, compr_type = msg.format.split(';')
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if compr_type != "compressedDepth":
        raise Exception("Compression type is not 'compressedDepth'."
                        "You probably subscribed to the wrong topic.")

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth_img_raw is None:
        # probably wrong header size
        raise Exception("Could not decode compressed depth image."
                        "You may need to change 'depth_header_size'!")

    if depth_fmt == "16UC1":
        # write raw image data
        # cv2.imwrite(os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"), depth_img_raw)
        return depth_img_raw

    elif depth_fmt == "32FC1":
        raw_header = msg.data[:depth_header_size]
        # header: int, float, float
        [compfmt, depthQuantA, depthQuantB] = struct.unpack('iff', raw_header)
        depth_img_scaled = depthQuantA / (depth_img_raw.astype(np.float32)-depthQuantB)
        # filter max values
        depth_img_scaled[depth_img_raw==0] = 0

        # depth_img_scaled provides distance in meters as f32
        # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
        depth_img_mm = (depth_img_scaled*1000).astype(np.uint16)
        # cv2.imwrite(os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"), depth_img_mm)
        return depth_img_mm

    else:
        raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")

def np_from_image(image):
  #Stolen from https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py
  dtype_class = np.uint8
  channels = 3
  dtype = np.dtype(dtype_class)
  dtype = dtype.newbyteorder('>' if image.is_bigendian else '<')
  shape = (image.height, image.width, channels)

  data = np.fromstring(image.data, dtype=dtype).reshape(shape)
  data.strides = (
    image.step,
    dtype.itemsize * channels,
    dtype.itemsize
  )
  return data

def depthnp_from_image(image):
  #Stolen from https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py
  dtype_class = np.uint16
  channels = 1
  dtype = np.dtype(dtype_class)
  dtype = dtype.newbyteorder('>' if image.is_bigendian else '<')
  shape = (image.height, image.width, channels)

  data = np.fromstring(image.data, dtype=dtype).reshape(shape)
  data.strides = (
    image.step,
    dtype.itemsize * channels,
    dtype.itemsize
  )
  #data = ((data[:,:,0].astype(np.float64)/data.max())*255).astype(np.uint8)
  return data

def logR(Rot):
    x = logm(Rot)
    return [x[2,1], x[0,2], x[1, 0]]

def compare_logR(R1, R2):
        R = np.matmul(R1.T, R2)
        logr = logm(R)
        z = np.array([logr[2,1], logr[0,2], logr[1,0]]);
        #print(z)
        z_ = np.array([logr[2,1], logr[0,2]])
        #print(z_)
        return LA.norm(z_), z_


def addDrop(R):
    R0 = np.eye(3)
    lr, z_ = compare_logR(R, R0)
    if lr < 1:
        return True

def saveData(save_loc, no_events, no_devents, odom_len, grip_len, start, end, time, odom, grip, bag_name, orange_pose=None):
    data_dict = {}
    data_dict["nEvents"] = no_events
    data_dict["nDEvents"] = no_devents
    data_dict["nOdom"] = odom_len
    if grip_len is not None:
        data_dict["nGrips"] = grip_len
        data_dict["grip"] = grip
    data_dict["start_secs"] = start.secs
    data_dict["start_nsecs"] = start.nsecs
    data_dict["end_secs"] = end.secs
    data_dict["end_nsecs"] = end.nsecs
    data_dict["time_secs"] = time.secs
    data_dict["time_nsecs"] = time.nsecs
    data_dict["data"] = odom
    data_dict["bag_name"] = bag_name
    if orange_pose is not None:
        data_dict["orange_pose"] = orange_pose
    with open(save_loc + '/data.pickle_no_parse','wb') as f:
        pickle.dump(data_dict,f,pickle.HIGHEST_PROTOCOL)

def parsePhase(info,trial_folder,name,extra_odom = [], stage_flag = False):
        folder = trial_folder + name
        global overwrite
        if overwrite:
            os.makedirs(folder)
        time = info[2]
        start_t = time[0]
        end_t = time[1]
        duration = time[1] - time[0]
        img_len = len(info[1])
        dimg_len = len(info[3])
        odom_len = len(info[0])
        odom = [np.zeros((6))]*odom_len

        no_events = img_len
        no_devents = dimg_len
        no_points = odom_len
        folder_time = duration.secs + (duration.nsecs/1e9)
        if folder_time < 1 and stage_flag:
            return None

        orange_pose = []
        
        for ii in range(odom_len):
            pt_odom = info[0][ii]
            if pt_odom is None:
                odom[ii] = None
            else:
                p = (pt_odom.transform.translation.x, pt_odom.transform.translation.y, pt_odom.transform.translation.z)
                RotR = R.from_quat([
                    pt_odom.transform.rotation.x,
                    pt_odom.transform.rotation.y,
                    pt_odom.transform.rotation.z,
                    pt_odom.transform.rotation.w
                    ])#possibly need to switch w to front
                Rot = RotR.as_dcm()
                p = np.array(p)
                pt = np.zeros((6))
                pt[0:3] = p
                pt[3:6] = RotR.as_euler('ZYX')
                odom[ii] = pt
        if overwrite:
            for ii in range(img_len):
                Img.fromarray(info[1][ii]).save(folder + "/image" + str(ii) + ".png")
            for ii in range(dimg_len):
                np.save(folder + "/depth_image" + str(ii) + ".npy", info[3][ii])

        op_len = len(info[5])
        for ii in range(op_len):
            msg = info[5][ii].poses[0]
            orange_loc = np.zeros((6))
            orange_loc[0] = msg.position.x
            orange_loc[1] = msg.position.x
            orange_loc[2] = msg.position.x
            temp = (msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w)
            orange_loc[3:6] = R.from_quat(temp).as_euler("ZYX")
            orange_pose.append(orange_loc)
#            orange_loc = None
#            time_frac = float(ii)/dimg_len
#            point_idx = int(time_frac*no_points)
#            img_idx = int(time_frac*img_len)
#            if odom[point_idx] is not None: #TODO Ask gabe to check
#                p0 = odom[point_idx][0:3]
#                R0 = R.from_euler('ZYX', odom[point_idx][3:6]).as_quat()
#                orange_loc = bof.process_loc(info[1][img_idx], info[3][ii], p0, R0)
#            orange_pose.append(orange_loc)

        load_len = len(info[4]) 
        load = [np.zeros((10))]*load_len
        for ii in range(load_len):
            load[ii] = np.array(info[4][ii])#CHANGE

        saveData(folder,img_len,dimg_len,odom_len,load_len,start_t, end_t, duration,odom + extra_odom,load,trial_folder, orange_pose)
        return duration,img_len,dimg_len,odom


def parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info,grip_info):
    total_time = None
    total_events = 0
    total_devents = 0
    trial_folder = bag_save_folder + "/trial" + str(trial_ctr)
    global overwrite
    if overwrite:
        os.makedirs(trial_folder)
    #Gripping
    if grip_info:
        duration, nevents, ndevents, grip_odom = parsePhase(grip_info,trial_folder,"/grip")
        if total_time is None:
            total_time = duration
        else:
            total_time += duration
        total_events += nevents
        total_devents += ndevents
    else:
       grip_odom = []
    #Final Rise
    if fr_info:
        duration, nevents, ndevents, fr_odom = parsePhase(fr_info,trial_folder,"/final",grip_odom)
        if total_time is None:
            total_time = duration
        else:
            total_time += duration
        total_events += nevents
        total_devents += ndevents
    else:
       fr_odom = []
    #Staging
    ret = parsePhase(stage_info,trial_folder,"/staging",fr_odom + grip_odom,stage_flag = True)
    if ret is None:
        return
    duration, nevents, ndevents, stage_odom = ret
    if total_time is None:
        total_time = duration
    else:
        total_time += duration
    total_events += nevents
    total_devents += ndevents
    #Resets
    num_resets = len(reset_info[0])
    for rr in range(num_resets):
        info = (reset_info[0][rr][0],reset_info[0][rr][1],reset_info[1][rr],reset_info[0][rr][2],reset_info[0][rr][3],reset_info[0][rr][4])
        duration, nevents, ndevents, reset_odom = parsePhase(info,trial_folder,"/reset" + str(rr),fr_odom + grip_odom)
        if total_time is None:
            total_time = duration
        else:
            total_time += duration
        total_events += nevents
        total_devents += ndevents
    print("")
    sum_str = trial_folder + ": " + str(total_events) + " events: " + str(total_devents) + " depth events: " + str(total_time.secs) + " seconds"
    print(sum_str)
    with open(summary_file,'a') as f:
        print(sum_str,file = f)

def chopBagName(fname):
    idx = 0
    for temp in range(3):
        idx = fname.find("_",idx+1)
    return fname[:(idx+11)]

def getStatus(data):
    for ctr,str_list in enumerate(status_table):
        for temp in str_list:
            if temp in data:
                return ctr
    print("Unknown State")
    with open(error_printout_file,'a') as f:
        print(error_prefix + "ERROR: " + "Unknown State " + data, file=f)
    return -1

BEGIN_STATUS = getStatus("Beginning")
MANUAL_STATUS = getStatus("Manual")
STAGE_STATUS = getStatus("OrangeStaging")
RESET_STATUS = getStatus("ResetOrangeTracking")
TRACK_STATUS = getStatus("FinalRise")
GRIP_STATUS = getStatus("GrippingState")
RESET_TRIAL_STATUS = getStatus("ResetTrial")
END_STATUS = getStatus("Ending")

def forceGood(name):
    for ss in force_good_list.keys():
        if ss in name:
            for trial_num in force_good_list[ss].keys():
                prefix = "_"+str(trial_num)+"_"
                if prefix in name:
                    return force_good_list[ss][trial_num]
    return []

def forceBad(name):
    for ss in force_bad_list:
        if ss in name:
            return True
    return False

def StateMachine(bag_name,old_state,new_state, good_flag = False, bad_flag = False):
    write_stage_resets = False
    write_final = False
    seal_stage = False
    seal_reset = False
    clear_flag = False
    clear_stage = False
    if new_state is MANUAL_STATUS:
        if (old_state is BEGIN_STATUS):
            return None
        if (old_state is TRACK_STATUS) or (old_state is RESET_STATUS):
            return None
            #Abort during track or final rise
            #Only save stage and resets
            #return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if (old_state is RESET_TRIAL_STATUS):
            return None
        if (old_state is STAGE_STATUS):
            return None
        if old_state is GRIP_STATUS:
            return None
        #    if good_flag:
        #        write_final = True
        #    write_stage_resets = True
        #    return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        print("Unknown Transition: " + str(old_state) + " > " + str(new_state))
        with open(error_printout_file,'a') as f:
            print(error_prefix + "ERROR: " + "Unknown Transition: " + str(old_state) + " > " + str(new_state),file=f)
        return -1
    if new_state is STAGE_STATUS:
        if (old_state is BEGIN_STATUS):
            return None
        if (old_state is MANUAL_STATUS):
            clear_stage = True
            clear_flag = True
            write_stage_resets = True
            if good_flag:
                write_final = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if (old_state is RESET_TRIAL_STATUS):
            clear_stage = True
            clear_flag = True
            write_stage_resets = True
            if not bad_flag:
                write_final = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if old_state is RESET_STATUS:
            seal_reset = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        print("Unknown Transition: " + str(old_state) + " > " + str(new_state))
        with open(error_printout_file,'a') as f:
            print(error_prefix + "ERROR: " + "Unknown Transition: " + str(old_state) + " > " + str(new_state),file=f)
        return -1
    if new_state is RESET_STATUS:
        if (old_state is STAGE_STATUS):
            clear_stage = True
            clear_flag = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if (old_state is TRACK_STATUS) or (old_state is GRIP_STATUS):
            return None
        print("Unknown Transition: " + str(old_state) + " > " + str(new_state))
        with open(error_printout_file,'a') as f:
            print(error_prefix + "ERROR: " + "Unknown Transition: " + str(old_state) + " > " + str(new_state),file=f)
        return -1
    if new_state is TRACK_STATUS:
        if (old_state is BEGIN_STATUS):
            return None
        if (old_state is MANUAL_STATUS):
            return None
        if (old_state is STAGE_STATUS):
            seal_stage = True
            clear_flag = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if (old_state is RESET_STATUS):
            clear_flag = True
            seal_reset = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        print("Unknown Transition: " + str(old_state) + " > " + str(new_state))
        with open(error_printout_file,'a') as f:
            print(error_prefix + "ERROR: " + "Unknown Transition: " + str(old_state) + " > " + str(new_state),file=f)
        return -1
    if new_state is GRIP_STATUS:
        if (old_state is TRACK_STATUS):
            return None
        print("Unknown Transition: " + str(old_state) + " > " + str(new_state))
        with open(error_printout_file,'a') as f:
            print(error_prefix + "ERROR: " + "Unknown Transition: " + str(old_state) + " > " + str(new_state),file=f)
        return -1
    if new_state is RESET_TRIAL_STATUS:
        if (old_state is MANUAL_STATUS) or (old_state is BEGIN_STATUS):
            return None
        if old_state is GRIP_STATUS:
            return None
            #if not bad_flag:
            #    write_final = True
            #write_stage_resets = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        print("Unknown Transition: " + str(old_state) + " > " + str(new_state))
        with open(error_printout_file,'a') as f:
            print(error_prefix + "ERROR: " + "Unknown Transition: " + str(old_state) + " > " + str(new_state),file=f)
        return -1
    if new_state is END_STATUS:
        if (old_state is MANUAL_STATUS):
            write_stage_resets = True
            if good_flag:
                write_final = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if (old_state is RESET_TRIAL_STATUS):
            write_stage_resets = True
            if not bad_flag:
                write_final = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if (old_state is STAGE_STATUS):
            return None
        if (old_state is RESET_STATUS):
            write_stage_resets = True
            if good_flag:
                write_final = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if (old_state is GRIP_STATUS):
            write_stage_resets = True
            if good_flag:
                write_final = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        if (old_state is TRACK_STATUS):
            write_stage_resets = True
            return (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage)
        print("Unknown Transition: " + str(old_state) + " > " + str(new_state))
        with open(error_printout_file,'a') as f:
            print(error_prefix + "ERROR: " + "Unknown Transition: " + str(old_state) + " > " + str(new_state),file=f)
        return -1
    print("Unknown New State: " + str(old_state) + " > " + str(new_state))
    with open(error_printout_file,'a') as f:
        print(error_prefix + "ERROR: " + "Unknown New State: " + str(old_state) + " > " + str(new_state),file=f)
    return -1

def parseBag(bag_dir,bag_name,bag_ctr):
    vrpn_topic = "/vrpn_client/matrice/pose"
    img_topic = "/camera/color/image_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
#    success_topic = "/magnet_values"
    success_topic = "/orange_tracking_node/arm/jaw_grip_data"
    status_topic = "/rqt_gui/system_status"
    tracker_topic = "/ros_tracker"
    bag_time = None
    bag_events = 0
    total_time = None
    total_events = 0
    print(str(bag_ctr) + ": " + bag_name),
    fname_list = []
    chopped_name = chopBagName(bag_name) 
    bag_save_folder = bag_dir + 'jaw_data/bag' + str(bag_ctr)
    global overwrite
    if overwrite:
        if os.path.isdir(bag_save_folder):
            shutil.rmtree(bag_save_folder)
        os.makedirs(bag_save_folder)
    global error_prefix
    error_prefix = bag_save_folder + ": "
    if bag_name.endswith("_0.bag"):
        fname_list = glob.glob(bag_dir + "/" + chopped_name + "*")
        fname_list.sort()
    else:
        fname_list.append(bag_dir + "/" + bag_name)
    bag_list = []
    for fname in fname_list:
        try:
            bag_list.append(rosbag.Bag(fname))
        except:
            print(fname + " is not loadable")
            #Add sudden death marker??
            break
    trial_ctr = 0
    stage_odom = []
    stage_img = []
    stage_dimg = []
    stage_load = []
    stage_op = []
    fr_odom = []
    fr_img = []
    fr_dimg = []
    fr_load = []
    fr_op = []
    resets = []
    reset_odom = []
    reset_img = []
    reset_dimg = []
    reset_load = []
    reset_op = []
    grip_odom = []
    grip_img = []
    grip_dimg = []
    grip_load = []
    grip_op = []
    img = None
    dimg = None
    odom = None
    grip = None
    op = None
    img_t = -1
    dimg_t = -1
    odom_t = -1
    grip_t = -1
    op_t = -1
    stage_time = [None,None]
    final_time = [None,None]
    reset_times = []
    reset_time = [None,None]
    grip_time = [None,None]
    status = BEGIN_STATUS
    stageFlag = True
    drop_ctr = 0
    drop_reset_thresh = 10
    success_ctr = 0
    #success_thresh = 550
    #success_thresh = 0.5
    #success_num_thresh = 4

    good_list = forceGood(chopped_name)
    bad_flag = forceBad(chopped_name)
    for bag in bag_list:
        for topic, msg, t in bag.read_messages(topics=[vrpn_topic, img_topic, status_topic, success_topic, depth_topic,tracker_topic]):
            if topic == success_topic:
                grip_t = t 
                grip = np.array(msg.values)
                if status is GRIP_STATUS:
                    grip_load.append(grip)
                if status is STAGE_STATUS:
                    stage_load.append(grip)
                if status is TRACK_STATUS:
                    fr_load.append(grip)
                if status is RESET_STATUS:
                    reset_load.append(grip)
            if topic == status_topic:
                new_status = getStatus(msg.data)
                if (new_status != status):
                    good_flag = False
                    if trial_ctr < len(good_list):
                        if (good_list[trial_ctr] == -1) or (good_list[trial_ctr] >= len(resets)):
                            good_flag = True
                    ret = StateMachine(chopped_name,status,new_status,good_flag,bad_flag)
                    if ret:
                        if ret == -1:
                            return
                        (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage) = ret
                        if seal_stage:
                            stageFlag = False
                        if seal_reset:
                            resets.append((reset_odom,reset_img,reset_dimg,reset_load,reset_op))
                            reset_odom = []
                            reset_img = []
                            reset_dimg = []
                            reset_load = []
                            reset_op = []
                            reset_times.append(reset_time)
                            if bag_time is None:
                                bag_time = reset_time[1] - reset_time[0]
                            else:
                                bag_time += (reset_time[1] - reset_time[0])
                            reset_time = [None,None]
                        if write_stage_resets:
                            if not stageFlag:
                                if (stage_time[0] is not None) and (stage_time[1] is not None):
                                    if (stage_time[1] - stage_time[0]).secs > 1:
                                        stage_info = (stage_odom,stage_img,stage_time,stage_dimg,stage_load,stage_op)
                                    else:
                                        stage_info = None
                                else:
                                    stage_info = None
                                if stage_info is not None:
                                    fr_info = None
                                    grip_info = None
                                    if write_final:
                                        if (final_time[0] is not None) and (final_time[1] is not None):
                                            fr_info = (fr_odom,fr_img,final_time,fr_dimg,fr_load,fr_op)
                                        if (grip_time[0] is not None) and (grip_time[1] is not None):
                                            grip_info = (grip_odom,grip_img,grip_time,grip_dimg,grip_load,grip_op)
                                    reset_info = (resets,reset_times)
                                    parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info,grip_info)
                                    if bag_time is None:
                                        bag_time = (stage_time[1] - stage_time[0])
                                    else:
                                        bag_time += (stage_time[1] - stage_time[0])
                                    if (final_time[1] is not None) and (final_time[0] is not None):
                                        bag_time += (final_time[1] - final_time[0])
                                    trial_ctr += 1
                        if clear_stage or write_stage_resets:
                            stageFlag = True
                            stage_odom = []
                            stage_img = []
                            stage_dimg = []
                            stage_load = []
                            stage_op = []
                            stage_time = [None,None]
                            reset_time = [None,None]
                            reset_times = []
                            resets = []
                        if clear_flag or write_final:
                            fr_odom = []
                            fr_img = []
                            fr_dimg = []
                            fr_load = []
                            fr_op = []
                            grip_odom = []
                            grip_img = []
                            grip_dimg = []
                            grip_load = []
                            grip_op = []
                            final_time = [None,None]
                            grip_time = [None,None]
                    status = new_status
            #Store Odometry
            if topic == vrpn_topic:
                Rot0 = R.from_quat([
                    msg.transform.rotation.x,
                    msg.transform.rotation.y,
                    msg.transform.rotation.z,
                    msg.transform.rotation.w
                    ])#possibly need to switch w to front
                R0 = Rot0.as_dcm()#.flatten()
                if not addDrop(R0):
                    drop_ctr += 1
                    #if drop_ctr > drop_reset_thresh:
                    #    odom = None
                    #continue
                    odom = None
                else:
                    if drop_ctr > drop_reset_thresh:
                        print("Dropped " + str(drop_ctr)),
                        with open(error_printout_file,'a') as f:
                            print(error_prefix + "ERROR: " + "Dropped " + str(drop_ctr), file=f)
                    drop_ctr = 0
                    odom = msg
                odom_t = t
                if status is STAGE_STATUS and stageFlag:
                    stage_odom.append(odom)
                if status is TRACK_STATUS:
                    fr_odom.append(odom)
                if status is RESET_STATUS:
                    reset_odom.append(odom)
                if status is GRIP_STATUS:
                    grip_odom.append(odom)
            #Store Images
            if topic == img_topic:
                img = msg
                img_t = t
                if status is STAGE_STATUS and stageFlag:
                    img_np = np_from_image(img)
                    stage_img.append(img_np)
                    bag_events += 1
                if status is TRACK_STATUS:
                    img_np = np_from_image(img)
                    fr_img.append(img_np)
                    bag_events += 1
                if status is GRIP_STATUS:
                    img_np = np_from_image(img)
                    grip_img.append(img_np)
                    bag_events += 1
                if status is RESET_STATUS:
                    img_np = np_from_image(img)
                    reset_img.append(img_np)
                    bag_events += 1
                img_np = None
            #Store Depth Images
            if topic == depth_topic:
                dimg = msg
                dimg_t = t
                if status is STAGE_STATUS and stageFlag:
                    dimg_np = depthnp_from_image(dimg)
                    stage_dimg.append(dimg_np)
                if status is TRACK_STATUS:
                    dimg_np = depthnp_from_image(dimg)
                    fr_dimg.append(dimg_np)
                if status is GRIP_STATUS:
                    dimg_np = depthnp_from_image(dimg)
                    grip_dimg.append(dimg_np)
                if status is RESET_STATUS:
                    dimg_np = depthnp_from_image(dimg)
                    reset_dimg.append(dimg_np)
                dimg_np = None
            if topic == tracker_topic:
                op = msg
                op_t = t
                if status is STAGE_STATUS and stageFlag:
                    stage_op.append(op)
                if status is TRACK_STATUS:
                    fr_op.append(op)
                if status is RESET_STATUS:
                    reset_op.append(op)
                if status is GRIP_STATUS:
                    grip_op.append(op)
            #Store Times
            if (img is not None) and (dimg is not None) and (odom is not None) and (grip is not None) and (op is not None):
                if status is STAGE_STATUS and stageFlag:
                    if stage_time[0] is None:
                        stage_time[0] = max((odom_t,img_t,dimg_t,grip_t,op_t))
                    stage_time[1] = max((odom_t,img_t,dimg_t,grip_t,op_t))
                if status is TRACK_STATUS:
                    if final_time[0] is None:
                        final_time[0] = max((odom_t,img_t,dimg_t,grip_t,op_t))
                    final_time[1] = max((odom_t,img_t,dimg_t,grip_t,op_t))
                if status is GRIP_STATUS:
                    if grip_time[0] is None:
                        grip_time[0] = max((odom_t,img_t,dimg_t,grip_t,op_t))
                    grip_time[1] = max((odom_t,img_t,dimg_t,grip_t,op_t))
                if status is RESET_STATUS:
                    if reset_time[0] is None:
                        reset_time[0] = max((odom_t,img_t,dimg_t,grip_t,op_t))
                    reset_time[1] = max((odom_t,img_t,dimg_t,grip_t,op_t))
                odom = None
                img = None
                dimg = None
                grip = None
                op = None
    #End of Bag
    new_status = getStatus("Ending")
    if (not stageFlag):
        good_flag = False
        if trial_ctr < len(good_list):
            if (good_list[trial_ctr] == -1) or (good_list[trial_ctr] >= len(resets)):
                good_flag = True
        ret = StateMachine(chopped_name,status,new_status,good_flag,bad_flag)
        if ret:
            if ret == -1:
                return
            (write_stage_resets,write_final,seal_stage,seal_reset,clear_flag,clear_stage) = ret
            if seal_stage:
                stageFlag = False
            if seal_reset:
                resets.append((reset_odom,reset_img,reset_dimg,reset_load,reset_op))
                reset_times.append(reset_time)
                if bag_time is None:
                    bag_time = reset_time[1] - reset_time[0]
                else:
                    bag_time += (reset_time[1] - reset_time[0])
            if write_stage_resets:
                if not stageFlag:
                    if (stage_time[0] is not None) and (stage_time[1] is not None):
                        if (stage_time[1] - stage_time[0]).secs > 1:
                            stage_info = (stage_odom,stage_img,stage_time,stage_dimg,stage_load,stage_op)
                        else:
                            stage_info = None
                    else:
                        stage_info = None
                    if stage_info is not None:
                        fr_info = None
                        grip_info = None
                        if write_final:
                            if (final_time[0] is not None) and (final_time[1] is not None):
                                fr_info = (fr_odom,fr_img,final_time,fr_dimg,fr_load,fr_op)
                            if (grip_time[0] is not None) and (grip_time[1] is not None):
                                grip_info = (grip_odom,grip_img,grip_time,grip_dimg,grip_load,grip_op)
                        reset_info = (resets,reset_times)
                        parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info,grip_info)
                        if bag_time is None:
                            bag_time = (stage_time[1] - stage_time[0])
                        else:
                            bag_time += (stage_time[1] - stage_time[0])
                        if (final_time[1] is not None) and (final_time[0] is not None):
                            bag_time += (final_time[1] - final_time[0])
                        trial_ctr += 1
    if bag_time:
        print("")
        print(bag_ctr, bag_time.secs, bag_time.nsecs, bag_events)
        #with open(summary_file,'a') as f:
        #    print(bag_ctr, bag_time.secs, bag_time.nsecs, bag_events,file=f)
    else:
        print(bag_ctr,"Bag Time None",bag_events)
        #with open(summary_file,'a') as f:
        #    print(bag_ctr,"Bag Time None",bag_events,file=f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_dir', help="bag dir")
    parser.add_argument('--bag_f', help="bag_file")
    parser.add_argument('--bag_ctr', help="bag ctr")
    parser.add_argument('-j', type=int,help="num_workers")
    parser.add_argument('--summary', help="do the summary only")
    args = parser.parse_args()
    global error_printout_file
    error_printout_file = args.bag_dir + "/error.txt"
    global summary_file
    summary_file = args.bag_dir + "/summary.txt"
    if args.summary is not None:
        print("Summary Only: ")
        read_summary(verbose = True)
        return
    if os.path.exists(summary_file):
        shutil.move(summary_file,summary_file+".old")
    if args.bag_f is not None:
        parseBag(args.bag_dir,args.bag_f,args.bag_ctr)
        return
    folder_list = [""] + os.listdir(args.bag_dir)
    for folder_name in folder_list:
        if not os.path.isdir(args.bag_dir + folder_name):
            continue
        folder_list = folder_list + [folder_name + "/" + temp for temp in os.listdir(args.bag_dir + folder_name) if os.path.isdir(args.bag_dir + folder_name + "/" + temp)]
    bag_list = []
    for folder_name in folder_list:
        if not os.path.isdir(args.bag_dir + folder_name):
            continue
        for temp in os.listdir(args.bag_dir + folder_name):
            if temp.endswith(".bag"):
                if temp.find("_",20):
                    if temp.endswith("_0.bag"):
                       bag_list.append(temp)
                else:
                    bag_list.append(temp)
    bag_ctr = 0
    bag_time = None
    bag_events = 0
    total_time = None
    total_events = 0
    p = ""
    if args.j:
        num_workers = args.j
    else:
        num_workers = 4
    print("Number of workers: " + str(num_workers))
    workers = []
    for bag_name in tqdm(bag_list):
        if not bag_name.endswith(".bag"):
            print("Trying to parse non-bag file: " + bag_name)
            continue
        if num_workers > 1:
            cmd = ["python","scripts/bagCombineModeParse.py",args.bag_dir,"--bag_f",bag_name,"--bag_ctr",str(bag_ctr)] 
            while (len(workers) >= num_workers):
                w = workers.pop(0)
                w.poll()
                if w.returncode is None:
                    workers.append(w)
                time.sleep(1.0)
            print("Starting Bag: " + bag_name + " " + str(bag_ctr))
            workers.append(subprocess.Popen(cmd))
            bag_ctr += 1
        else:
            parseBag(args.bag_dir,bag_name,bag_ctr)
    while (len(workers) > 0):
        w = workers.pop(0)
        w.poll()
        if w.returncode is None:
            workers.append(w)
        time.sleep(1.0)
    read_summary()
    global overwrite
    if overwrite:
        create_mean_image(args.bag_dir + '/jaw_data/')

def read_summary(verbose = False):
    total_events = 0
    total_devents = 0
    if verbose:
        print(summary_file)
    with open(summary_file,'r') as f:
        for l in f:
            if verbose:
                print(l)
            split_str = l.split(": ")
            total_events += int(split_str[1].split(" events")[0])
            total_devents += int(split_str[2].split(" depth events")[0])
    print("Total Events: " + str(total_events))
    print("Total Depth Events: " + str(total_devents))

if __name__ == "__main__":
    main()
