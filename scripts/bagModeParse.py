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
from visual_servoing_pose_estimate import BaselineOrangeFinder

bad_angles = []

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
        #if z_[0] > 0.0000001 or z_[1] > 0.0000001:
        return True

#def getNextPts(pts_odom, pt_num, events_per_sec, ax):
#    pt_odom = pts_odom[pt_num]
#    p0 = (pt_odom.transform.translation.x, pt_odom.transform.translation.y, pt_odom.transform.translation.z)
#    Rot0 = R.from_quat([
#                      pt_odom.transform.rotation.x,
#                      pt_odom.transform.rotation.y,
#                      pt_odom.transform.rotation.z,
#                      pt_odom.transform.rotation.w
#                     ])#possibly need to switch w to front
#    R0 = Rot0.as_dcm()#.flatten()
#    if not addDrop(R0):
#        bad_angles.append(R0)
#        return None, None, None, None, None
#    p = np.array(p0)
#
#    indices = np.floor(np.add(np.array([1, 2, 3]) * events_per_sec, pt_num)).astype(int)
#
#    for x, i in enumerate(indices):
#        if i >= len(pts_odom):
#            delta = pt_num if (x == 0) else indices[x-1]
#            dt = np.floor(((len(pts_odom) -1) - delta)/(3-x)).astype(int)
#            z = 3 - x
 #           while (dt < 1) and z != 0:
 #               dt = np.floor(((len(pts_odom) - 1) - delta)/z).astype(int)
 #               z -= 1
 #           for j in range(0,z):
 #               indices[j+x] = delta + ((j+1)*dt)
#
#            delta = 3 - (z+x)
#            for j in range(0, delta):
#                indices[x+z+j] = len(pts_odom) - 1
#            break
#
#    point_list = []
#    rot_list = []
#    rot_list_po = []
#    for x, i in enumerate(indices):
#        #print(i)
#        if (i < pt_num):
#            print(pt_num, i)
#        p =  (pts_odom[i].transform.translation.x, pts_odom[i].transform.translation.y, pts_odom[i].transform.translation.z)
#        #print(p)
#        point_list.append(p)
#        Roti = R.from_quat([
#            pts_odom[i].transform.rotation.x,
#            pts_odom[i].transform.rotation.y,
#            pts_odom[i].transform.rotation.z,
#            pts_odom[i].transform.rotation.w
#            ])#possibly need to switch w to front
#        Ri = Roti.as_dcm()
#        R_relative = np.matmul(np.array(R0).T, np.array(Ri))
#        ri = R.from_dcm(R_relative)
#        r_zyx = ri.as_euler('zyx')
#        r_po_zyx = Roti.as_euler('zyx')
#        rot_list.append([r_zyx[0], r_zyx[1], r_zyx[2]])
#        rot_list_po.append([r_po_zyx[0], r_po_zyx[1], r_po_zyx[2]])
#        #print(rot_list[x])
#
#    point_list = np.array(point_list)
#    #print(point_list)
#    rot_list = np.array(rot_list)
#    #print(rot_list)
#
#    pos_only_point_list = np.matmul(np.array(R0).T, (point_list - np.array(p0)).T).T
#
#    point_list = np.matmul(np.array(R0).T, (point_list - np.array(p0)).T).T
#
#    #print(point_list)
#    point_list = np.concatenate((point_list, rot_list), axis=1)
#    #print(point_list)
#    rot0 = Rot0.as_euler('zyx')
#    pos_only_p0 = list(p0)
#    p0 = list(p0)
#    p0.extend([rot0[0], rot0[1], rot0[2]])
#
#    return point_list, pos_only_point_list, indices, p0, pos_only_p0
#

def saveData(save_loc, no_events, odom_len, time, odom, bag_name, orange_pose=None):
    data_dict = {}
    data_dict["nEvents"] = no_events
    data_dict["nOdom"] = odom_len
    data_dict["time_secs"] = time.secs
    data_dict["time_nsecs"] = time.nsecs
    data_dict["data"] = odom
    data_dict["bag_name"] = bag_name
    if orange_pose is not None:
        data_dict["orange_pose"] = orange_pose
    with open(save_loc + '/data.pickle_no_parse','wb') as f:
        pickle.dump(data_dict,f,pickle.HIGHEST_PROTOCOL)

def parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info):
    total_time = None
    total_events = 0
    trial_folder = bag_save_folder + "/trial" + str(trial_ctr)
    os.makedirs(trial_folder)
    bof = BaselineOrangeFinder()
    #Final Rise
    if fr_info:
        fr_folder = trial_folder + "/final"
        os.makedirs(fr_folder)
        # print("Line 187:", fr_info)
        # print("Line 188:", fr_info[2])
        fr_time = fr_info[2]
        # print("Line 190:", fr_info[0], fr_info[1])
        fr_duration = fr_time[1] - fr_time[0]
        fr_len = min((len(fr_info[1]), len(fr_info[3])))
        odom_len = len(fr_info[0])
        fr_odom = [np.zeros((6))]*odom_len

        no_events = fr_len
        no_points = odom_len
        folder_time = fr_duration.secs + (fr_duration.nsecs/1e9)

        orange_pose = []
        
        for ii in range(odom_len):
            pt_odom = fr_info[0][ii]
            if pt_odom is None:
                fr_odom[ii] = None
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
                pt[3:6] = RotR.as_euler('zyx')
                fr_odom[ii] = pt
        for ii in range(fr_len):
            orange_loc = None
            time_frac = float(ii)/no_events
            point_idx = int(time_frac*no_points)

            Img.fromarray(fr_info[1][ii]).save(fr_folder + "/image" + str(ii) + ".png")
            np.save(fr_folder + "/depth_image" + str(ii) + ".npy", fr_info[3][ii])
            
            if fr_odom[point_idx] is not None: #TODO Ask gabe to check
                p0 = fr_odom[point_idx][0:3]
                R0 = R.from_euler('zyx', fr_odom[point_idx][3:6]).as_quat()
                orange_loc = bof.process_loc(fr_info[1][ii], fr_info[3][ii], p0, R0)

            orange_pose.append(orange_loc)

        saveData(fr_folder,fr_len,odom_len,fr_duration,fr_odom,trial_folder, orange_pose)
        if total_time is None:
            total_time = fr_duration
        else:
            total_time += fr_duration
        total_events += fr_len
    else:
        fr_odom = []
    #Staging
    stage_folder = trial_folder + "/staging"
    os.makedirs(stage_folder)
    stage_time = stage_info[2]
    stage_duration = stage_time[1] - stage_time[0]
    stage_len = min((len(stage_info[1]), len(stage_info[3])))
    odom_len = len(stage_info[0])
    stage_odom = [np.zeros((6))]*odom_len

    no_events = stage_len
    no_points = odom_len
    folder_time = stage_duration.secs + (stage_duration.nsecs/1e9)

    orange_pose = []
    
    for ii in range(odom_len):
        pt_odom = stage_info[0][ii]
        if pt_odom is None:
            stage_odom[ii] = None
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
            pt[3:6] = RotR.as_euler('zyx')
            stage_odom[ii] = pt
    for ii in range(stage_len):
        orange_loc = None
        time_frac = float(ii)/no_events
        point_idx = int(time_frac*no_points)

        Img.fromarray(stage_info[1][ii]).save(stage_folder + "/image" + str(ii) + ".png")
        np.save(stage_folder + "/depth_image" + str(ii) + ".npy", stage_info[3][ii])

                    
        if stage_odom[point_idx] is not None: #TODO Ask gabe to check
            p0 = stage_odom[point_idx][0:3]
            R0 = R.from_euler('zyx', stage_odom[point_idx][3:6]).as_quat()
            orange_loc = bof.process_loc(stage_info[1][ii], stage_info[3][ii], p0, R0)

        orange_pose.append(orange_loc)


    saveData(stage_folder,stage_len,odom_len,stage_duration,stage_odom + fr_odom,trial_folder, orange_pose)
    if total_time is None:
        total_time = stage_duration
    else:
        total_time += stage_duration
    total_events += stage_len
    #Resets
    num_resets = len(reset_info[0])
    for rr in range(num_resets):
        reset_folder = trial_folder + "/reset" + str(rr) + "/"
        os.makedirs(reset_folder)
        reset_time = reset_info[1][rr]
        reset_duration = reset_time[1] - reset_time[0]
        reset_len = min((len(reset_info[0][rr][1]), len(reset_info[0][rr][2])))
        odom_len = len(reset_info[0][rr][0])
        reset_odom = [np.zeros((6))]*odom_len
        for ii in range(odom_len):
            pt_odom = reset_info[0][rr][0][ii]
            if pt_odom is None:
                reset_odom[ii] = None
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
                pt[3:6] = RotR.as_euler('zyx')
                reset_odom[ii] = pt
        for ii in range(reset_len):
            Img.fromarray(reset_info[0][rr][1][ii]).save(reset_folder + "/image" + str(ii) + ".png")
            np.save(reset_folder + "/depth_image" + str(ii) + ".npy", reset_info[0][rr][2][ii])

        saveData(reset_folder,reset_len,odom_len,reset_duration,reset_odom + fr_odom,trial_folder)
        if total_time is None:
            total_time = reset_duration
        else:
            total_time += reset_duration
        total_events += reset_len
    print("")
    print(trial_folder + ": " + str(total_events) + " events: " + str(total_time.secs) + " seconds")

def main():
    vrpn_topic = "/vrpn_client/matrice/pose"
    img_topic = "/camera/color/image_raw/compressed"
    depth_topic = "/camera/aligned_depth_to_color/image_raw/compressedDepth"
    success_topic = "/magnet_values"
    status_topic = "/rqt_gui/system_status"

    parser = argparse.ArgumentParser()
    parser.add_argument('bag_dir', help="bag dir")
    args = parser.parse_args()

    folder_list = os.listdir(args.bag_dir)
    bag_list = []
    for folder_name in folder_list:
        if not os.path.isdir(args.bag_dir + folder_name):
            continue
        bag_list = bag_list + [folder_name + "/" + temp for temp in os.listdir(args.bag_dir + folder_name)]
    bag_ctr = 0
    bag_time = None
    bag_events = 0
    total_time = None
    total_events = 0
    for bag_name in bag_list:
        if not bag_name.endswith(".bag"):
            continue
        print(str(bag_ctr) + ": " + bag_name),
        fname = args.bag_dir + "/" + bag_name
        bag = rosbag.Bag(fname)

        bag_save_folder = args.bag_dir + 'real_world_traj_bag/bag' + str(bag_ctr)
        if os.path.isdir(bag_save_folder):
            shutil.rmtree(bag_save_folder)
        os.makedirs(bag_save_folder)

        trial_ctr = 0
        stage_odom = []
        stage_img = []
        stage_dimg = []
        fr_odom = []
        fr_img = []
        fr_dimg = []
        resets = []
        reset_odom = []
        reset_img = []
        reset_dimg = []
        img = None
        dimg = None
        odom = None
        img_t = -1
        dimg_t = -1
        depth_t = -1
        odom_t = -1
        stage_time = [None,None]
        final_time = [None,None]
        reset_times = []
        reset_time = [None,None]
        status = "Manual"
        stageFlag = True
        drop_ctr = 0
        drop_reset_thresh = 10
        success_ctr = 0
        success_thresh = 550
        success_num_thresh = 4

        for topic, msg, t in bag.read_messages(topics=[vrpn_topic, img_topic, status_topic, success_topic, depth_topic]):
            # print(status)
            if topic == success_topic:
                if np.linalg.norm(msg.data) > success_thresh:
                    success_ctr += 1
                else:
                    success_ctr = 0
                if (success_ctr >= success_num_thresh) and (status is not "ResetTrial") and (status is not "Manual"):
                    print("Magnet Success"),
                    status = "ResetTrial"
                    stage_info = (stage_odom,stage_img,stage_time,stage_dimg)
                    fr_info = (fr_odom,fr_img,final_time,fr_dimg)
                    reset_info = (resets,reset_times)
                    parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info)
                    if bag_time is None:
                        bag_time = (stage_time[1] - stage_time[0]) + (final_time[1] - final_time[0])
                    else:
                        bag_time += (stage_time[1] - stage_time[0]) + (final_time[1] - final_time[0])
                    trial_ctr += 1 
                    stage_odom = []
                    stage_img = []
                    stage_dimg = []
                    fr_odom = []
                    fr_img = []
                    fr_dimg = []
                    resets = []
                    reset_odom = []
                    reset_img = []
                    reset_dimg = []
                    img = None
                    dimg = None
                    odom = None
                    img_t = -1
                    dimg_t = -1 
                    depth_t = -1
                    odom_t = -1
                    stage_time = [None,None]
                    stageFlag = True
                    final_time = [None,None]
                    reset_times = []
                    reset_time = [None,None]

            if topic == status_topic:
                if ("Manual" in msg.data) or ("Hover" in msg.data):
                    #if status is "Stage":
                    if status is "Final":
                        if not stageFlag:
                            stage_info = (stage_odom,stage_img,stage_time,stage_dimg)
                            fr_info = None
                            reset_info = (resets,reset_times)
                            parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info)
                            trial_ctr += 1
                            if bag_time is None:
                                bag_time = stage_time[1] - stage_time[0]
                            else:
                                bag_time += (stage_time[1] - stage_time[0])
                    if status is "Reset":
                        if not stageFlag:
                            stage_info = (stage_odom,stage_img,stage_time,stage_dimg)
                            fr_info = None
                            reset_info = (resets,reset_times)
                            parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info)
                            trial_ctr += 1
                            if bag_time is None:
                                bag_time = stage_time[1] - stage_time[0]
                            else:
                                bag_time += (stage_time[1] - stage_time[0])
                    #if status is "ResetTrial":
                    if status is not "Manual":
                        print("Manual"),
                        status = "Manual"
                        stageFlag = True
                        stage_odom = []
                        stage_img = []
                        stage_dimg = []
                        fr_odom = []
                        fr_img = []
                        fr_dimg = []
                        stage_time = [None,None]
                        final_time = [None,None]
                        reset_time = [None,None]
                        reset_times = []
                        resets = []
                if ("Staging" in msg.data):
                    #if status = "Manual":
                    #if status = "Final":
                    if status is "Reset":
                        resets.append((reset_odom,reset_img, reset_dimg))
                        reset_odom = []
                        reset_img = []
                        reset_dimg = []
                        reset_times.append(reset_time)
                        if bag_time is None:
                            bag_time = reset_time[1] - reset_time[0]
                        else:
                            bag_time += (reset_time[1] - reset_time[0])
                        reset_time = [None,None]
                    #if status is "ResetTrial":
                    if status is not "Stage":
                        print("Stage"),
                        status = "Stage"
                if (("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data)) or ("FinalRise" in msg.data):
                    #if status is "Manual":
                    #if status is "Stage":
                    if status is "Reset":
                        print("From Reset")
                        #print(msg.data)
                    if (status is not "Final") and (status is not "ResetTrial"):
                        print("Final"),
                        stageFlag = False
                        status = "Final"
                if "ResetOrangeTracking" in msg.data:
                    #if status is "Manual":
                    #if status is "Stage":
                    if status is "Final":
                        fr_odom = []
                        fr_img = []
                        fr_dimg = []
                        final_time = [None,None]
                    #if status is "ResetTrial":
                    if (status is not "Reset") and (status is not "ResetTrial"):
                        print("Reset"),
                        #print(msg.data)
                        status = "Reset"
                if "ResetTrial" in msg.data:
                    #if status is "Manual":
                    if (status is "Stage") or (status is "Final") or (status is "Reset"):
                        stage_info = (stage_odom,stage_img,stage_time,stage_dimg)
                        fr_info = (fr_odom,fr_img,final_time,fr_dimg)
                        reset_info = (resets,reset_times)
                        parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info)
                        if bag_time is None:
                            bag_time = (stage_time[1] - stage_time[0]) + (final_time[1] - final_time[0])
                        else:
                            bag_time += (stage_time[1] - stage_time[0]) + (final_time[1] - final_time[0])
                        trial_ctr += 1 
                        stage_odom = []
                        stage_img = []
                        stage_dimg = []
                        fr_odom = []
                        fr_img = []
                        fr_dimg = []
                        resets = []
                        reset_odom = []
                        reset_img = []
                        reset_dimg = []
                        img = None
                        dimg = None
                        odom = None
                        img_t = -1
                        dimg_t = -1
                        depth_t = -1
                        odom_t = -1
                        stage_time = [None,None]
                        stageFlag = True
                        final_time = [None,None]
                        reset_times = []
                        reset_time = [None,None]
                    if status is not "ResetTrial":
                        print("ResetTrial"),
                        status = "ResetTrial"

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
                    drop_ctr = 0
                    odom = msg
                odom_t = t
                if status is "Stage" and stageFlag:
                    stage_odom.append(odom)
                if status is "Final":
                    fr_odom.append(odom)
                if status is "Reset":
                    reset_odom.append(odom)


            if topic == img_topic:
                img = msg
                img_t = t
                if status is "Stage" and stageFlag:
                    img_np = np_from_compressed_image(img)
                    stage_img.append(img_np)
                    bag_events += 1
                if status is "Final":
                    img_np = np_from_compressed_image(img)
                    fr_img.append(img_np)
                    bag_events += 1
                if status is "Reset":
                    img_np = np_from_compressed_image(img)
                    reset_img.append(img_np)
                    bag_events += 1
                img_np = None

            if topic == depth_topic:
                dimg = msg
                dimg_t = t
                if status is "Stage" and stageFlag:
                    dimg_np = depthnp_from_compressed_image(dimg)
                    stage_dimg.append(dimg_np)
                    bag_events += 1
                if status is "Final":
                    dimg_np = depthnp_from_compressed_image(dimg)
                    fr_dimg.append(dimg_np)
                    bag_events += 1
                if status is "Reset":
                    dimg_np = depthnp_from_compressed_image(dimg)
                    reset_dimg.append(dimg_np)
                    bag_events += 1
                dimg_np = None

            if (img is not None) and (dimg is not None) and (odom is not None):
                if status is "Stage" and stageFlag:
                    if stage_time[0] is None:
                        stage_time[0] = max((odom_t,img_t,dimg_t))
                    stage_time[1] = max((odom_t,img_t,dimg_t,))
                if status is "Final":
                    if final_time[0] is None:
                        final_time[0] = max((odom_t,img_t,dimg_t))
                    final_time[1] = max((odom_t,img_t,dimg_t))
                if status is "Reset":
                    if reset_time[0] is None:
                        reset_time[0] = max((odom_t,img_t,dimg_t))
                    reset_time[1] = max((odom_t,img_t,dimg_t))
                odom = None
                img = None
                dimg = None

        if not stageFlag:
            stage_info = (stage_odom,stage_img,stage_time,stage_dimg)
            fr_info = None
            reset_info = (resets,reset_times)
            parseTrialData(bag_save_folder,trial_ctr,stage_info,fr_info,reset_info)
            if bag_time is None:
                bag_time = stage_time[1] - stage_time[0]
            else:
                bag_time += (stage_time[1] - stage_time[0])
        print("")
        print(bag_ctr, bag_time.secs, bag_time.nsecs, bag_events)
        bag_ctr += 1
        if total_time is None:
            total_time = bag_time
        else:
            total_time += bag_time
        total_events += bag_events
        bag_time = None
        bag_events = 0
    print(str(bag_ctr) + " Bags: " + str(total_time.secs) + " seconds of flight and " + str(total_events) + " images")

if __name__ == "__main__":
    main()
