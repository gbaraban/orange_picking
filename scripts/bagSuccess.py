import rosbag
import numpy as np
from scipy import ndimage
from segmentation.segmentnetarch import *
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
import PIL.Image
import matplotlib.pyplot as plt
from visual_servoing_pose_estimate import BaselineOrangeFinder
import cv2
import pickle

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
  data = data[:,:,0].astype(np.float64)
  cm_from_pixel = 0.095
  return data*cm_from_pixel


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

def parseBaselineBag(bag_dir,bag_name):
    print("Parsing: ",bag_name)
    vrpn_topic = "/vrpn_client/matrice/pose"
    odom_topic = "/gcop_odom"
    img_topic = "/camera/color/image_raw/compressed"
    depth_topic = "/camera/aligned_depth_to_color/image_raw/compressedDepth"
    success_topic = "/magnet_values"
    status_topic = "/rqt_gui/system_status"
    fname = bag_name #bag_dir + "/" + bag_name
    try:
        bag = rosbag.Bag(fname)
    except:
        print(fname + " is not loadable")
        return None

    trials = []
    resets = 0
    odom = None
    start_t = None
    startOdom = None
    stage_t = None
    stageOdom = None
    status = "Start"
    stageFlag = True
    drop_ctr = 0
    drop_reset_thresh = 10
    success_ctr = 0
    success_thresh = 550
    success_num_thresh = 4
    for topic, msg, t in bag.read_messages(topics=[status_topic, odom_topic, vrpn_topic, success_topic]):
        if topic == vrpn_topic:
            p = [msg.transform.translation.x,
                 msg.transform.translation.y,
                 msg.transform.translation.z]
            q = [msg.transform.rotation.x,
                 msg.transform.rotation.y,
                 msg.transform.rotation.z,
                 msg.transform.rotation.w]
            odom = np.hstack((p,R.from_quat(q).as_euler('ZYX')))

        if topic == odom_topic:
            p = [msg.pose.pose.position.x,
                 msg.pose.pose.position.y,
                 msg.pose.pose.position.z]
            q = [msg.pose.pose.orientation.x,
                 msg.pose.pose.orientation.y,
                 msg.pose.pose.orientation.z,
                 msg.pose.pose.orientation.w]
            odom = np.hstack((p,R.from_quat(q).as_euler('ZYX')))
        #print(status)
        """if topic == success_topic:
            if np.linalg.norm(msg.data) > success_thresh:
                success_ctr += 1
            else:
                success_ctr = 0
            if (success_ctr >= success_num_thresh) and (status is not "ResetTrial") and (status is not "Manual"):
                print("Magnet Success")
                if start_t:
                    temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                    trials.append(temp_dict)
                status = "ResetTrial"
                start_t = None
                startOdom = None
                stage_t = None
                stageOdom = None
                end_t = None
                stageFlag = True
"""
        if (odom is not None) and (topic == status_topic):
            if status is "Start":
                if "Manual" in msg.data:
                    status = "Manual"
                if "Hover" in msg.data:
                    status = "Hover"
                if "Staging" in msg.data:
                    status = "Stage"
                    start_t = t
                    startOdom = odom
                if ("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data):
                    print("Start to Tracking Transition... Weird")
                    exit(0)
                    return
                if ("FinalRise" in msg.data):
                    print("Start to Final Transtion... Weird")
                    exit(0)
                    return
                if "ResetOrangeTracking" in msg.data:
                    status = "Reset"
                if "PathFollow" in msg.data:
                    status = "PathFollow"
                if "ResetTrial" in msg.data:
                    status = "ResetTrial"
                continue
            if status is "Manual":
                if "Manual" in msg.data:
                    continue
                if "Hover" in msg.data:
                    status = "Hover"
                if "Staging" in msg.data:
                    status = "Stage"
                    start_t = t
                    startOdom = odom
                if ("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data):
                    print("Manual to Tracking Transition... Weird")
                    exit(0)
                    return
                if ("FinalRise" in msg.data):
                    print("Manual to Final Transtion... Weird")
                    exit(0)
                    return
                if "ResetOrangeTracking" in msg.data:
                    print("Manual to Reset Transtion... Weird")
                    exit(0)
                    return
                if "ResetTrial" in msg.data:
                    print("Manual to ResetTrial Transtion... Weird")
                    exit(0)
                    return
                if ("PathFollow" in msg.data):
                    status = "PathFollow"
                    start_t = t
                continue
            if status is "Hover":
                if "Manual" in msg.data:
                    status = "Hover"
                    start_t = None
                if "Hover" in msg.data:
                    continue
                if "Staging" in msg.data:
                    status = "Stage"
                    start_t = t
                    startOdom = odom
                if ("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data):
                    print("Hover to Tracking Transition... Weird")
                    exit(0)
                    return
                if ("FinalRise" in msg.data):
                    print("Hover to Final Transtion... Weird")
                    exit(0)
                    return
                if "ResetOrangeTracking" in msg.data:
                    print("Hover to Reset Transtion... Weird")
                    exit(0)
                    return
                if "ResetTrial" in msg.data:
                    print("Hover to ResetTrial Transtion... Weird")
                    exit(0)
                    return
                if ("PathFollow" in msg.data):
                    status = "PathFollow"
                    start_t = t
                    startOdom = odom
                continue
            if status is "Stage":
                if "Manual" in msg.data:
                    print("Stage Intervention")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":None}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "Manual"
                if "Hover" in msg.data:
                    print("Stage Abort")
                    if start_t and False:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "Hover"
                if "Staging" in msg.data:
                    continue
                if ("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data):
                    if stageOdom is None:
                        stage_t = t
                        stageOdom = odom
                    status = "Track"
                    stageFlag = False
                if ("FinalRise" in msg.data):
                    status = "Final"
                    stageFlag = False
                    if stageOdom is None:
                        stageOdom = odom
                        stage_t = t
                if "ResetOrangeTracking" in msg.data:
                    status = "Reset"
                if "ResetTrial" in msg.data:
                    print("Stage Success")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "ResetTrial"
                if ("PathFollow" in msg.data):
                    print("Stage to PathFollow Transtion... Weird")
                    exit(0)
                    return
                continue
            if status is "Track":
                if "Manual" in msg.data:
                    print("Track Intervention")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":None}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "Manual"
                if "Hover" in msg.data:
                    print("Track Abort")
                    if start_t and False:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "Hover"
                if "Staging" in msg.data:
                    status = "Stage"
                if ("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data):
                    continue
                if ("FinalRise" in msg.data):
                    status = "Final"
                if "ResetOrangeTracking" in msg.data:
                    resets += 1
                    status = "Reset"
                if "ResetTrial" in msg.data:
                    print("Track Success")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "ResetTrial"
                if ("PathFollow" in msg.data):
                    print("Track to PathFollow Transition... Weird")
                    exit(0)
                    return
                continue
            if status is "Final":
                if "Manual" in msg.data:
                    print("Final Intervention")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":None}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "Manual"
                if "Hover" in msg.data:
                    print("Final Abort")
                    if start_t and False:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "Hover"
                if "Staging" in msg.data:
                    print("Final to Staging Transition... Weird")
                    exit(0)
                    return
                if ("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data):
                    print("Final to Tracking Transition... Weird")
                    exit(0)
                    return
                if ("FinalRise" in msg.data):
                    continue
                if "ResetOrangeTracking" in msg.data:
                    status = "Reset"
                    resets += 1
                if "ResetTrial" in msg.data:
                    print("Final Success")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "ResetTrial"
                if ("PathFollow" in msg.data):
                    print("Final to PathFollow Transition... Weird")
                    exit(0)
                    return
                continue
            if status is "Reset":
                if "Manual" in msg.data:
                    print("Reset Intervention")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":None}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "Manual"
                if "Hover" in msg.data:
                    print("Reset Abort")
                    if start_t and False:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "Hover"
                if "Staging" in msg.data:
                    status = "Stage"
                if ("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data):
                    status = "Track"
                if ("FinalRise" in msg.data):
                    status = "Final"
                if "ResetOrangeTracking" in msg.data:
                    continue
                if "ResetTrial" in msg.data:
                    print("Reset Success")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    stageFlag = True
                    status = "ResetTrial"
                if ("PathFollow" in msg.data):
                    print("Reset to PathFollow Transition... Weird")
                    exit(0)
                    return
                continue
            if status is "ResetTrial":
                if "Manual" in msg.data:
                    status = "Manual"
                if "Hover" in msg.data:
                    status = "Hover"
                if "Staging" in msg.data:
                    status = "Stage"
                    start_t = t
                if ("OrangeTracking" in msg.data) and ("ResetOrangeTracking" not in msg.data):
                    status = "Track"
                    start_t = t
                if ("FinalRise" in msg.data):
                    print("ResetTrial to Final Transtion... Weird")
                    exit(0)
                    return
                if "ResetOrangeTracking" in msg.data:
                    print("ResetTrial to Reset Transtion... Weird")
                    exit(0)
                    return
                if "ResetTrial" in msg.data:
                    continue
                if ("PathFollow" in msg.data):
                    print("ResetTrial to PathFollow Transition... Weird")
                    exit(0)
                    return
                continue
    if not stageFlag:
        if start_t:
            temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":resets,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
            trials.append(temp_dict)
    return trials

def parseBag(bag_dir,bag_name):
    print("Parsing: ",bag_name)
    odom_topic = "/gcop_odom"
    status_topic = "/rqt_gui/system_status"
    img_topic = "/camera/color/image_raw/compressed"
    depth_topic = "/camera/aligned_depth_to_color/image_raw/compressedDepth"
    success_topic = "/magnet_values"
    success_ctr = 0
    success_thresh = 550
    success_num_thresh = 4
    fname = bag_name #bag_dir + "/" + bag_name
    try:
        bag = rosbag.Bag(fname)
    except:
        print(fname + " is not loadable")
        return None

    bof = BaselineOrangeFinder()
    stage_ctr = 0
    stage_threshold = 5

    trials = []
    img = None
    dimg = None
    odom = None
    start_t = None
    startOdom = None
    stage_t = None
    stageOdom = None
    stageFlag = True
    orange_loc = None
    status = "Start"
    for topic, msg, t in bag.read_messages(topics=[odom_topic,img_topic,depth_topic,status_topic,success_topic]):
        if topic == img_topic:
            img = msg
        if topic == depth_topic:
            dimg = msg
        if (odom is not None) and (topic == status_topic):
            if status is "Start":
                if "Manual" in msg.data:
                    status = "Manual"
                if "Hover" in msg.data:
                    status = "Hover"
                if "PathFollow" in msg.data:
                    status = "PathFollow"
                    start_t = t
                    startOdom = odom
                if "ResetTrial" in msg.data:
                    status = "ResetTrial"
                continue
            if status is "Manual":
                if "Manual" in msg.data:
                    continue
                if "Hover" in msg.data:
                    status = "Hover"
                if "PathFollow" in msg.data:
                    status = "PathFollow"
                    start_t = t
                    startOdom = odom
                if "ResetTrial" in msg.data:
                    print("Manual to ResetTrial Transition.. Weird")
                    exit(0)
                    return
                continue
            if status is "Hover":
                if "Manual" in msg.data:
                    status = "Manual"
                if "Hover" in msg.data:
                    continue
                if "PathFollow" in msg.data:
                    status = "PathFollow"
                    start_t = t
                    startOdom = odom
                if "ResetTrial" in msg.data:
                    print("Manual to ResetTrial Transition.. Weird")
                    exit(0)
                    return
                continue
            if status is "PathFollow":
                if "Manual" in msg.data:
                    print("PathFollow Intervention")
                    if start_t:
                        if (stageOdom is not None):
                            print("Staging Done")
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":None,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":None}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    status = "Manual"
                if "Hover" in msg.data:
                    print("PathFollow Abort")
                    if start_t and False:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":None,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":None}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    status = "Hover"
                if "PathFollow" in msg.data:
                    continue
                if "ResetTrial" in msg.data:
                    print("PathFollow Success")
                    if start_t:
                        temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":None,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":odom}
                        trials.append(temp_dict)
                    start_t = None
                    startOdom = None
                    stage_t = None
                    stageOdom = None
                    end_t = None
                    status = "ResetTrial"
                continue
            if status is "ResetTrial":
                if "Manual" in msg.data:
                    status = "Manual"
                if "Hover" in msg.data:
                    status = "Hover"
                if "PathFollow" in msg.data:
                    status = "PathFollow"
                    start_t = t
                    startOdom = odom
                if "ResetTrial" in msg.data:
                    continue
                continue
        if topic == odom_topic:
            p = [msg.pose.pose.position.x,
                 msg.pose.pose.position.y,
                 msg.pose.pose.position.z]
            q = [msg.pose.pose.orientation.x,
                 msg.pose.pose.orientation.y,
                 msg.pose.pose.orientation.z,
                 msg.pose.pose.orientation.w]
            odom = np.hstack((p,R.from_quat(q).as_euler('ZYX')))
            if (status is "PathFollow") and (img) and (dimg) and (stage_t is None):
                img_np = np_from_compressed_image(img)
                dimg_np = depthnp_from_compressed_image(dimg)
                orange_loc = bof.process_loc(img_np,dimg_np,odom[0:3],R.from_euler('ZYX',odom[3:6]).as_quat())
                #TODO: check coordinates of orange_loc
                flag = check_staged(orange_loc)
                if flag > 0:
                    stage_ctr += 1
                    if stage_ctr > stage_threshold:
                        stage_t = t
                        stageOdom = odom
                elif flag < 0:#TODO: this might go away
                    stage_ctr = 0
                    stage_t = None
                    stageOdom = None
                img = None
                dimg = None

    if status is "PathFollow":
        print("PathFollow End")
        if start_t:
            temp_dict = {"startTime":start_t,"stageTime":stage_t,"endTime":t,"resets":None,"status":status,"startOdom":startOdom,"stageOdom":stageOdom,"finalOdom":None}
            trials.append(temp_dict)
    return trials 

def readData(data):
    success = 0
    trials = 0
    stages = 0
    total_t = []
    stage_t = []
    pick_t = []
    total_d = []
    stage_d = []
    pick_d = []
    resets = 0
    for arr in data:
        for trial in arr:
            trials += 1
            if trial["resets"] is not None:
                resets += trial["resets"]
            if trial["finalOdom"] is not None:
                success +=1
                stages += 1
                total_d.append(np.linalg.norm(trial["finalOdom"][0:3] - trial["startOdom"][0:3]))
                total_t.append(durToFloat(trial["endTime"] - trial["startTime"]))
                if trial["stageOdom"] is not None:
                    pick_d.append(np.linalg.norm(trial["finalOdom"][0:3] - trial["stageOdom"][0:3]))
                    pick_t.append(durToFloat(trial["endTime"] - trial["stageTime"]))
                    stage_d.append(np.linalg.norm(trial["stageOdom"][0:3] - trial["startOdom"][0:3]))
                    stage_t.append(durToFloat(trial["stageTime"] - trial["startTime"]))
            elif trial["stageOdom"] is not None:
                stages += 1
                stage_d.append(np.linalg.norm(trial["stageOdom"][0:3] - trial["startOdom"][0:3]))
                stage_t.append(durToFloat(trial["stageTime"] - trial["startTime"]))
    print(resets, " Total Resets")
    print(trials, " Total Trials")
    print(success, " Total Successes: ", float(success)/trials)
    print(stages, " Total Stages: ", float(stages)/trials)
    print(np.mean(total_t), " Total T (s)")
    print(np.mean(pick_t), " Pick T (s)")
    print(np.mean(stage_t), " Stage T (s)")
    total_v = [float(d)/t for d,t in zip(total_d,total_t)]
    pick_v = [float(d)/t for d,t in zip(pick_d,pick_t)]
    stage_v = [float(d)/t for d,t in zip(stage_d,stage_t)]
    print(np.mean(total_v), " Total v")
    print(np.mean(pick_v), " Pick v")
    print(np.mean(stage_v), " Stage v")

def durToFloat(d):
    return d.secs + d.nsecs/1e9

def check_staged(orange_loc):
    if orange_loc is None:
        return 0
    if orange_loc[0] is None:
        return 0
    if orange_loc[0][2] < 1.0 and orange_loc[0][2] > 0.5:
        if np.linalg.norm(orange_loc[0][0:2]) < 0.4:
            return 1
    return -1

def recursive_bag_search(loc):
    r = []
    for file in os.listdir(loc):
        if file.endswith(".bag"):
            r.append(loc + "/" + file)
        elif os.path.isdir(loc + "/" + file):
            r += recursive_bag_search(loc + "/" + file)
    return r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_dir', help='bag dir')
    parser.add_argument("--baseline",type=bool,default=False)
    args = parser.parse_args()
    
    if "pickle" in args.bag_dir:
        with open(args.bag_dir,'rb') as f:
            data = pickle.load(f)#,encoding='latin1')
        readData(data)
        return
    
    bag_list = recursive_bag_search(args.bag_dir)

#    folder_list = [""] + os.listdir(args.bag_dir)
#    for folder_name in folder_list:
#        if not os.path.isdir(args.bag_dir + folder_name):
#            continue
#        folder_list = folder_list + [folder_name + "/" + temp for temp in os.listdir(args.bag_dir + folder_name) if os.path.isdir(args.bag_dir + folder_name + "/" + temp)]
#    bag_list = []
#    for folder_name in folder_list:
#        if not os.path.isdir(args.bag_dir + folder_name):
#            continue
#        bag_list = bag_list + [folder_name + "/" + temp for temp in os.listdir(args.bag_dir + folder_name) if temp.endswith(".bag")]

    data = []
    for bag_name in bag_list:
        if not bag_name.endswith(".bag"):
            continue
        if args.baseline:
            ret_val = parseBaselineBag(args.bag_dir,bag_name)
        else:
            ret_val = parseBag(args.bag_dir,bag_name)
        if ret_val:
            data.append(ret_val)

    with open(args.bag_dir+"/success_data.pickle",'wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    readData(data)

if __name__ == '__main__':
    main()
