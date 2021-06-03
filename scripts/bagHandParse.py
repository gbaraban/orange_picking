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
from scipy.spatial.transform import Slerp
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

    depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
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

def saveData(save_loc, odom, coeff,time):
    data_dict = {}
    data_dict["data"] = odom
    data_dict["coeff"] = coeff
    data_dict["time"] = time
    with open(save_loc + '/data.pickle_no_parse','wb') as f:
        pickle.dump(data_dict,f,pickle.HIGHEST_PROTOCOL)

def calculateOrangeState(odom,image_msg,dimg,bof):
    p = (odom.transform.translation.x,odom.transform.translation.y,odom.transform.translation.z)
    quat = (odom.transform.rotation.x,
            odom.transform.rotation.y,
            odom.transform.rotation.z,
            odom.transform.rotation.w)
    img_np = np_from_image(image_msg)
    dimg_np = depthnp_from_image(dimg)
    retval = bof.process_loc(img_np,dimg_np,p,quat,world=True)
    if retval is None:
        return None
    orange_p = retval[0]
    orange_R = R.from_dcm(retval[1])
    return (orange_p,orange_R) #postion as np.array, Rotation as Rotation object

def createTrajectory(odom, orange_p, orange_R):
    start_p = np.array((odom.transform.translation.x, odom.transform.translation.y, odom.transform.translation.z))
    start_R = R.from_quat([odom.transform.rotation.x, odom.transform.rotation.y,
                        odom.transform.rotation.z, odom.transform.rotation.w])
    start_yaw = start_R.as_euler('ZYX')[0]
    orange_yaw = orange_R.as_euler('ZYX')[0]
    max_v = 0.15
    min_tf = 4.0
    offset_p = np.array([0.95,-0.05,0.15]) 
    offset_yaw = np.pi
    start_p_in_orange_frame = orange_R.inv().apply(start_p - orange_p)
    o_frame_error = start_p_in_orange_frame - offset_p
    if (o_frame_error[0] < 0.05) and (np.linalg.norm(o_frame_error[1:]) < 0.05):
        print("Starting in tracking mode") 
        offset_p = np.array([0.60,-0.05,0.3])
        o_frame_error = start_p_in_orange_frame - offset_p
    #else: TODO: add multi-step phases here
    distance = np.linalg.norm(o_frame_error)
    tf = max(distance/max_v,min_tf)
    target_p = orange_p + orange_R.apply(offset_p)
    target_R = orange_R*R.from_euler('z',offset_yaw)
    target_yaw = target_R.as_euler('ZYX')[0]
    error_p = target_p - start_p
    error_yaw = target_yaw - start_yaw
    deg = 9
    dim = 4
    constraints = np.zeros((deg+1,dim))
    basis = np.zeros((deg+1,deg+1))
    constraints[0,0] = error_p[0]
    constraints[0,1] = error_p[1]
    constraints[0,2] = error_p[2]
    constraints[0,3] = error_yaw
    basis[0:5,:] = basisMatrix(tf)
    basis[5:,:] = basisMatrix(0)
    coeff = np.linalg.solve(basis,constraints)
    return coeff

def basisMatrix(time):
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

def coeffToWP(coeff,x0,times):
    wp_list = []
    for time in times:
        basis = basisMatrix(time)
        out = np.matmul(basis,coeff)
        posyaw = x0 + out[0,:]
        velyawrate = out[1,:]
        wp_list.append((posyaw,velyawrate))
    return wp_list

def saveTrajectories(bag_save_folder,trial_ctr,coeff_list,img_list,odom_list,tf):
    trial_folder = bag_save_folder + "/trial" + str(trial_ctr)
    os.makedirs(trial_folder)
    for ii, img in enumerate(img_list):
        Img.fromarray(img).save(trial_folder + "/image" + str(ii) + ".png")
        #cv2.imwrite(fr_folder + "/depth_image" + str(ii) + ".png", fr_info[3][ii])
    saveData(trial_folder, odom_list, coeff_list,tf)

def main():
    vrpn_topic = "/vrpn_client/matrice/pose"
    img_topic = "/camera/color/image_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    success_topic = "/magnet_values"
    status_topic = "/rqt_gui/system_status"

    parser = argparse.ArgumentParser()
    parser.add_argument('bag_dir', help="bag dir")
    args = parser.parse_args()

    folder_list = os.listdir(args.bag_dir)
    bag_list = ["/" + temp for temp in folder_list]
    for folder_name in folder_list:
        if not os.path.isdir(args.bag_dir + folder_name):
            continue
        bag_list = bag_list + [folder_name + "/" + temp for temp in os.listdir(args.bag_dir + folder_name)]
    print(bag_list)
    bag_ctr = 0
    bag_time = None
    bag_events = 0
    total_time = None
    total_events = 0
    for bag_name in bag_list:
        if not bag_name.endswith(".bag"):
            continue
        print(str(bag_ctr) + ": " + bag_name)
        fname = args.bag_dir + "/" + bag_name
        bag = rosbag.Bag(fname)

        bag_save_folder = args.bag_dir + 'bagHandParse/bag' + str(bag_ctr)
        if os.path.isdir(bag_save_folder):
            shutil.rmtree(bag_save_folder)
        os.makedirs(bag_save_folder)

        #Find the orange state
        img = None
        dimg = None
        odom = None
        bof = BaselineOrangeFinder()
        orange_state = None
        for topic, msg, t in bag.read_messages(topics=[vrpn_topic, img_topic, depth_topic]):
            if topic == vrpn_topic:
                odom = msg
            if topic == img_topic:
                img = msg
            if topic == depth_topic:
                dimg = msg
            if (odom is not None) and (img is not None) and (dimg is not None):
                retval = calculateOrangeState(odom,img,dimg,bof)
                if retval is not None:
                    orange_state = retval
                dimg = None
        if orange_state is None:
            print("No Orange Found In Bag")
            continue
        orange_p = np.array(orange_state[0])
        orange_R = orange_state[1]

        trial_ctr = 0
        odom_list = []
        img_list = []
        coeff_list = []
        img = None
        dimg = None
        odom = None
        start_time = None
        end_time = None
        drop_ctr = 0
        drop_reset_thresh = 10

        for topic, msg, t in bag.read_messages(topics=[vrpn_topic, img_topic, depth_topic]):

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

            if topic == img_topic:
                img = msg
                img_t = t

            if topic == depth_topic:
                dimg = msg
                dimg_t = t

            if (img is not None)and (odom is not None):# and (dimg is not None):
                start_p = np.array((odom.transform.translation.x, odom.transform.translation.y, odom.transform.translation.z))
                start_R = np.array([odom.transform.rotation.x, odom.transform.rotation.y, odom.transform.rotation.z, odom.transform.rotation.w])
                odom_data = [start_p, start_R]
                odom_list.append(odom_data)
                coeff_list.append(createTrajectory(odom,orange_p,orange_R))
                img_np = np_from_image(img)
                img_list.append(img_np)
                bag_events += 1
                img_np = None
                odom = None
                img = None
                dimg = None
                if start_time is None:
                    start_time = min((img_t,odom_t))
                end_time = max((img_t,odom_t))

        tf_duration = (end_time - start_time)
        tf = tf_duration.secs + (tf_duration.nsecs/1e9)
        saveTrajectories(bag_save_folder,trial_ctr,coeff_list,img_list,odom_list,tf)
        print("")
        print(bag_ctr, bag_events)
        bag_ctr += 1
        total_events += bag_events
        bag_events = 0
    print(str(bag_ctr) + " Bags: " + str(total_events) + " images")

if __name__ == "__main__":
    main()
