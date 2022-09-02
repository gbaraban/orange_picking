#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
import torch
import numpy as np
import time, os
from geometry_msgs.msg import Pose, PoseArray, Point32, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import tf
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import ndimage
from segmentation.segmentnetarch import *
from sensor_msgs.msg import Image
from customTransforms import *
import message_filters
from open3d import geometry
from open3d import open3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *
from baseline_image_inference import BaselineOrangeFinder
from orangenetarchmulticontrol import *

import math

num_points = 1
num_phases = True 
enable_seg = True
enable_depth = False
num_images = 1
states_v = False
states_orange = False
states_rp = False

img_hz = 6
queue_time = 0.5
queue_N = int(math.ceil(img_hz*queue_time))
queue = [-1 for i in range(queue_N)]
queue_ptr = 0
queue_empty = True

def set_queued_imgs(curr):
  global queue_ptr, queue
  #retVal = (img,queue[(queue_ptr + queue_N/2)%queue_N],queue[queue_ptr])
  queue[queue_ptr] = curr #cat
  queue_ptr = (queue_ptr + 1)%queue_N
  #torch.cat(retVal,1)#change to numpy h/v stack depending on image object type

def get_queued_imgs(seg_image, num_imgs=3):
  global queue_empty
  #print("getting")
  #img = image.cpu().detach().numpy().reshape(3,h,w)
  curr = seg_image.cpu().detach().numpy()
  #print(image.shape, seg_img.shape)
  msg = curr.copy()
  #print(msg.shape)

  for i in range(num_imgs-2, -1, -1):
    if queue_empty:
      msg = np.concatenate((msg, curr))

    else:
      loc = (queue_ptr - (img_hz * i))%queue_N
      while queue[loc] is -1:
        loc -= 1
      #print("loc:", loc)
      msg = np.concatenate((msg, queue[loc]))

  set_queued_imgs(curr)
  queue_empty = False

  return msg

class StateInference:
    def __init__(self, image_topic='/camera/color/image_raw', depth_topic='/camera/aligned_depth_to_color/image_raw', odom_topic="/gcop_odom", seg_flag=True):
        self.image_topic = image_topic
        self.depth_topic = depth_topic
        self.odom_topic = odom_topic
        self.topic_list = [self.image_topic]
        if depth_topic:
            self.topic_list.append(self.depth_topic)
        if odom_topic:
            self.topic_list.append(self.odom_topic)
        self.bof = BaselineOrangeFinder(image_topic=self.image_topic, depth_topic=self.depth_topic)
        self.stamp_now = None
        if rospy.has_param("model_cap"):
            self.cap = rospy.get_param("model_cap")
        else:
            self.cap = 1.0
        self.num_images = num_images
        self.num_points = num_points
        if rospy.has_param("model_bins"):
            self.num_bins = rospy.get_param("model_bins")
        else:
            self.num_bins = 100
        if rospy.has_param("phase_end"):
            self.phase_end = rospy.get_param("phase_end")
        else:
            self.phase_end = False
        self.setMinsMaxs()
        self.dt_list = [1.0, 0.25, 0.25, 0]
        self.num_coords = 6
        self.num_channels = 3
        if enable_seg:
            self.num_channels += 1
        if enable_depth:
            self.num_channels += 1
        self.num_states = 0
        if states_v:
            self.num_states += 3
        if states_orange:
            self.num_states += 6
        if states_rp:
            self.num_states += 2
        self.num_phases = num_phases
        self.model = OrangeNet8(self.cap,self.num_images,self.num_points,self.num_bins,self.mins,self.maxs,n_outputs=self.num_coords,num_channels=self.num_channels, state_count=self.num_states, num_controls=self.num_phases)
        self.model_path = rospy.get_param("model_path")
        if rospy.has_param("mean_image_loc"):
            self.mean_image_loc = rospy.get_param("mean_iamge_loc")
        else:
            self.mean_image_loc = "../useful_models/mean_color_image.npy"
        if rospy.has_param("mean_depth_loc"):
            self.mean_depth_image_loc = rospy.get_param("mean_depth_loc")
        else:
            self.mean_depth_image_loc = "../useful_models/mean_depth_image.npy"
        self.gpu = 0
        self.loadModel()
        self.wp_pub = rospy.Publisher("/goal_points",PoseArray,queue_size=10)
        self.phase_pub = rospy.Publisher("/phase_label",Int32,queue_size=10)
        self.success_pub = rospy.Publisher("/success",Int32,queue_size=10)
        self.img_hz = 6
        self.image_dt = 0.5
        self.queue_window = int(math.ceil(self.img_hz * self.image_dt * (self.num_images - 1)))
        self.im_queue = [-1 for i in range(self.queue_window)]
        self.queue_len = 0
        self.queue_ptr = 0

    def setMinsMaxs(self):
        if self.phase_end:
            self.num_points = 1
            stage_min = [(0, -0.7, -0.2, -0.20, -0.1, -0.1)]
            stage_max = [(2.5, 0.7, 0.2, 0.20, 0.1, 0.1)]
            final_min = [(0, -0.05, -0.01, -0.1, -0.05, -0.04)]
            final_max = [(0.25, 0.05, 0.2, 0.1, 0.05, 0.04)]
            reset_min = [(-0.30, -0.1, -0.20, -0.2, -0.06, -0.04)]
            reset_max = [(0.0, 0.1, 0.0, 0.2, 0.06, 0.04)]
            grip_min = [(0,0,0,0,0,0)]
            grip_max = [(0,0,0,0,0,0)]
            self.mins = stage_min
            self.maxs = stage_max
            self.extra_mins = [final_min,reset_min,grip_min]
            self.extra_maxs = [final_max,reset_max,grip_max]
        else:
            stage_min = [(0, -0.10, -0.1, -0.20, -0.1, -0.1), (0, -0.13, -0.1, -0.20, -0.1, -0.1), (0, -0.14, -0.1, -0.20, -0.1, -0.1)]
            stage_max = [(0.75, 0.1, 0.1, 0.20, 0.1, 0.1), (0.75, 0.13, 0.1, 0.20, 0.1, 0.1), (0.75, 0.14, 0.1, 0.20, 0.1, 0.1)]
            final_min = [(-0.01, -0.01, -0.01, -0.01, -0.03, -0.03), (-0.01, -0.01, -0.01, -0.01, -0.1, -0.1), (-0.01, -0.01, -0.01, -0.01, -0.1, -0.1)]
            final_max = [(0.04, 0.01, 0.03, 0.01, 0.03, 0.04), (0.04, 0.01, 0.03, 0.01, 0.04, 0.04), (0.04, 0.01, 0.03, 0.01, 0.04, 0.04)]
            reset_min = [(-0.05, -0.01, -0.03, -0.04, -0.03, -0.03), (-0.05, -0.01, -0.03, -0.04, -0.1, -0.1), (-0.05, -0.01, -0.04, -0.01, -0.1, -0.1)]
            reset_max = [(0.0, 0.01, 0.0, 0.04, 0.03, 0.04), (0.0, 0.01, 0.0, 0.04, 0.04, 0.04), (0.0, 0.01, 0.0, 0.04, 0.04, 0.04)]
            grip_min = [(0,0,0,0,0,0),(0,0,0,0,0,0),(0,0,0,0,0,0)]
            grip_max = [(0,0,0,0,0,0),(0,0,0,0,0,0),(0,0,0,0,0,0)]
            self.mins = stage_min
            self.maxs = stage_max
            self.extra_mins = [final_min,reset_min,grip_min]
            self.extra_maxs = [final_max,reset_max,grip_max]


    def loadModel(self):
        if os.path.isfile(self.model_path):
            if not self.gpu is None:
                    checkpoint = torch.load(self.model_path,map_location=torch.device('cuda'))
            else:
                    checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            torch.no_grad()
            print("Loaded Model: ", self.model_path)
        else:
            print("No checkpoint found at: ", self.model_path)
            exit(0)
        if os.path.exits(self.mean_image_loc):
            self.mean_image = np.load(self.mean_image_loc)
            self.mean_depth_image = np.load(self.mean_depth_image_loc)
        else:
            print("No file found at: ", self.mean_image_loc)
            exit(0)

    def predictInference(self,tensor,states=None):
        if states is None:
            logits = self.model(tensor)
        else:
            logits = self.model(tensor,states)
        logits = logits.cpu()
        if self.model.num_controls > 1:
            classifier = logits[:,:self.model.num_controls]
            logits = logits[:, self.model.num_controls:]
            softmax = nn.Softmax(dim=1)
            predicted_phases = softmax(classifier_logits).to('cpu').detach().numpy()
            phase_pred = np.argmax(predicted_phases, axis=1)[0]
            logits = logits.view(1,self.model.outputs,self.model.num_points,self.model.bins*self.model.num_controls).detach().numpy()
            logits = logits[:, :, :, np.arange(self.model.bins*phase_pred, self.model.bins*(phase_pred+1))]
        else:
            logits = logits.view(1,self.model.outputs,self.model.num_points,self.model.bins*self.model.num_controls).detach().numpy()
            max_pred = None
        predict = np.argmax(logits,axis=3)
        return predict, max_pred

    def tensor2wp(self,pred,phase):
        retList = []
        if (phase is not None) or (phase == 0):
            bin_min = self.mins
            bin_max = self.maxs
        else:
            bin_min = self.extra_mins[phase-1]
            bin_max = self.extra_maxs[phase-1]
        for pt in range(self.model.num_points):
            point = []
            for coord in range(self.model.outputs):
                if not self.regression:
                    bin_size = (bin_max[pt][coord] - bin_min[pt][coord])/float(num_bins)
                    point.append(bin_min[pt][coord] + bin_size*pred[0,coord,pt])
                else:
                    point.append(logits[0,pt, coord])
            if self.__spherical:
                pointList = [np.array(point)]
                pl = sphericalToXYZ().__call__(pointList)
                print("Before: ", pl.shape)
                point = pl.flatten()
                print("After: ", point.shape)
            point = np.array(point)
            retList.append(point)
        return retList

    def publishWP(self,wpList,phase):
        msg = PoseArray()
        header = std_msgs.msg.Header()
        if self.phase_end:
            max_v = 0.3
            min_dt = 4.0
            d = np.linalg.norm(wpList[0][0:3])
            dt = d/max_v
            dt = max((min_tf,dt))
            header.stamp = rospy.Duration(dt)
        else:
            if phase is None:
                header.stamp = rospy.Duration(0.0)
            else:
                dt = self.dt_list[phase]
                dt = dt*self.num_points
                header.stamp = rospy.Duration(dt)

        for wp in wpList:
            pt_pose = Pose()
            pt_pose.position.x = wp[0]
            pt_pose.position.y = wp[1]
            pt_pose.position.z = wp[2]
            if len(wp) == 4:
                R_quat = R.from_euler('Z',wp[3]).as_quat()
            else:
                R_quat = R.from_euler('Z',wp[3:]).as_quat()
            pt_pose.orientation.x = R_quat[0]
            pt_pose.orientation.y = R_quat[1]
            pt_pose.orientation.z = R_quat[2]
            pt_pose.orientation.w = R_quat[3]
            msg.poses.append(pt_pose)
        msg.header = header
        self.wp_pub.publish(msg)

    def publishPhase(self,phase):
        #print("Phase is : " + str(phase))
        msg = Int32()
        msg.data = phase
        self.phase_pub.publish(msg)
        success_msg = Bool()
        success_msg.data = (phase == 4)
        self.success_pub.publish(success_msg)

    def createTensor(self,image,depth,seg):
        im_tensor = torch.tensor(image - self.mean_image)
        print("Image shape: " + str(im_tensor.shape))
        if depth is not None:
            d_tensor = torch.tensor(depth - self.mean_depth_image)
            im_tensor = torch.cat((im_tensor,d_tensor),1)
        if seg is not None:
            s_tensor = torch.tensor(seg)
            im_tensor = torch.cat((im_tensor,s_tensor),1)
        im_tensor = self.getQueueImages(im_tensor)
        return im_tensor 
    
    def getQueueImages(self,im):
        if self.queue_len < len(self.queue):
            self.queue[self.queue_ctr] = im
            self.queue_ctr += 1
            self.queue_ctr = self.queue_ctr%len(self.queue)
            self.queue_len += 1
            return None
        output = im.copy()
        print("Output Shape: " + output.shape)
        for i in range(1,self.num_images):
            loc = self.queue_ptr - (self.image_hz*i)
            loc = loc%len(self.queue)
            output = torch.cat((output,self.queue[loc]))
        print("Output Shape: " + output.shape)
        self.queue[self.queue_ctr] = im
        self.queue_ctr += 1
        self.queue_ctr = self.queue_ctr%len(self.queue)
        return output

    def getBodyV(self,odom,rot):
        rotMat = rot.as_dcm()
        body_v = []
        body_v.append(odom.twist.twist.linear.x)
        body_v.append(odom.twist.twist.linear.y)
        body_v.append(odom.twist.twist.linear.z)
        body_v = list(np.matmul(rotMat.T, np.array(body_v)))
        body_v.append(odom.twist.twist.angular.x)
        body_v.append(odom.twist.twist.angular.y)
        body_v.append(odom.twist.twist.angular.z)
        return list(body_v)

    def getOrangePose(self,image,depth_image,seg_np):
        area, extra_area = bof.findArea(image,seg_np)
        camera_intrinsics = CameraIntrinsics()
        ret = bof.getCloudData(depth_image, camera_intrinsics, area, extra_area)
        if ret is None:
            return None
        pts,mean_pt,extra_pts = ret
        orientation = bof.__find_plane(extra_pts,mean_pt)
        return list(np.concatenate((mean_pt,orientation)))

    def callback(self, image_data, depth_image_data, odom):
        print("Reaching Callback")
        t1 = time.time()
        self.stamp_now = image_data.header.stamp
        #Build Tensor Up
        image = bof.__rosmsg2np(image_data)
        if enable_depth or enable_orange_pos:
            depth_image = bof.__depthnpFromImage(depth_image_data)
        else:
            depth_image = None
        if enable_seg or enable_orange_pos:
            seg_input = bof.__process4model(image)
            seg_np = bof.__segmentationInference(image_tensor)
        else:
            seg_np = None
        tensor = self.createTensor(image,depth_image,seg_np)
        #Build states
        quat = odom.pose.pose.orientation
        rot = R.from_quat((quat.x,quat.y,quat.z,quat.w))
        states = []
        if states_v:
            v = self.getBodyV(odom,rot)
            if v is None:
                return
            states+=(v)
        if states_orange:
            op = self.getOrangePose()
            if op is None:
                return
            states.append(op)
        if states_rp:
            rp = rot.as_euler("ZYX")[1:]
            states.append(rp)
        if len(states) is 0:
            states = None
        else:
            states = np.array(states)
        #Run Inf
        pred, phase = self.predictInference(tensor,states)
        #Convert to Waypoint
        wp = self.tensor2wp(pred,phase)
        #Publish
        self.publishWP(wp,phase)
        if phase is not None:
            self.publishPhase(phase)

def main():
    rospy.init_node('states_image_inference')
    global num_points 
    if rospy.has_param("num_points"):
        num_points = rospy.get_param("num_points")
    else:
        num_points = 3
    global num_phases 
    if rospy.has_param("num_phases"):
        num_phases = rospy.get_param("num_phases")
    else:
        num_phases = 4
    global enable_seg
    if rospy.has_param("enable_seg"):
        enable_seg = rospy.get_param("enable_seg")
    else:
        enable_seg = True
    global enable_depth
    if rospy.has_param("enable_depth"):
        enable_depth = rospy.get_param("enable_depth")
    else:
        enable_depth = True
    global num_images
    if rospy.has_param("num_images"):
        num_images = rospy.get_param("num_images")
    else:
        num_images = 1
    global states_v
    if rospy.has_param("states_v"):
        states_v = rospy.get_param("states_v")
    else:
        states_v = False
    global states_orange
    if rospy.has_param("states_orange"):
        states_orange = rospy.get_param("states_orange")
    else:
        states_orange = False 
    global states_rp
    if rospy.has_param("states_rp"):
        states_rp = rospy.get_param("states_rp")
    else:
        states_rp = True
    si = StateInference()
    ts = message_filters.ApproximateTimeSynchronizer(si.topic_list, queue_size=2, slop=1.0,  allow_headerless=True)
    ts.registerCallback(si.callback)
    rospy.spin()


if __name__ == "__main__":
    main()
