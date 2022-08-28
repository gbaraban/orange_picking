import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
import torch
import numpy as np
import time, os
from geometry_msgs.msg import Pose, PoseArray, Point32
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

import math

enable_multi_control = True 
enable_seg = True
enable_depth = False
enable_states = True
num_images = 1
states_v = False
states_orange = False
states_rp = False

if enable_multi_control:
    from orangenetarchmulticontrol import *
elif enable_states:
    from orangenetarchstates import *
else:
    from orangenetarch import *


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
        self.bof = BaselineOrangeFinder(image_topic="/camera/color/image_raw", depth_topic="/camera/aligned_depth_to_color/image_raw")
        self.stamp_now = None
        self.cap = ...
        self.num_images = ...
        self.num_points = ...
        self.num_bins = ...
        self.mins = ...
        self.maxs = ...
        self.num_coords = ...
        self.num_channels = ...
        self.num_states = ...
        self.num_phases = ...
        self.model = OrangeNet8(capacity,self.num_images,self.num_points,self.bins,self.mins,self.maxs,n_outputs=self.num_coords,num_channels=self.num_channels, state_count=self.num_states, num_controls=self.num_phases)
        self.model_path = ...
        self.mean_image_loc = ...
        self.mean_depth_image_loc = ...
        self.gpu = 0
        self.loadModel()
        self.wp_pub = rospy.Publisher("/goal_points",PoseArray,queue_size=10)
        self.phase_pub = rospy.Publisher("/phase_label",Int32,queue_size=10)

    def loadModel(self):
        if os.path.isfile(self.model_path):
            if not self.gpu is None:
                    checkpoint = torch.load(self.model_path,map_location=torch.device('cuda'))
            else:
                    checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            torch.no_grad()
            print("Loaded Model: ", self.__modelload)
        else:
            print("No checkpoint found at: ", self.__modelload)
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
        if phase is None:
            header.stamp = rospy.Duration(0.0)
        else:
            header.stamp = rospy.Duration(self.dt_list[phase])
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
        msg = Int32()
        msg.data = phase
        self.phase_pub.publish(msg)

    def createTensor(self,image,depth,seg):
        if self.num_images > 1:
            print("too many images")
            return 

    def callback(self, image_data, depth_image_data, odom):
        print("Reaching Callback")
        t1 = time.time()
        self.stamp_now = image_data.header.stamp
        #Build Tensor Up
        image = bof.__rosmsg2np(image_data)
        if enable_depth or enable_orange_pos:
            depth_image = bof.__depthnpFromImage(depth_image_data)
            depth_image -= self.__mean_depth_image
        else:
            depth_image = None
        if enable_seg or enable_orange_pos:
            seg_input = bof.__process4model(image)
            seg_np = bof.__segmentationInference(image_tensor)
        else:
            seg_np = None
        #TODO: Add multiply image stacking later
        tensor = self.createTensor(image,depth_image,seg_np)
        #Build states
        if enable_orange_pos:
            area, extra_area = bof.findArea(image,seg_np)
            camera_intrinsics = CameraIntrinsics()
            trans, rot = None, None
            try:
                (trans,rot) = self.listener.lookupTransform('world', 'camera_depth_optical_frame', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("Lookup Failed")
                return
            if trans is None or rot is None:
                print("tf None")
                return

            t5 = time.time()
            ret = self.getCloudData(depth_image, camera_intrinsics, area, extra_area)
            if ret is None:
                return
            t6 = time.time()
            pts,mean_pt,extra_pts = ret
            self.publishData(pts,extra_pts,Trans=trans, Rot=rot, mean_pt=mean_pt)

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
    si = StateInference()
    ts = message_filters.ApproximateTimeSynchronizer(si.topic_list, queue_size=2, slop=1.0,  allow_headerless=True)
    ts.registerCallback(si.callback)
    rospy.spin()


if __name__ == "__main__":
    main()
