#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *

eps = 1.0
yaw_coeff = 1.0
gamma_list = [1.0,1.0,1.0]

class DAgger:
    def __init__(self):
        inference_topic = ""
        baseline_topic = ""
        out_topic = ""
        self.inf_sub = message_filters.Subscriber(inference_topic,PoseArray)
        self.base_sub = message_filters.Subscriber(baseline_topic,PoseArray)
        self.wp_pub = Publisher(out_topic,PoseArray,queue_size=2)

    def callback(inf_msg,base_msg):
        cost = 0
        for inf_pose, base_pose, gamma in zip(inf_msg.poses,base_msg.poses, gamma_list):
            inf_p = np.array([inf_pose.position.x,inf_pose.position.y,inf_pose.position.z])
            inf_R = R.from_quat([inf_pose.orientation.x,inf_pose.orientation.y,inf_pose.orientation.z,inf_pose.orientation.w])
            base_p = np.array([base_pose.position.x,base_pose.position.y,base_pose.position.z])
            base_R = R.from_quat([base_pose.orientation.x,base_pose.orientation.y,base_pose.orientation.z,base_pose.orientation.w])
            cost += gamma*(np.linalg.norm(inf_p-base_p) + yaw_coeff*(inf_R.as_euler('ZYX')[0] - base_R.as_euler('ZYX')[0])
        print("DAgger Cost: " + str(cost))
        if cost > eps:
            self.wp_pub.publish(base_msg)
            #TODO: save t here
        else:
            self.wp_pub.publish(inf_msg)

def main():
    rospy.init_node('dagger_node')
    d = DAgger()
    ts = message_filters.ApproximateTimeSynchronizer([d.inf_sub,d.base_sub],queue_size=2, slop=1.0, allow_headerless=True)
    ts.registerCallback(d.callback)
    rospy.spin()

if __name__ == "__main__":
    main()
