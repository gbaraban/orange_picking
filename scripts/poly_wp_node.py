#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseArray, Point32, TransformStamped
import tf2_ros
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class tf_class:
    def __init__(self,pos, Rot):
        self.p = pos
        self.Rot = Rot
    
    def __mul__(self,other):
        temp_R = self.Rot * other.Rot
        temp_p = self.Rot.apply(other.p) + self.p
        return tf_class(temp_p,temp_R)

    def __str__(self):
        return "Pos" + str(self.p) + " Rot: " + str(self.Rot.as_euler('ZYX'))

    def inv(self):
        inv_R = self.Rot.inv()
        inv_p = -inv_R.apply(self.p)
        return tf_class(inv_p,inv_R) 

    def posyaw(self):
        return np.hstack((self.p,self.Rot.as_euler('ZYX')[0]))

    def publishTF(self,name):
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = name
        t.transform.translation.x = self.p[0]
        t.transform.translation.y = self.p[1]
        t.transform.translation.z = self.p[2]
        quat= self.Rot.as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        br.sendTransform(t)

def anglewrap(x):
    x = (x + np.pi) % (2*np.pi)
    if x < 0:
        x += 2*np.pi
    return x - np.pi

def coeffMatrix(start_py, goal_py, tf):
    error_py = goal_py - start_py
    error_py[3] = anglewrap(error_py[3])
    deg = 9
    dim = 4
    constraints = np.zeros((deg+1,dim))
    basis = np.zeros((deg+1,deg+1))
    constraints[0,0] = error_py[0]
    constraints[0,1] = error_py[1]
    constraints[0,2] = error_py[2]
    constraints[0,3] = error_py[3]
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

class PolyTraj:
    def __init__(self):
        self.tracker_topic = rospy.get_param("tracker_topic","/ros_tracker")
        self.wp_topic = rospy.get_param("wp_topic","/baseline_wp")
        self.gcop_topic = rospy.get_param("odom_topic","/gcop_odom")
        self.tracker_sub = rospy.Subscriber(self.tracker_topic,PoseArray,self.callback)
        self.odom_sub = rospy.Subscriber(self.gcop_topic,Odometry,self.odom_callback)
        self.wp_pub = rospy.Publisher(self.wp_topic,PoseArray,queue_size=2)
        self.odom = None
        self.start_posyaw = None
        self.start_time = None
        self.stage_flag = True
        self.camera_transform = tf_class(np.array([0.2093,0.0063,-0.0836]),R.from_euler('ZYX',[-1.5794,0,-1.5794]))
        self.tracking_offset_transform = tf_class(np.array([0,0,0]),R.from_euler('ZYX',[0,0,0]))
        self.staging_tf = tf_class(np.array([0.95,-0.05,0.15]),R.from_euler('ZYX',(np.pi,0,0)))
        self.final_tf = tf_class(np.array([0.60,-0.05,0.3]),R.from_euler('ZYX',(np.pi,0,0)))
        t_f = rospy.get_param("point_tf",3)
        self.num_pts = 3
        self.pred_dt = float(t_f)/self.num_pts
        self.relative = rospy.get_param("relative_pose",True)
    
    def odom_callback(self,data):
        pos = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        Rot = R.from_quat([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])
        self.odom = tf_class(pos,Rot)
        if self.start_posyaw is None:
            self.start_posyaw = self.odom.posyaw()
            self.start_time = data.header.stamp

    def callback(self,tracker_data):
        if self.odom is None:
            print("no odom")
            return
        tracking_pos = np.array([tracker_data.poses[0].position.x,tracker_data.poses[0].position.y,tracker_data.poses[0].position.z])
        tracking_Rot = R.from_quat([tracker_data.poses[0].orientation.x,tracker_data.poses[0].orientation.y,tracker_data.poses[0].orientation.z,tracker_data.poses[0].orientation.w])
        track_tf = tf_class(tracking_pos,tracking_Rot)
        track_global = self.odom * self.camera_transform * track_tf * self.tracking_offset_transform
        quad_o_frame = self.tracking_offset_transform.inv() * track_tf.inv() * self.camera_transform.inv()
        #TODO: change this to better reflect state machine/timeout
        if self.stage_flag:
            offset_tf = self.staging_tf
            pos_err = quad_o_frame.p - offset_tf.p
            if (pos_err[0] < 0.05) and (np.linalg.norm(pos_err[1:3]) < 0.05):
                self.stage_flag = False 
                offset_tf = self.final_tf
        else:
            offset_tf = self.final_tf
            pos_err = quad_o_frame.p - offset_tf.p
            if (np.linalg.norm(pos_err) > 1):
                self.stage_flag = True
                offset_tf = self.staging_tf
        goal_tf = track_global*offset_tf
    	max_v = 0.15
    	min_tf = 4.0
        distance = np.linalg.norm(goal_tf.p - self.start_posyaw[0:3])
    	tf = max(distance/max_v,min_tf)
        coeff = coeffMatrix(self.start_posyaw,goal_tf.posyaw(),tf)
        t_duration = (tracker_data.header.stamp - self.start_time)
        t = t_duration.secs + t_duration.nsecs/1e9
        wp_t = [t+(ii+1)*self.pred_dt for ii in range(self.num_pts)]
        wp_t = [min(tf,temp) for temp in wp_t]
        wp = coeffToWP(coeff,self.start_posyaw,wp_t)
        msg = PoseArray()
        tf0 = self.odom
        for state in wp:
            py = state[0]
            p_i = py[0:3]
            R_i = R.from_euler('ZYX',(py[3],0,0))
            tf_i = tf_class(p_i,R_i)
            tf_rel = tf0.inv()*tf_i
            pose_i = Pose()
            pose_i.position.x = tf_rel.p[0]
            pose_i.position.y = tf_rel.p[1]
            pose_i.position.z = tf_rel.p[2]
            quat_i = tf_rel.Rot.as_quat()
            pose_i.orientation.x = quat_i[0]
            pose_i.orientation.y = quat_i[1]
            pose_i.orientation.z = quat_i[2]
            pose_i.orientation.w = quat_i[3]
            msg.poses.append(pose_i)
            if self.relative:
                tf0 = tf_i
        self.wp_pub.publish(msg)
        yaw_only = tf_class(np.zeros(3),R.from_euler('ZYX',[-1.5794,0,0]))
        yaw_only.publishTF("yaw_only")
        roll_only = tf_class(np.zeros(3),R.from_euler('ZYX',[0,0,-1.5794]))
        roll_only.publishTF("roll_only")
        yaw_times_roll = yaw_only*roll_only
        yaw_times_roll.publishTF("yaw * roll")
        direct_yr = tf_class(np.zeros(3),R.from_euler('ZYX',[-1.5794,0,-1.5794]))
        direct_yr.publishTF("direct yr")
        

def main():
    rospy.init_node('poly_wp')
    pt = PolyTraj()
    rospy.spin()
     
if __name__ == "__main__":
    main()
