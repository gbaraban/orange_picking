#!/usr/bin/env python3
import roslib
import rospy
import numpy as np
import tf
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseArray, Point, Pose, PoseStamped, Quaternion, Twist, Vector3
#from orange_ros.msg import GoalPoints
from orangesimulation import sys_f_gcop

def sendTF(pos,rot,name,base):
    br = tf.TransformBroadcaster()
    br.sendTransform(pos,rot,rospy.Time.now(),name,base)


def callback(data,pub):
    #Unpack data
    #x0 = data.x0.pose.pose
    goals = data.poses
    gcopgoals = []
    x_tf_name = rospy.get_param('quad_name','matric')
    for ii, goal in enumerate(goals):
        pos = goal.position
        pos = (pos.x,pos.y,pos.z)
        rot = pose.orientation
        rot = (rot.x,rot.y,rot.z,rot.w)
        sendTF(pos,rot,rospy.get_param('goal_name','Goal') + " "+str(ii),x_tf_name)
        euler = R.from_quat(rot).as_euler('zyx')
        gcopgoals.append(np.hstack((pos,euler)))
    #Call gcop
    (x_pos, x_rot) = tf.TransformListener().lookupTransform(x_tf_name,'world',rospy.Time(0))
    x_rot = R.from_quat(x_rot).as_euler('zyx')
    x0 = np.hstack((x_pos,x_rot))
    ref_traj = sys_f_gcop(x0,gcopgoals,3)
    x_traj = ref_traj[0]
    #Send tfs
    #sendGCOPtoTF(x0,"Quad","woorld")
    path_msg = Path()
    path_msg.header = data.header
    for ii, x in enumerate(x_traj):
        #Convert gcop to Pose
        pose_stmp = PoseStamped()
        pose_stmp.header = data.header
        pose_stmp.pose.position.x = x[0][0]
        pose_stmp.pose.position.y = x[0][1]
        pose_stmp.pose.position.z = x[0][2]
        euler = R.from_dcm(x[1]).as_euler('zyx')
        quat = tf.transformations.quat_from_euler(euler[2],euler[1],euler[0])
        pose_stmp.pose.orientation.x = quat[0]
        pose_stmp.pose.orientation.y = quat[1]
        pose_stmp.pose.orientation.z = quat[2]
        pose_stmp.pose.orientation.w = quat[3]
        path_msg.poses.append(pose_stmp)
        #sendGCOPtoTF(x,"x "+str(ii),"world")#TODO: possbily change to other broadcasting type
    pub.publish(path_msg)

def main():
    topic_name = rospy.get_param('goal_topic','goal')
    path_topic = rospy.get_param('path_topic','path')
    rospy.init_node('trajectory_node')
    pub = rospy.Publisher(path_topic,Path)
    cb = lambda d : callback(d,pub)
    sub = rospy.Listener(topic_name,PoseArray,cb)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
