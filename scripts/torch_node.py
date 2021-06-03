#!/usr/bin/env python3
import roslib
import rospy
import numpy as np
import tf
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import Image as ImageMessage
from orangesimulation import run_model

class nodeClass:
    def __init__(self,out_topic,model,mean_name):
        self.pub = rospy.Publisher(out_topic,PoseArray)
        self.model = model
        self.model.eval()
        self.image_list = []
        self.mean = np.load(mean_name)
        self.dev = None #TODO: possibly change this

    def callback(self,data):
        #Unpack data into torch tensor
        shape = (msg.height, msg.width, 3)#Hardcoding 3 channels and uint8
        dtype = np.dtype('uint8').newbyteorder('>' if data.is_bigendian else '<')
        img_data = np.fromstring(data.data, dtype=dtype).reshape(shape)
        image_arr = img_data #TODO: double check this code snippet
        self.image_list.insert(0,image_tensor)
        if len(self.image_list) is self.model.num_images:
            #Concatentate images
            input_arr = ...
            goal_tensor = run_model(self.model,input_arr,mean_image=self.mean,device=self.dev)
            out_msg = PoseArray()
            for ii in range(self.model.num_points):
                goal_point = goal_tensor[ii]
                temp_pose = Pose()
                temp_pose.position.x = goal_point[0]
                temp_pose.position.y = goal_point[1]
                temp_pose.position.z = goal_point[2]
                quat = R.from_euler('ZYX',(goal_point[3],goal_point[2],goal_point[1])).as_quat()
                temp_pose.orientation.x = quat[0]
                temp_pose.orientation.y = quat[1]
                temp_pose.orientation.z = quat[2]
                temp_pose.orientation.w = quat[3]
                out_msg.poses.append(temp_pose)
            self.image_list = self.image_list[0:self.model.num_images-1]
            self.pub.publish(out_msg)

def main():
    image_topic = rospy.get_param('image_topic')
    output_topic = rospy.get_param('goal_topic')
    rospy.init_node('torch_node')
    model = OrangeNet...#TODO: load in NN Model here
    node = nodeClass(output_topic,model,rospy.get_param('mean_image'))
    sub = rospy.Listener(image_topic,ImageMessage,node.callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
