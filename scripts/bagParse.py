import rosbag
import cv2
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
import gcophrotor
import numpy as np
import sys
import os
import pickle
from scipy.spatial.transform import Rotation as R
import argparse

def gcop_solve(odom,tree_pos,orange_pos):
  N = 100
  tf = 15
  epochs = 50
  stiffness = 1000
  stiff_mult = 2.0
  q = (0.2,0.2,0,#rotation log
       0,0,0,#position
       1,1,1,#rotation rate
       5,5,5)#velocity
  qf = (10,10,10,#rotation log
       50,50,50,#position
       0.5,0.5,0.5,#rotation rate
       5,5,5)#velocity
  r = (.1,.1,.1,1)
  yaw_g = 10
  #tree_R = #0.6
  #treeHeight = #1.6
  rp_g = 0
  direction_gain = 0 
  p0 = (odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z)
  Rot0 = R.from_quat([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])#possibly need to switch w to front
  R0 = Rot0.as_matrix().flatten()
  ref_traj = gcophrotor.trajgen_R(N,tf,epochs,tuple(p0),tuple(R0),tuple(orange_pos),yawf,
                                  tuple(tree_pos),tree_R,treeHeight,
                                  tuple(q),tuple(qf),tuple(r),yaw_g,rp_g,direction_gain,
                                  stiffness,stiff_mult)
  pts_per_sec = float(N)/tf
  indices = np.floor(np.array([1, 2, 3])*pts_per_sec).astype(int)
  point_list = [ref_traj[temp][0] for temp in indices]
  return point_list

def main():
  vrpn_topic = '/vrpn_client/matrice/pose'
  image_topic = '/camera/color/image_raw'
  parser = argparse.ArgumentParser()
  parser.add_argument('bag_dir', help='bag dir')
  args = parser.parse_args()
  f = open(args.bag_dir + "/bag_data.txt")
  location={}
  for line in f:
    data = line.strip(" ").strip("\n").split(",")
    location[data[0]] = [float(data[1]), float(data[2]), float(data[3])]
  #Get list of bags
  bag_list = os.listdir(args.bag_dir)
  print(bag_list)
  ctr = 0
  bridge = CvBridge()
  for bag_name in bag_list:
    print(bag_name)
    if not bag_name.endswith(".bag"):
      continue
    filename = args.bag_dir + '/' + bag_name
    bag = rosbag.Bag(filename)
    save_folder = arg.bag_dir + '/bag' + str(ctr)
    os.makedirs(save_folder)
    data_dict = dict()
    tree_pos = location["tree"]
    orange_pos = location[bag_name]
    image = None
    odom = None
    image_t = -1
    odom_t = -1
    image_idx = 0
    for topic, msg, t in bag.read_messages(topics=[vrpn_topic,image_topic]):
      if topic == vrpn_topic:
        odom = msg
        odom_t = t
      if topic == image_topic:
        image = msg
        image_t = t
      if (image is not None) and (odom is not None):
        #Save Image
        cv_img = bridge.imgmsg_to_cv2(image, desired_encoding+"passthrough")
        cv2.imwrite(save_folder,"image" + str(image_idx) + ".png",cv_img)
        #Calculate GCOP
        data_dict[image_idx] = gcop_solve(odom,tree_pos,orange_pos)
        #Clean Up
        image_idx += 1
        odom = image = None
    with open(save_folder + '/data.pickle','wb') as f:
      pickle.dump(data_dict,f,pickle.HIGHEST_PROTOCOL)
      ctr += 1

if __name__ == '__main__':
    main()
