from numpy import linalg as LA
import rosbag
import PIL.Image as Img
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse
from scipy.linalg import logm
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *
import shutil

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


def main():
        vrpn_topic = "/vrpn_client/matrice/pose"
        img_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"

        parser = argparse.ArgumentParser()
        parser.add_argument('bag_dir', help="bag dir")

        args = parser.parse_args()

        avg_total_time = 0.0

        bag_list = os.listdir(args.bag_dir)
        #print(bag_list)
        ctr = 0

        for bag_name in bag_list:
                if not bag_name.endswith(".bag"):
                        continue
                #print(bag_name)
                fname = args.bag_dir + "/" + bag_name
                bag = rosbag.Bag(fname)

                data = {}
                img_data = {}
                depth_data = {}
                img = None
                depth_img = None
                odom = None

                img_t = -1
                depth_t = -1
                odom_t = -1
                img_id = 0

                time = []

                for topic, msg, t in bag.read_messages(topics=[vrpn_topic, img_topic, depth_topic]):
                        #print(t)
                        if topic == vrpn_topic and (img is not None):
                                odom = msg
                                odom_t = t

                        if topic == img_topic:
                                img = msg
                                img_t = t

                        if topic == depth_topic:
                                depth_img = msg
                                depth_t = t

                        if (img is not None) and (odom is not None) and (depth_img is not None):
                                img_np = None #np_from_image(img)
                                depth_np = None #depthnp_from_image(depth_img)
                                #Img.fromarray(img_np).save(save_folder + "/image" + str(img_id) + "$
                                Rot0 = R.from_quat([
                                        odom.transform.rotation.x,
                                        odom.transform.rotation.y,
                                        odom.transform.rotation.z,
                                        odom.transform.rotation.w
                                        ])#possibly need to switch w to front
                                R0 = Rot0.as_dcm()#.flatten()

                                if not addDrop(R0):
                                        odom = None
                                        img = None
                                        img_np = None
                                        continue

                                data[img_id] = odom
                                img_data[img_id] = img_np
                                depth_data[img_id] = depth_np

                                img_id += 1
                                odom = None
                                img = None
                                img_np = None
                                time.append(t)


                total_time = time[-1] - time[0]
                no_events = len(time)
                #print(ctr, total_time.secs, total_time.nsecs, no_events)
                ctr += 1
                avg_total_time += total_time.secs


        print(avg_total_time/float(ctr))


if __name__ == "__main__":
        main()



