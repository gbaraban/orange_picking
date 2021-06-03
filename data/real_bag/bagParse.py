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

bad_angles = []

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
  data = ((data[:,:,0].astype(np.float64)/data.max())*255).astype(np.uint8)
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

def getNextPts(pts_odom, pt_num, events_per_sec, ax):
	pt_odom = pts_odom[pt_num]
	p0 = (pt_odom.transform.translation.x, pt_odom.transform.translation.y, pt_odom.transform.translation.z)
	Rot0 = R.from_quat([
                      pt_odom.transform.rotation.x,
                      pt_odom.transform.rotation.y,
                      pt_odom.transform.rotation.z,
                      pt_odom.transform.rotation.w
                     ])#possibly need to switch w to front
	#print(pt_num)
	R0 = Rot0.as_dcm()#.flatten()
	if not addDrop(R0):
		bad_angles.append(R0)
	#	print(R0)
	#	print("Oopsie")
		return None, None, None, None, None
        #print(logR(R0))
	#print(p0)
	#ax = plot_basis(R=np.eye(3), ax_s=2)
	p = np.array(p0)
	p_ = np.array([0, 0, 0])
	#plot_basis(ax, R0, p, alpha=0.5)


	#print(p0)
	#print(R0)


	indices = np.floor(np.add(np.array([1, 2, 3]) * events_per_sec, pt_num)).astype(int)

	#if indices[-1] >= len(pts_odom):
	#	x = 4
	#	dt = np.floor(((len(pts_odom) - 1) - pt_num)/3)
	#	indices = []
	#	while(dt < 1) and x != 0:
	#		dt = np.floor(((len(pts_odom) - 1) - pt_num)/x)
	#		x -= 1

	#	indices.extend([(pt_num + (dt*j)) for j in range(x-1)])
	#	delta = 3 - len(indices)
	#	indices.extend([len(pts_odom)-1 for j in range(delta)])

	for x, i in enumerate(indices):
		if i >= len(pts_odom):
			delta = pt_num if (x == 0) else indices[x-1]
			dt = np.floor(((len(pts_odom) -1) - delta)/(3-x)).astype(int)
			z = 3 - x
			#print(z)
			#print(indices)
			while (dt < 1) and z != 0:
				dt = np.floor(((len(pts_odom) - 1) - delta)/z).astype(int)
				z -= 1
			for j in range(0,z):
				indices[j+x] = delta + ((j+1)*dt)

			delta = 3 - (z+x)
			#print(delta)
			#print(x)
			#print(z)
			#print(indices)
			for j in range(0, delta):
				#print("in loop", j)
				indices[x+z+j] = len(pts_odom) - 1

			break

	point_list = []
	rot_list = []
	rot_list_po = []
	for x, i in enumerate(indices):
		#print(i)
		if (i < pt_num):
			print(pt_num, i)
		p =  (pts_odom[i].transform.translation.x, pts_odom[i].transform.translation.y, pts_odom[i].transform.translation.z)
		#print(p)
		point_list.append(p)
		Roti = R.from_quat([
			pts_odom[i].transform.rotation.x,
			pts_odom[i].transform.rotation.y,
			pts_odom[i].transform.rotation.z,
			pts_odom[i].transform.rotation.w
			])#possibly need to switch w to front
		Ri = Roti.as_dcm()
		R_relative = np.matmul(np.array(R0).T, np.array(Ri))
		ri = R.from_dcm(R_relative)
		r_zyx = ri.as_euler('ZYX')
		r_po_zyx = Roti.as_euler('ZYX')
		rot_list.append([r_zyx[0], r_zyx[1], r_zyx[2]])
		rot_list_po.append([r_po_zyx[0], r_po_zyx[1], r_po_zyx[2]])
		#print(rot_list[x])

	point_list = np.array(point_list)
	#print(point_list)
	rot_list = np.array(rot_list)
	#print(rot_list)

	pos_only_point_list = np.matmul(np.array(R0).T, (point_list - np.array(p0)).T).T

	point_list = np.matmul(np.array(R0).T, (point_list - np.array(p0)).T).T

	#print(point_list)
	point_list = np.concatenate((point_list, rot_list), axis=1)
	#print(point_list)
	rot0 = Rot0.as_euler('ZYX')
	pos_only_p0 = list(p0)
	p0 = list(p0)
	p0.extend([rot0[0], rot0[1], rot0[2]])

	return point_list, pos_only_point_list, indices, p0, pos_only_p0



def main():
	vrpn_topic = "/vrpn_client/matrice/pose"
	img_topic = "/camera/color/image_raw"
	depth_topic = "/camera/aligned_depth_to_color/image_raw"

	parser = argparse.ArgumentParser()
	parser.add_argument('bag_dir', help="bag dir")
	parser.add_argument('--parse_traj', type=int, default=1, help="Pre parse data") 
	args = parser.parse_args()

	bag_list = os.listdir(args.bag_dir)
	#print(bag_list)
	ctr = 0

	plot_folder = args.bag_dir + '/plot/'
	if os.path.isdir(plot_folder):
		shutil.rmtree(plot_folder)
	os.makedirs(plot_folder)


	sv_folder = args.bag_dir + '/real_world_traj_bag/'
	if os.path.isdir(sv_folder):
		shutil.rmtree(sv_folder)
	os.makedirs(sv_folder)

	for bag_name in bag_list:
		if not bag_name.endswith(".bag"):
			continue
		print(bag_name)
		fname = args.bag_dir + "/" + bag_name
		bag = rosbag.Bag(fname)

		save_folder = args.bag_dir + '/real_world_traj_bag/bag' + str(ctr)
		if os.path.isdir(save_folder):
			shutil.rmtree(save_folder)
		os.makedirs(save_folder)

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
				img_np = np_from_image(img)
				depth_np = depthnp_from_image(depth_img)
				#Img.fromarray(img_np).save(save_folder + "/image" + str(img_id) + ".png")
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
		print(ctr, total_time.secs, total_time.nsecs, no_events)
		ctr += 1
		data_dict = {}
		po_data_dict = {}
		bad_angles = None

		if True:
			pt_n = 0
			while True:
				pt_odom = data[pt_n]
				p0 = (pt_odom.transform.translation.x, pt_odom.transform.translation.y, pt_odom.transform.translation.z)

				Rot0 = R.from_quat([
					pt_odom.transform.rotation.x,
					pt_odom.transform.rotation.y,
					pt_odom.transform.rotation.z,
					pt_odom.transform.rotation.w
				])#possibly need to switch w to front

				#print(pt_num)
				R0 = Rot0.as_dcm()#.flatten()
				p0 = np.array(p0)

				if addDrop(R0):
					break

				pt_n += 1

			R0_ = np.eye(3)
			p0_ = np.array([0, 0, 0])
			#ax = plot_basis(R=R0,p=p0,ax_s=3)
			ax = None
			j = 0
			for i in data.keys():
				#ax = plot_basis(R=R0,p=p0,ax_s=3)
				ret, po_ret, indices, pos0, po_pos0 = getNextPts(data, i, (float(no_events)/total_time.secs), ax)
				#exit(0)
				if ret is not None:
					data_dict[j] = ret
					data_dict[str(j) + "_index"] = indices
					data_dict[str(j) + "_pos"] = pos0
					po_data_dict[j] = po_ret
					po_data_dict[str(j) + "_index"] = indices
					po_data_dict[str(j) + "_pos"] = po_pos0
					Img.fromarray(img_data[i]).save(save_folder + "/image" + str(j) + ".png")
				        Img.fromarray(depth_data[i]).save(save_folder + "/depth_image" + str(j) + ".png")
					j += 1

			#plt.show()
			#plt.savefig(plot_folder + "bag" + str(ctr - 1) + ".png")
			print(len(data_dict))
			data_dict["nEvents"] = no_events
			data_dict["time_secs"] = total_time.secs
			data_dict["time_nsecs"] = total_time.nsecs
			po_data_dict["nEvents"] = no_events
			po_data_dict["time_secs"] = total_time.secs
			po_data_dict["time_nsecs"] = total_time.nsecs
			with open(save_folder + '/data.pickle','wb') as f:
				pickle.dump(po_data_dict,f,pickle.HIGHEST_PROTOCOL)
			with open(save_folder + '/orientation_data.pickle','wb') as f:
				pickle.dump(data_dict,f,pickle.HIGHEST_PROTOCOL)


		if args.parse_traj == 0:
			data_dict = {}
			data_list = []
			pt_n = 0
			print(img_id)
			for i in range(img_id):
				pt_odom = data[i]
				p = (pt_odom.transform.translation.x, pt_odom.transform.translation.y, pt_odom.transform.translation.z)

				RotR = R.from_quat([
					pt_odom.transform.rotation.x,
					pt_odom.transform.rotation.y,
					pt_odom.transform.rotation.z,
					pt_odom.transform.rotation.w
				])#possibly need to switch w to front

				Rot = RotR.as_dcm()
				p = np.array(p)

				if not addDrop(Rot):
					continue

				pt = np.zeros((6))
				pt[0:3] = p
				pt[3:6] = RotR.as_euler('ZYX')
				data_list.append(pt)
				#Img.fromarray(img_data[i]).save(save_folder + "/image" + str(pt_n) + ".png")
				pt_n += 1

			print(len(data_list), pt_n)
			data_dict["nEvents"] = pt_n
			data_dict["time_secs"] = total_time.secs
			data_dict["time_nsecs"] = total_time.nsecs
			data_dict["data"] = data_list
			with open(save_folder + '/data.pickle_no_parse','wb') as f:
				pickle.dump(data_dict,f,pickle.HIGHEST_PROTOCOL)


	if args.parse_traj == 1 and not bad_angles is None:
		fopen = open("points.pickle", "wb")
		pickle.dump(bad_angles, fopen)
		R0 = np.eye(3)
		p0 = np.array([0, 0, 0])
		#ax = plot_basis(R=R0,p=p0,ax_s=3)
		#for x in bad_angles:
		#plot_basis(ax, x, p0, alpha=0.5)
		#plt.show()

if __name__ == "__main__":
	main()
