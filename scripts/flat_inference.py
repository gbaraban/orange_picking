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

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *

import math

enable_multi_control = True 
enable_seg = True
enable_depth = False
enable_states = True
enable_multi_image = False
num_images = 1
states_version = 4
state_count = 2


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

name_to_dtypes = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),
	
    # for bayer image (based on cv_bridge.cpp)
	"bayer_rggb8":	(np.uint8,  1),
	"bayer_bggr8":	(np.uint8,  1),
	"bayer_gbrg8":	(np.uint8,  1),
	"bayer_grbg8":	(np.uint8,  1),
	"bayer_rggb16":	(np.uint16, 1),
	"bayer_bggr16":	(np.uint16, 1),
	"bayer_gbrg16":	(np.uint16, 1),
	"bayer_grbg16":	(np.uint16, 1),

    # OpenCV CvMat types
	"8UC1":    (np.uint8,   1),
	"8UC2":    (np.uint8,   2),
	"8UC3":    (np.uint8,   3),
	"8UC4":    (np.uint8,   4),
	"8SC1":    (np.int8,    1),
	"8SC2":    (np.int8,    2),
	"8SC3":    (np.int8,    3),
	"8SC4":    (np.int8,    4),
	"16UC1":   (np.uint16,   1),
	"16UC2":   (np.uint16,   2),
	"16UC3":   (np.uint16,   3),
	"16UC4":   (np.uint16,   4),
	"16SC1":   (np.int16,  1),
	"16SC2":   (np.int16,  2),
	"16SC3":   (np.int16,  3),
	"16SC4":   (np.int16,  4),
	"32SC1":   (np.int32,   1),
	"32SC2":   (np.int32,   2),
	"32SC3":   (np.int32,   3),
	"32SC4":   (np.int32,   4),
	"32FC1":   (np.float32, 1),
	"32FC2":   (np.float32, 2),
	"32FC3":   (np.float32, 3),
	"32FC4":   (np.float32, 4),
	"64FC1":   (np.float64, 1),
	"64FC2":   (np.float64, 2),
	"64FC3":   (np.float64, 3),
	"64FC4":   (np.float64, 4)
}

class CameraIntrinsics:
    def __init__(self):
        self.ppx = 333.42010498046875
        self.ppy = 250.38986206054688
        self.fx = 607.265625
        self.fy = 607.2756958007812

class Rotations:
    def __init__(self):
        pass

    def rotx(self, angle):
        r = np.array([[1, 0, 0],
                    [0, np.cos(angle), np.sin(angle)],
                    [0, -np.sin(angle), np.cos(angle)]]).astype(np.float64)

        return r

    def roty(self, angle):
        r = np.array([[np.cos(angle), 0, -np.sin(angle)],
                    [0, 1, 0],
                    [np.sin(angle), 0, np.cos(angle)]]).astype(np.float64)

        return r

    def rotz(self, angle):
        r = np.array([[np.cos(angle), np.sin(angle), 0],
                    [-np.sin(angle), np.cos(angle), 0 ],
                    [0, 0, 1]]).astype(np.float64)

        return r


class FlatNode:
    def __init__(self, image_topic='/d400/color/image_raw', depth_topic='/d400/aligned_depth_to_color/image_raw', gcop_topic="/gcop_odom"):
        global enable_multi_control, enable_depth, enable_states, enable_seg, states_version, enable_multi_image, num_images, state_count
        self.enable_multi_control = enable_multi_control
        self.enable_depth = enable_depth
        self.enable_states = enable_states
        self.enable_seg = enable_seg
        self.states_version = states_version
        self.enable_multi_image = enable_multi_image
        self.num_images = num_images
        self.state_count = state_count
        self.yaw_only = False
        
        print(image_topic, depth_topic)
        self.image_sub = message_filters.Subscriber(image_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.odom_node = message_filters.Subscriber(gcop_topic, Odometry)
        
        self.__params()
        
        self.__bridge = CvBridge()

        self.__segmodel = SegmentationNet()

        self.__model = OrangeNet8(self.__capacity,self.__num_images,self.__num_pts,self.__bins,mins = self.mins,maxs = self.maxs,n_outputs=self.__outputs,num_channels=self.__num_channels, state_count=self.state_count, num_controls=self.__num_controls)

        self.__loadModel()
        
        torch.no_grad()

        self.__loadMeanImages()

        self.__pub = rospy.Publisher("/goal_points",PoseArray,queue_size=50)
        self.__pub_seg = rospy.Publisher('/orange_picking/seg_image', Image,queue_size=50)

        self.__pointcloud_publisher = rospy.Publisher("/pointcloud_topic", PointCloud, queue_size=2)
        self.__pointcloud_publisher_extra = rospy.Publisher("/pointcloud_topic_extra", PointCloud, queue_size=2)
        
        self.__goalpoint_publisher = rospy.Publisher("/ros_tracker", PoseArray, queue_size=100)
        self.__normal_vec_publisher = rospy.Publisher("/normal_tracker", PoseArray, queue_size=100)

        self.__seg_image_publisher = rospy.Publisher("/seg_image", Image, queue_size=5)
        
        rot = Rotations()
        self.world2orange = None #np.eye(3)#np.matmul(rot.rotz(0), np.matmul(rot.rotx(np.pi/2),np.matmul(rot.rotz(np.pi/2), rot.roty(np.pi/2))))
        #self.world2orange = np.matmul(rot.rotz(np.pi/2), rot.roty(np.pi/2))

        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()
        
        self.world2orange_pos = None
      
        self.alpha = 0.5
        self.min_alpha = 0.05
        self.max_steps = 50

        self.stamp_now = None

        print("Setup complete")

    def __loadModel(self):
        if os.path.isfile(self.__segload):
            if not self.__gpu is None:
                    checkpoint = torch.load(self.__segload,map_location=torch.device('cuda'))
            else:
                    checkpoint = torch.load(self.__segload)
            self.__segmodel.load_state_dict(checkpoint)
            self.__segmodel.eval()
            print("Loaded Model: ", self.__segload)
        else:
                print("No checkpoint found at: ", self.__segload)
                exit(0)

        if os.path.isfile(self.__modelload):
            if not self.__gpu is None:
                    checkpoint = torch.load(self.__modelload,map_location=torch.device('cuda'))
            else:
                    checkpoint = torch.load(self.__modelload)
            self.__model.load_state_dict(checkpoint)
            self.__model.eval()
            print("Loaded Model: ", self.__modelload)
        else:
                print("No checkpoint found at: ", self.__modelload)
                exit(0)


        if not self.__gpu is None:
            self.__device = torch.device('cuda:'+str(self.__gpu))
        else:
            self.__device = torch.device('cpu')
        
        self.__segmodel = self.__segmodel.to(self.__device)
        self.__model = self.__model.to(self.__device)

        self.__segmodel.eval()

        self.__model.min = self.mins
        self.__model.max = self.maxs
    
    def __loadMeanImages(self):
        if not (os.path.exists(self.__mean_image_loc)):
            print('mean image file not found', self.__mean_image_loc)
            exit(0)
        else:
            print('mean image file found')
            self.__mean_image = np.load(self.__mean_image_loc)

        self.__mean_depth_image = np.load(self.__mean_depth_loc)/10000.

        self.__seg_mean_image = torch.tensor(np.load(self.__seg_mean)).to('cuda')

        
    def __params(self, gpu=True, relative=True):
        self.__num_images = 1
#        self.__segload = "/home/matricex/matrice_ws/src/orange_picking/model/model_seg145.pth.tar"
        self.__segload = "model/segmentation/model_seg145.pth.tar"
#        self.__modelload = "/home/matricex/matrice_ws/src/orange_picking/model/best_fine_dt_025_jun22/model15.pth.tar"
        self.__modelload = "/mnt/samsung/gabe/save_models/variable_log/2021-08-09_22-38-15/model9.pth.tar"
                
        self.__mean_image_loc = "data/mean_imgv2_data_data_collection4_real_world_traj_bag.npy"
        self.__seg_mean = "data/depth_data/data/mean_seg.npy"
        self.__mean_depth_loc = "data/mean_depth_imgv2.npy"
        if torch.cuda.is_available() and gpu:
            self.__gpu = 0
        else:
            self.__gpu = None

        self.__h = 480
        self.__w = 640

        self.__capacity = 1.0
        self.__num_images = self.num_images
        self.__num_channels = 3
        if self.enable_seg:
            self.__num_channels += 1

        if self.enable_depth:
            self.__num_channels += 1

        self.__num_pts = 1
        self.__bins = 100
        if self.yaw_only:
            self.__outputs = 4
        else:
            self.__outputs = 6
        
        
        self.__spherical = False
        self.__regression = False

        self.__num_controls = 1
        if self.enable_multi_control:
            self.__num_controls = 3
        

        self.mins = [(-0.2,-0.4,-0.2,-np.pi,-np.pi,-np.pi)] 
        self.maxs = [(2,0.4,0.2,np.pi,np.pi,np.pi)] 
        self.extra_mins = [[(-0.05,-0.05,-0.05,-0.1,-0.01,-0.02)],[(-1.5,-0.15,-0.4,-0.15,-0.2,-0.02)]]
        self.extra_maxs = [[(0.4,0.05,0.15,0.1,0.01,0.02)],[(0.02,0.15,0.05,0.15,0.2,0.02)]]

    def __rosmsg2np(self,data):
        try:
            image_arr = self.__bridge.imgmsg_to_cv2(data,"rgb8")
        except CvBridgeError as e:
            print(e)
        return image_arr

    def __meanSubtract(self, cv_image):
        cv_image = cv2.resize(cv_image,(self.__w,self.__h))
        cv2.imwrite("test_pre.png", cv_image)
        cv_image = cv_image/255.0
        #print(cv_image.shape)
        if self.__mean_image is None:
            mean_subtracted = (cv_image)
            print("ISSUEE: NO MEAN!!! ")
        else:
            mean_subtracted = (cv_image-self.__mean_image)
        return mean_subtracted

    def __process4model(self, image_arr):
        image_arr = self.__meanSubtract(image_arr)
        image_arr = image_arr.transpose(2,0,1)
        image_arr = image_arr.reshape((1,3,self.__h,self.__w))

        image_tensor = torch.tensor(image_arr)
        image_tensor = image_tensor.to(self.__device,dtype=torch.float)

        return image_tensor

    def __segmentationInference(self, image_tensor):
        seglogits = self.__segmodel(image_tensor)
        seglogits = seglogits.view(-1,2,self.__segmodel.h,self.__segmodel.w)
        segimages = (torch.max(seglogits, 1).indices).to('cpu') #.type(torch.FloatTensor).to(device)
        seg_np = np.array(segimages[0,:,:])
        
        pub_np = (255 * seg_np.copy()).astype(np.uint8).reshape((self.__h, self.__w))
        image_message = self.__bridge.cv2_to_imgmsg(pub_np, encoding="passthrough")
        self.__seg_image_publisher.publish(image_message)
        cv2.imwrite('test.png', pub_np)

        return seg_np, segimages

    def __process4InfModel(self,image_tensor, segimages, depth_image=None):
        if depth_image is not None:
            depth_tensor = torch.tensor(depth_image)
            depth_tensor = torch.reshape(depth_tensor, (1, 1, depth_tensor.shape[0], depth_tensor.shape[1]))
            depth_tensor = depth_tensor.type(torch.FloatTensor).to(self.__device)

        segimages = segimages.type(torch.FloatTensor).to(self.__device)
        segimages -= self.__seg_mean_image
        segimages = torch.reshape(segimages, (segimages.shape[0], 1, segimages.shape[1], segimages.shape[2]))

        if depth_image is not None:
            seg_tensor_image = torch.cat((image_tensor, depth_tensor, segimages), 1)
        else:
            seg_tensor_image = torch.cat((image_tensor, segimages), 1)

        return seg_tensor_image


    def __predictionInference(self, seg_tensor_image, states=None):
        if states is None:
            logits = self.__model(seg_tensor_image)
        else:
            logits = self.__model(seg_tensor_image, states)
        logits = logits.cpu()
        logits = logits.view(1,self.__model.outputs,self.__model.num_points,self.__model.bins).detach().numpy()
        predict = np.argmax(logits,axis=3)

        return predict

    def __predictionInferenceMultiControl(self, seg_tensor_image, states=None):
        if states is None:
            logits = self.__model(seg_tensor_image)
        else:
            logits = self.__model(seg_tensor_image, states)
        logits = logits.cpu()
        classifier_logits = logits[:, :self.__num_controls]
        logits = logits[:, self.__num_controls:]
        softmax = nn.Softmax(dim=1)
        predicted_phases = softmax(classifier_logits).to('cpu').detach().numpy()
        max_pred = np.argmax(predicted_phases, axis=1)[0]

        logits = logits.view(1,self.__model.outputs,self.__model.num_points,self.__model.bins*self.__num_controls).detach().numpy()
        logits = logits[:, :, :, np.arange(self.__bins*max_pred, self.__bins*(max_pred+1))]

        predict = np.argmax(logits,axis=3)

        return predict, max_pred

    def __depthnpFromImage(self, msg):
        #Stolen from https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py
        if not msg.encoding in name_to_dtypes:
            raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
        
        dtype_class, channels = name_to_dtypes[msg.encoding]
        dtype = np.dtype(dtype_class)
        dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
        shape = (msg.height, msg.width, channels)

        data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
        data.strides = (
            msg.step,
            dtype.itemsize * channels,
            dtype.itemsize
        )

        if channels == 1:
            data = data[...,0]
        return data.astype(np.float64)
        # dtype_class = np.uint16
        # channels = 1
        # dtype = np.dtype(dtype_class)
        # dtype = dtype.newbyteorder('>' if image.is_bigendian else '<')
        # shape = (image.height, image.width, channels)

        # data = np.fromstring(image.data, dtype=dtype).reshape(shape)
        # data.strides = (
        #     image.step,
        #     dtype.itemsize * channels,
        #     dtype.itemsize
        # )
        # data = data[:,:,0].astype(np.float64)
        # cm_from_pixel = 0.095
        # return data #*cm_from_pixel
    
    def _getBodyV(self, odom, rot):
        rotMat = R.from_quat(rot).as_dcm()
        body_v = []
        body_v.append(odom.twist.twist.linear.x)
        body_v.append(odom.twist.twist.linear.y)
        body_v.append(odom.twist.twist.linear.z)
        body_v = list(np.matmul(rotMat.T, np.array(body_v)))
        body_v.append(odom.twist.twist.angular.x)
        body_v.append(odom.twist.twist.angular.y)
        body_v.append(odom.twist.twist.angular.z)
        return np.array(body_v)

    def __getCentroid(self, seg_image):
        centroid = ndimage.measurements.center_of_mass(seg_np)
        if np.isnan(centroid[0]) or np.isnan(centroid[1]):
            return None
        #if (c[0] < 100) or (c[1] < 100):
        #    PIL.Image.fromarray((seg_np*255).astype('uint8')).show()
        return centroid

    def __convert_depth_frame_to_pointcloud(self, depth_image, camera_intrinsics, area=None):
        #stolen from https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/box_dimensioner_multicam/helper_functions.py#L151
        """
        Convert the depthmap to a 3D point cloud
        Parameters:
        -----------
        depth_frame 	 	 : rs.frame()
                            The depth_frame containing the depth map
        camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
        Return:
        ----------
        x : array
            The x values of the pointcloud in meters
        y : array
            The y values of the pointcloud in meters
        z : array
            The z values of the pointcloud in meters
        """
        if area is None:
            [height, width] = depth_image.shape

            nx = np.linspace(0, width-1, width)
            ny = np.linspace(0, height-1, height)
            u, v = np.meshgrid(nx, ny)
        else:
            u = area[:, 1]
            v = area[:, 0]

        x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
        y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy
        if area is None:
            z = depth_image.flatten() / 1000
        else:
            z = depth_image[area[:,0], area[:,1]].flatten() / 1000
        x = np.multiply(x,z)
        y = np.multiply(y,z)

        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]

        return x, y, z

    def __publishPointCloud(self, x, y, z, publ, step=100):
        pointcloud = PointCloud()
        header = std_msgs.msg.Header()
        header.stamp = self.stamp_now #rospy.Time.now()
        header.frame_id = 'camera_depth_optical_frame'
        #header.frame_id = 'camera_link'
        
        pointcloud.header = header

        for i in range(0, len(x), step):
            pointcloud.points.append(Point32(x[i], y[i], z[i]))

        publ.publish(pointcloud)


    def __rotate_frame(self, x, y, z):
        x = x.reshape((x.shape[0], 1))
        y = y.reshape((y.shape[0], 1))
        z = z.reshape((z.shape[0], 1))
        points = np.concatenate((x, y, z), axis=1)
        #print(x.shape, points.shape)
        r = Rotations()
        points = (np.matmul(r.rotz(np.pi/2), np.matmul(r.rotx(np.pi/2), points.T))).T
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        return x, y, z

    def __orange_orientation(self, trans, rot, mean_pos, orientation=None):

        if self.alpha != self.min_alpha:
            delta = (self.alpha - self.min_alpha)/self.max_steps   
            self.alpha -= delta

        print(self.alpha)

        #decaying alpha in future
        rot = R.from_quat(np.array(rot))
        #fixed_transform = R.from_quat(np.array([-0.500, 0.500, 0.500, 0.500])).as_dcm()
        temp_world2orange_pos = np.matmul(rot.as_dcm(), mean_pos) + np.array(trans)
            
        if self.world2orange_pos is None:
            self.world2orange_pos = temp_world2orange_pos.copy()
        else:
            self.world2orange_pos = self.world2orange_pos + self.alpha*(temp_world2orange_pos-self.world2orange_pos)

        if orientation is not None:
            rot_obj = Rotations()
            orientation = R.from_dcm(np.matmul(np.matmul(rot.as_dcm(),R.from_quat(orientation).as_dcm()), rot_obj.rotx(np.pi/2))).as_quat()
            if self.world2orange is None:
                self.world2orange = R.from_quat(orientation).as_dcm()
            else:
                new_rot = orientation
                old_rot = R.from_dcm(self.world2orange).as_quat()
                key_times = [0, 1]
                key_rots = np.array([old_rot, new_rot])
                key_rots = R.from_quat(key_rots)
                slerp = Slerp(key_times, key_rots)

                times = [self.alpha]
                mean_rot = slerp(times)
                mean_rot = mean_rot[0]
                self.world2orange = mean_rot.as_dcm()        

        self.br.sendTransform((self.world2orange_pos[0], self.world2orange_pos[1], self.world2orange_pos[2]),
                     R.from_dcm(self.world2orange).as_quat(),
                     self.stamp_now, #rospy.Time.now(),
                     "orange",
                     "world")

        Rot = R.from_dcm(np.matmul(np.linalg.inv(rot.as_dcm()), self.world2orange)).as_quat()
        Trans = np.matmul(np.linalg.inv(rot.as_dcm()), (self.world2orange_pos - np.array(trans)))
        return Rot, Trans

    def _publishWaypointsMultiControl(self, predict, max_pred):
        goal = []
        msg = PoseArray()
        header = std_msgs.msg.Header()
        if max_pred == 0:
            bin_min = self.mins
            bin_max = self.maxs
            num_bins = self.__bins
            time_secs = self.waypoint_secs[max_pred]
        else:
            bin_min = self.__extra_mins[max_pred-1]
            bin_max = self.__extra_maxs[max_pred-1]
            num_bins = self.__bins
            time_secs = self.waypoint_secs[max_pred]

        for pt in range(self.__model.num_points):
            point = []
            for coord in range(self.__model.outputs):
                if not self.__regression:
                    bin_size = (bin_max[pt][coord] - bin_min[pt][coord])/float(num_bins)
                    point.append(bin_min[pt][coord] + bin_size*predict[0,coord,pt])
                else:
                    point.append(logits[0,pt, coord])
            if self.__spherical:
                pointList = [np.array(point)]
                pl = sphericalToXYZ().__call__(pointList)
                print("Before: ", pl.shape)
                point = pl.flatten()
                print("After: ", point.shape)

            point = np.array(point)
            #print(point)
            goal.append(point)
            pt_pose = Pose()
            pt_pose.position.x = point[0]
            pt_pose.position.y = point[1]
            pt_pose.position.z = point[2]
            R_quat = R.from_euler('ZYX', point[3:6]).as_quat()
            pt_pose.orientation.x = R_quat[0]
            pt_pose.orientation.y = R_quat[1]
            pt_pose.orientation.z = R_quat[2]
            pt_pose.orientation.w = R_quat[3]
            msg.poses.append(pt_pose)
        #exit()
        header.stamp = rospy.Duration(time_secs) 
        msg.header = header
        self.__pub.publish(msg)

    def _publishWaypoints(self, predict):
        goal = []
        msg = PoseArray()
        header = std_msgs.msg.Header()
        for pt in range(self.__model.num_points):
            point = []
            for coord in range(self.__model.outputs):
                if not self.__regression:
                    bin_size = (self.__model.max[pt][coord] - self.__model.min[pt][coord])/float(self.__model.bins)
                    point.append(self.__model.min[pt][coord] + bin_size*predict[0,coord,pt])
                else:
                    point.append(logits[0,pt, coord])
            if self.__spherical:
                pointList = [np.array(point)]
                pl = sphericalToXYZ().__call__(pointList)
                print("Before: ", pl.shape)
                point = pl.flatten()
                print("After: ", point.shape)

            point = np.array(point)
            #print(point)
            goal.append(point)
            pt_pose = Pose()
            pt_pose.position.x = point[0]
            pt_pose.position.y = point[1]
            pt_pose.position.z = point[2]
            R_quat = R.from_euler('ZYX', point[3:6]).as_quat()
            pt_pose.orientation.x = R_quat[0]
            pt_pose.orientation.y = R_quat[1]
            pt_pose.orientation.z = R_quat[2]
            pt_pose.orientation.w = R_quat[3]
            msg.poses.append(pt_pose)
        #exit()
        header.stamp = rospy.Duration(self.waypoint_secs[0]) 
        msg.header = header
        self.__pub.publish(msg)

    def _publishGoalPoints(self, x, y, z, trans=None, rot=None, orientation=None, odom_data=None, num_points=1):
        
        if trans is None or rot is None:
            try:
                # (trans,rot) = self.listener.lookupTransform('world', 'camera_depth_optical_frame_filtered', rospy.Time(0))
                (trans,rot) = self.listener.lookupTransform('world', 'camera_depth_optical_frame', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return

        if np.any(np.isnan(trans)) or np.any(np.isnan(rot)) or np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(z)):
            print("NANANNANANANANANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANANANANANANANANNANNANANA")
            return
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        mean_z = np.mean(z)
        # if odom_data is not None:
        #     rot = np.array([odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])
        orientation, mean_pos = self.__orange_orientation(trans, rot, np.array([mean_x, mean_y, mean_z]), orientation)

        mean_x, mean_y, mean_z = mean_pos[0], mean_pos[1], mean_pos[2] 

        header = std_msgs.msg.Header()
        header.stamp = self.stamp_now #rospy.Time.now()
        header.frame_id = 'camera_depth_optical_frame_filtered'
        header.frame_id = 'camera_depth_optical_frame'
        #header.frame_id = 'camera_link'
        
        goalarray_msg = PoseArray()
        goalarray_msg.header = header
        
        for i in range(num_points): 
            goal = Pose()
            goal.position.x = mean_x
            goal.position.y = mean_y
            goal.position.z = mean_z

            goal.orientation.x = orientation[0]
            goal.orientation.y = orientation[1]
            goal.orientation.z = orientation[2]
            goal.orientation.w = orientation[3]

            goalarray_msg.poses.append(goal)
        # print("Publishing")
        self.__goalpoint_publisher.publish(goalarray_msg)
    

    def __skew(self, vec):
        return np.array([[0, -vec[2], vec[1]],
                        [vec[2], 0, -vec[0]],
                        [-vec[1], vec[0], 0]])

    def __find_rot(self, vector):
        vector = np.array(vector)
        vector = vector/np.linalg.norm(vector)

        vector2 = np.array([1, 0, 0])

        v = np.cross(vector, vector2)
        s = np.linalg.norm(v)
        c = np.dot(vector, vector2)

        rot = np.eye(3) + self.__skew(v) + (np.matmul(self.__skew(v), self.__skew(v))*(1-c)/(s*s)) 
        return R.from_dcm(rot).as_quat() 
        
    def debugPlanePlotter(self, pts, plane, orientation, position):
        x = np.linspace(-2.,2.,20)
        y = np.linspace(-2.,2.,20)

        X,Y = np.meshgrid(x,y)
        Z=(plane[0]*X + plane[1]*Y + plane[3])/-plane[2]

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        #surf = ax.plot_surface(X, Y, Z)
        ax.scatter3D(X, Y, Z, c='b')

        ax.scatter3D(pts[:,0], pts[:,1], pts[:,2], c='r')
        # ax.set_aspect('equal')
        ax.set_zlabel("x")
        ax.set_xlabel("-y")
        ax.set_ylabel("-z")
        
        pos = np.array(position) + np.array(plane[:3])
        print("Pos: ", pos)
        ax.scatter3D(pos[0], pos[1], pos[2], c='g')
        ax.scatter3D(pos[0], position[1], pos[2], c='k')
        ax.scatter3D(position[0], position[1], position[2], c='y')
        R_m = R.from_quat(orientation).as_dcm()
        plot_basis(ax, R_m, np.array(position), alpha=1)

        ax.set_xlim(-2. + position[0], 2.+ position[0])
        ax.set_ylim(-2.+ position[1], 2.+ position[1])        
        ax.set_zlim(-2.+ position[2], 2.+ position[2])
        ax.set_aspect('equal', 'box')

        plt.show()

    def __find_plane(self, x, y, z, Trans, Rot, mean_pt, debug=False):
        #mean_pt = mean_pt/np.linalg.norm(mean_pt)
        pts = np.array([x, y, z]).T
        print(pts.shape)
        t_size = np.min((500, pts.shape[0]))
        t_pts = np.random.choice(np.arange(pts.shape[0]), size=t_size, replace=False)
        pts = pts[t_pts]
        print(pts.shape)
        all_points = open3d.utility.Vector3dVector(pts)
        pc = geometry.PointCloud(all_points)
        plane, success_pts = pc.segment_plane(0.05, int(pts.shape[0]*0.7), 500)
        plane = np.array(plane)

        # print("Plane:", pts.shape, len(success_pts))

        if debug:
            print("Plane: ", plane)
            print("Mean pt: ", mean_pt)
            plane_debug = plane.copy()
        d = np.dot(mean_pt, plane[:3])

        if d > 0:
            plane = -plane
        
        plane_ = plane[:3] #np.matmul(R.from_quat(Rot).as_dcm(), plane[:3].T)
        plane_[1] = 0
        plane_ /= np.linalg.norm(plane_)
        y = np.array([0, -1, 0])
        z = np.cross(plane_, y)
        rot = np.array([plane_, y, z]).T
        # print(rot, np.linalg.det(rot), R.from_dcm(rot).as_euler('ZYX', degrees=True))
        orientation = R.from_dcm(rot).as_quat()
        if debug:
            self.debugPlanePlotter(pts, plane, orientation, position=mean_pt)


        normal_vec = Pose()
        normal_vec.position.x = plane_[0]
        normal_vec.position.y = plane_[1]
        # normal_vec.position.z = plane[2]
        #orientation = [0, 0, 0, 1]#self.__find_rot(plane[0:3])
        #normal_vec.orientation.x = orientation[0]
        #normal_vec.orientation.y= orientation[1]
        #normal_vec.orientation.z= orientation[2]
        #normal_vec.orientation.w = orientation[3]

        vecs = PoseArray()
        vecs.poses.append(normal_vec)
        header = std_msgs.msg.Header()
        header.stamp = self.stamp_now #rospy.Time.now()
        header.frame_id = "orange"

        vecs.header = header
        self.__normal_vec_publisher.publish(vecs)
        # print(plane)
        return orientation

    def publishData(self, depth_image, camera_intrinsics, area, publ, Trans, Rot, mean_pt = None, norm_tracker=False, tracker=False):
        if area.shape[0] > 30:
            # print(area.shape)
            x, y, z = self.__convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics, area)
            #print(x.shape, y.shape, z.shape)
            if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(z)):
                return 
            # print(np.mean(x), np.mean(y), np.mean(z))
            # print(np.min(x),np.max(x),np.min(y),np.max(y), np.min(z),np.max(z))
            # print("\n\n\n")
            filter = True
            if filter:
                pts = np.array([x, y, z]).T
                pts = self.reject_outliers3d(pts)
                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            
            total_points = 1000
            step = int(x.shape[0]/total_points)
            step = np.max((1, step))
            #z -= 0.02
            self.__publishPointCloud(x, y, z, publ, step=step)

            if x.shape[0] < 10:
                return None
            """
            orientation = None
            if norm_tracker:
                if mean_pt is None:
                    print("ISSUUEE")
                    exit(0)
                    #mean_pt = np.array([np.mean(x), np.mean(y), np.mean(z)])
                orientation = self.__find_plane(x, y, z, Trans, Rot, mean_pt)
            if tracker:
                if orientation is None or mean_pt is None:
                    print("This should not happen!")
                    exit(0)
                self._publishGoalPoints(mean_pt[0], mean_pt[1], mean_pt[2], orientation=orientation, trans=Trans, rot=Rot, odom_data=None)
            else:
                pass
                #self._publishGoalPoints(x, y, z, orientation=orientation, odom_data=None)
            """

            return np.array([np.mean(x), np.mean(y), np.mean(z)])
        else:
            print("Failed in Publish Data check")
    
    def reject_outliers(self, data, m=1.5):
        return data[np.multiply(np.abs(data[:, 0] - np.mean(data[:, 0])) < m * np.std(data[:, 0]), np.abs(data[:,1] - np.mean(data[:,1])) < m * np.std(data[:,1]))]
    

    def reject_outliers3d(self, data, m=3):
        return data[np.multiply(np.multiply(np.abs(data[:, 0] - np.mean(data[:, 0])) < m * np.std(data[:, 0]), np.abs(data[:,1] - np.mean(data[:,1])) < m * np.std(data[:,1])), np.abs(data[:,2] - np.mean(data[:,2])) < m * np.std(data[:,2]))]
    
    def reject_outliers3dMedian(self, data, m=3):
        return data[np.multiply(np.multiply(np.abs(data[:, 0] - np.median(data[:, 0])) < m * np.std(data[:, 0]), np.abs(data[:,1] - np.median(data[:,1])) < m * np.std(data[:,1])), np.abs(data[:,2] - np.median(data[:,2])) < m * np.std(data[:,2]))]
    

    def callback(self, image_data, depth_image_data, odom):
        t1 = time.time()
        self.stamp_now = image_data.header.stamp
        image = self.__rosmsg2np(image_data)
        depth_image = self.__depthnpFromImage(depth_image_data)

        image_tensor = self.__process4model(image)  

        seg_np, segimages = self.__segmentationInference(image_tensor)

        #centroid = self.__getCentroid(seg_np)

        camera_intrinsics = CameraIntrinsics()
        area = np.argwhere(seg_np == 1)
        pre_area = area.copy()
        # print("Orignal shape: ", area.shape)
        # """
        states = None
        """
        if self.enable_states:
            if area.shape[0] > 30:
                # print("Early return??", area.shape)
                # return
            
                area = self.reject_outliers(area)
                # print("No outlier shape: ", area.shape)

                mean_x, mean_y = np.mean(area, axis=0)

                min_ = np.min(area, axis=0)
                max_ = np.max(area, axis=0)
                min_x, min_y = min_[0], min_[1]
                max_x, max_y = max_[0], max_[1]
                size_x = np.abs(max_x - min_x)
                size_y = np.abs(max_y - min_y)
                size_ = np.max((size_x, size_y))
                # mult_x, mult_y = 1.5, 1.5
                #extra_min_x, extra_max_x = np.max((0, min_x-int(mult_x*size_x))), np.min((seg_np.shape[0], max_x+int(mult_x*size_)))
                #extra_min_y, extra_max_y = np.max((0, min_y-int(mult_y*size_y))), np.min((seg_np.shape[1], max_y+int(mult_y*size_)))
                mult_x, mult_y = 2., 2.
                extra_min_x, extra_max_x = np.max((0, mean_x-int(mult_x*size_x))), np.min((seg_np.shape[0], mean_x+int(mult_x*size_)))
                extra_min_y, extra_max_y = np.max((0, mean_y-int(mult_y*size_y))), np.min((seg_np.shape[1], mean_y+int(mult_y*size_)))
                
                nx = np.linspace(extra_min_x, extra_max_x-1, extra_max_x - extra_min_x, dtype=np.int32)
                ny = np.linspace(extra_min_y, extra_max_y-1, extra_max_y - extra_min_y, dtype=np.int32)
                extra_area = np.transpose([np.tile(nx, len(ny)), np.repeat(ny, len(nx))])
           """     
        trans, rot = None, None
        try:
            # (trans,rot) = self.listener.lookupTransform('world', 'camera_depth_optical_frame_filtered', rospy.Time(0))
            (trans,rot) = self.listener.lookupTransform('world', 'camera_depth_optical_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        if trans is None or rot is None:
            return

        mean_pt = np.array([0., 0., 0.])
        orientation = np.array([0., 0., 0.])


        # print(area.shape)
        if area.shape[0] > 30:
            # print("In orange: ")
            mean_pt = self.publishData(depth_image, camera_intrinsics, area, self.__pointcloud_publisher, Trans=trans, Rot=rot, tracker=False)
            if mean_pt is None:
                return
        else:
            print("Left pre-area check", area.shape, pre_area.shape)

        rp = R.from_quat(rot).as_euler("ZYX")[1:]
        #TODO: Change States Here
        if self.enable_states and self.states_version == 2:
            body_v = self._getBodyV(odom, rot)
            states = torch.tensor(np.concatenate((body_v, mean_pt, orientation, rp))).to(self.__device)
        else:    
            states = torch.tensor(np.concatenate((mean_pt, orientation, rp))).to(self.__device)
        states = torch.reshape(states, (1, states.shape[0]))
        states = states.type(torch.FloatTensor).to(self.__device)
    # """

        if self.enable_depth:
            depth_image /= 10000.
            depth_image -= self.__mean_depth_image
        else:
            depth_image = None

        seg_tensor_image = self.__process4InfModel(image_tensor, segimages, depth_image)

        if self.enable_multi_image:
            seg_tensor_image = get_queued_imgs(seg_tensor_image, self.__num_images)
    	    seg_tensor_image = seg_tensor_image.reshape((1,self.__num_channels*self.__num_images,self.__h,self.__w))
            seg_tensor_image = torch.tensor(seg_tensor_image)
            seg_tensor_image = seg_tensor_image.to(self.__device,dtype=torch.float)

        if self.enable_multi_control:
            prediction, max_pred = self.__predictionInferenceMultiControl(seg_tensor_image, states)
            print("State:", "Staging" if max_pred == 0 else "Final")
            # self._publishWaypoints(prediction)
            self._publishWaypointsMultiControl(prediction, max_pred)
        else:
            prediction = self.__predictionInference(seg_tensor_image, states)
            self._publishWaypoints(prediction)
        print("Total time", time.time()-t1)

def main():
    global enable_depth, enable_states
    rospy.init_node('flat_inference')
    node = FlatNode(image_topic="/camera/color/image_raw/uncompressed", depth_topic="/camera/aligned_depth_to_color/image_raw/uncompressed")
    ts = message_filters.ApproximateTimeSynchronizer([node.image_sub, node.depth_sub, node.odom_node], queue_size=2, slop=1.0,  allow_headerless=True)
    ts.registerCallback(node.callback)
    rospy.spin()


if __name__ == "__main__":
    main()
