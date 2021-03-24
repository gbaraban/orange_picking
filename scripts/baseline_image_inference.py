import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import torch
import numpy as np
import time, os
from geometry_msgs.msg import Pose, PoseArray, Point32
from sensor_msgs.msg import PointCloud
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from segmentation.segmentnetarch import *
from sensor_msgs.msg import Image
from customTransforms import *
import message_filters


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

class BaselineOrangeFinder:
    def __init__(self, image_topic='/d400/color/image_raw', depth_topic='/d400/aligned_depth_to_color/image_raw'):
        print(image_topic, depth_topic)
        self.image_sub = message_filters.Subscriber(image_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        
        self.__params()
        
        self.__bridge = CvBridge()

        self.__segmodel = SegmentationNet()
        self.__loadModel()
        torch.no_grad()

        self.__loadMeanImages()

        self.__pointcloud_publisher = rospy.Publisher("/pointcloud_topic", PointCloud, queue_size=25)
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

        if not self.__gpu is None:
            self.__device = torch.device('cuda:'+str(self.__gpu))
            self.__segmodel = self.__segmodel.to(self.__device)
        else:
            self.__device = torch.device('cpu')
            self.__segmodel = self.__segmodel.to(self.__device)

        self.__segmodel.eval()
    
    def __loadMeanImages(self):
        if not (os.path.exists(self.__mean_image_loc)):
            print('mean image file not found', self.__mean_image_loc)
            exit(0)
        else:
            print('mean image file found')
            self.__mean_image = np.load(self.__mean_image_loc)
        
    def __params(self, gpu=True, relative=True):
        self.__num_images = 1
        self.__segload = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/model_seg145.pth.tar"
        self.__mean_image_loc = "/home/siddharth/Desktop/orange_picking/data/mean_imgv2_data_data_collection4_real_world_traj_bag.npy"
        
        if torch.cuda.is_available() and gpu:
            self.__gpu = 0
        else:
            self.__gpu = None

        self.__h = 480
        self.__w = 640

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
        cv2.imwrite('test.png', pub_np)

        return seg_np

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
            u = area[:, 0]
            v = area[:, 1]

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

    def __publishPointCloud(self, x, y, z, step=100):
        pointcloud = PointCloud()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_link'
        pointcloud.header = header

        for i in range(0, len(x), step):
            pointcloud.points.append(Point32(x[i], y[i], z[i]))

        self.__pointcloud_publisher.publish(pointcloud)


    def callback(self, image_data, depth_image_data):
        print("Reaching callback")
        t1 = time.time()
        image = self.__rosmsg2np(image_data)
        depth_image = self.__depthnpFromImage(depth_image_data)

        image_tensor = self.__process4model(image)  

        seg_np = self.__segmentationInference(image_tensor)
        #centroid = self.__getCentroid(seg_np)

        camera_intrinsics = CameraIntrinsics()
        area = np.argwhere(seg_np == 1)
        print(area.shape)
        if area.shape[0] != 0:
            x, y, z = self.__convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics, area)
            print(x.shape, y.shape, z.shape)
            print(np.mean(x), np.mean(y), np.mean(z))
            print(np.min(x),np.max(x),np.min(y),np.max(y), np.min(z),np.max(z))
            # print("\n\n\n")
    
            self.__publishPointCloud(x, y, z, step=1)
            print(time.time()-t1)
            print("x: ", 3.25 - 0.91)
            print("y: ", -0.03 - 0.13)
            print("z: ", 1.078 - 1.41)
            

def main():
    rospy.init_node('baseline_inference')
    bof = BaselineOrangeFinder(image_topic="/camera/color/image_raw", depth_topic="/camera/aligned_depth_to_color/image_raw")
    ts = message_filters.ApproximateTimeSynchronizer([bof.image_sub, bof.depth_sub], queue_size=20, slop=0.1,  allow_headerless=True)
    ts.registerCallback(bof.callback)
    rospy.spin()


if __name__ == "__main__":
    main()