import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import torch
import numpy as np
import time, os
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from segmentation.segmentnetarch import *
from sensor_msgs.msg import Image
from customTransforms import *
import message_filters


class BaselineOrangeFinder:
    def __init__(self, image_topic='/camera/color/image_raw', depth_topic='/camera/aligned_depth_to_color/image_raw'):
        self.image_sub = message_filters.Subscriber(image_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        
        self.__params()
        
        self.__bridge = CvBridge()

        self.__segmodel = SegmentationNet()
        self.__loadModel()
        torch.no_grad()

        self.__loadMeanImage()

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
            device = torch.device('cuda:'+str(self.__gpu))
            self.__segmodel = self.__segmodel.to(device)
        else:
            device = torch.device('cpu')
            self.__segmodel = self.__segmodel.to(device)

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
            image_arr = self.__bridge.imgmsg_to_cv2(data,"passthrough")
        except CvBridgeError as e:
            print(e)
        
        return image_arr

    def __process4model(self, image_arr):
        image_arr = image_arr.transpose(2,0,1)
        image_arr = image_arr.reshape((1,3,self.__h,self.__w))

        image_tensor = torch.tensor(image_arr)
        image_tensor = image_tensor.to(device,dtype=torch.float)

        return image_tensor

    def __segmentationInference(self, image_tensor):
        seglogits = self.__segmodel(image_tensor)
        seglogits = seglogits.view(-1,2,self.__segmodel.h,self.__segmodel.w)
        segimages = (torch.max(seglogits, 1).indices).to('cpu') #.type(torch.FloatTensor).to(device)
        seg_np = np.array(segimages[0,:,:])

        return seg_np

    def __depthnpFromImage(self, image):
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
        data = data[:,:,0].astype(np.float64)
        cm_from_pixel = 0.095
        return data*cm_from_pixel

    def __getCentroid(self, seg_image):
        centroid = ndimage.measurements.center_of_mass(seg_np)
        if np.isnan(centroid[0]) or np.isnan(centroid[1]):
            return None
        #if (c[0] < 100) or (c[1] < 100):
        #    PIL.Image.fromarray((seg_np*255).astype('uint8')).show()
        return centroid

    def callback(self, image_data, depth_image_data):
        image = self.__rosmsg2np(image_data)
        depth_image = self.__depthnpFromImage(depth_image_data)

        image_tensor = self.__process4model(image)  
        
        seg_np = self.__segmentationInference(image_tensor)
        centroid = self.__getCentroid(seg_np)





def main():
    bof = BaselineOrangeFinder()
    ts = message_filters.TimeSynchronizer([bof.image_sub, bof.depth_sub], 20)
    ts.registerCallback(bof.callback)
    rospy.spin()


if __name__ == "__main__":
    main()