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
from customTransforms import *
import message_filters

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *
from open3d import geometry
from open3d import open3d


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


class BaselineOrangeFinder:
    def __init__(self):      
        self.__params()

        self.__segmodel = SegmentationNet()
        self.__loadModel()
        torch.no_grad()

        self.__loadMeanImages()

        
        rot = Rotations()
        self.world2orange = None #np.eye(3)#np.matmul(rot.rotz(0), np.matmul(rot.rotx(np.pi/2),np.matmul(rot.rotz(np.pi/2), rot.roty(np.pi/2))))
        #self.world2orange = np.matmul(rot.rotz(np.pi/2), rot.roty(np.pi/2))

        self.world2orange_pos = None
        self.t = 0
      
        self.alpha = 0.5
        self.min_alpha = 0.1
        self.max_steps = 20

        print("Setup complete")

    def reset_pos(self):
        self.world2orange = None
        self.world2orange_pos = None

    def __loadModel(self):
        if os.path.isfile(self.__segload):
            if not self.__gpu is None:
                    checkpoint = torch.load(self.__segload,map_location=torch.device('cuda:'+str(self.__gpu)))
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
        self.__segload =  "/home/gabe/ws/ros_ws/src/orange_picking/useful_models/model_seg145.pth.tar"
        self.__mean_image_loc = "/home/gabe/ws/ros_ws/src/orange_picking/data/mean_imgv2_data_data_collection4_real_world_traj_bag.npy"
        
        if torch.cuda.is_available() and gpu:
            self.__gpu = 0
        else:
            self.__gpu = None

        self.__h = 480
        self.__w = 640

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
        #cv2.imwrite('test.png', pub_np)

        return seg_np

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
            z = depth_image.flatten() / 1000.
        else:
            z = depth_image[area[:,0], area[:,1]].flatten() / 1000.
        x = np.multiply(x,z)
        y = np.multiply(y,z)

        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]

        return x, y, z

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

        #decaying alpha in future
        rot = R.from_quat(np.array(rot))
        #fixed_transform = R.from_quat(np.array([-0.500, 0.500, 0.500, 0.500])).as_dcm()
        temp_world2orange_pos = np.matmul(rot.as_dcm(), mean_pos) + np.array(trans)
            
        if self.world2orange_pos is None:
            self.world2orange_pos = temp_world2orange_pos.copy()
        else:
            self.world2orange_pos = self.world2orange_pos + self.alpha*(temp_world2orange_pos-self.world2orange_pos)

        if orientation is not None:
            orientation = R.from_dcm(np.matmul(rot.as_dcm(),R.from_quat(orientation).as_dcm())).as_quat()
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

        # self.br.sendTransform((self.world2orange_pos[0], self.world2orange_pos[1], self.world2orange_pos[2]),
        #              R.from_dcm(self.world2orange).as_quat(),
        #              self.stamp_now, #rospy.Time.now(),
        #              "orange",
        #              "world")

        Rot = R.from_dcm(np.matmul(np.linalg.inv(rot.as_dcm()), self.world2orange)).as_quat()
        Trans = np.matmul(np.linalg.inv(rot.as_dcm()), (self.world2orange_pos - np.array(trans)))
        return Rot, Trans


    def __find_plane(self, x, y, z, Trans, Rot, mean_pt, debug=False):
        #mean_pt = mean_pt/np.linalg.norm(mean_pt)
        pts = np.array([x, y, z]).T
        all_points = open3d.utility.Vector3dVector(pts)
        #mean_pt = mean_pt/np.linalg.norm(mean_pt)
        pc = geometry.PointCloud(all_points)
        n_pts =  max(3, int(pts.shape[0]*0.8))
        plane, success_pts = pc.segment_plane(0.1, n_pts, 1000)
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
        #print(rot, np.linalg.det(rot), R.from_dcm(rot).as_euler('zyx', degrees=True))
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

        # vecs = PoseArray()
        # vecs.poses.append(normal_vec)
        # header = std_msgs.msg.Header()
        # header.stamp = self.stamp_now #rospy.Time.now()
        # header.frame_id = "orange"

        # vecs.header = header
        # self.__normal_vec_publisher.publish(vecs)
        # print(plane)
        return orientation


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
        

    def publishDataLoc(self, depth_image, camera_intrinsics, area, mean_pt = None, norm_tracker=False, tracker=False):
        # if area.shape[0] < 30:
        #     return None

        # print(area.shape)
        x, y, z = self.__convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics, area)
        #print(x.shape, y.shape, z.shape)
        if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(z)):
            return 
        # print(np.mean(x), np.mean(y), np.mean(z))
        # print(np.min(x),np.max(x),np.min(y),np.max(y), np.min(z),np.max(z))
        # print("\n\n\n")
        # filter = True
        # if filter:
        #     pts = np.array([x, y, z]).T
        #     pts = self.reject_outliers3d(pts)
        #     x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        
       
        return np.array([np.mean(x), np.mean(y), np.mean(z)])

    def publishData(self, depth_image, camera_intrinsics, area, Trans, Rot, mean_pt = None, norm_tracker=False, tracker=False):
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
        
        orientation = None
        if norm_tracker:
            if mean_pt is None:
                print("ISSUUEE")
                exit(0)
                #mean_pt = np.array([np.mean(x), np.mean(y), np.mean(z)])
            if len(x) is 0:
                print("Empty List")
                return
            orientation = self.__find_plane(x, y, z, Trans, Rot, mean_pt)
        if tracker:
            if orientation is None or mean_pt is None:
                print("This should not happen!")
                exit(0)
            #self._publishGoalPoints(mean_pt[0], mean_pt[1], mean_pt[2], orientation=orientation, trans=Trans, rot=Rot, odom_data=None)
            orientation, mean_pos = self.__orange_orientation(Trans, Rot, mean_pt, orientation)
            return mean_pos, orientation
        else:
            pass
            #self._publishGoalPoints(x, y, z, orientation=orientation, odom_data=None)

        # if odom_data is not None:
        #     rot = np.array([odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])
        

        return np.array([np.mean(x), np.mean(y), np.mean(z)])

    
    def reject_outliers(self, data, m=1.5):
        return data[np.multiply(np.abs(data[:, 0] - np.mean(data[:, 0])) < m * np.std(data[:, 0]), np.abs(data[:,1] - np.mean(data[:,1])) < m * np.std(data[:,1]))]
    

    def reject_outliers3d(self, data, m=3):
        return data[np.multiply(np.multiply(np.abs(data[:, 0] - np.mean(data[:, 0])) < m * np.std(data[:, 0]), np.abs(data[:,1] - np.mean(data[:,1])) < m * np.std(data[:,1])), np.abs(data[:,2] - np.mean(data[:,2])) < m * np.std(data[:,2]))]
    
    def reject_outliers3dMedian(self, data, m=3):
        return data[np.multiply(np.multiply(np.abs(data[:, 0] - np.median(data[:, 0])) < m * np.std(data[:, 0]), np.abs(data[:,1] - np.median(data[:,1])) < m * np.std(data[:,1])), np.abs(data[:,2] - np.median(data[:,2])) < m * np.std(data[:,2]))]
    

    def process_loc(self, image_data, depth_image_data, trans, rot, world=False):
        self.t += 1
        t1 = time.time()
        image = image_data
        depth_image = depth_image_data

        image_tensor = self.__process4model(image)  

        seg_np = self.__segmentationInference(image_tensor)
        #centroid = self.__getCentroid(seg_np)
        #cv2.imwrite("seg.png", (seg_np*255).astype(np.uint8))

        camera_intrinsics = CameraIntrinsics()
        area = np.argwhere(seg_np == 1)
        
        area = self.reject_outliers(area)
        
        if area.shape[0] <= 5:
            return None

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
        
        if trans is None or rot is None:
            return


        #print(area.shape)
        # if area.shape[0] > 30:
        # mean_pt = self.publishData(depth_image, camera_intrinsics, area, tracker=False)
        mean_pt = self.publishData(depth_image, camera_intrinsics, area, Trans=trans, Rot=rot, tracker=False)
        temp = self.publishData(depth_image, camera_intrinsics, extra_area, Trans=trans, Rot=rot, mean_pt=mean_pt, norm_tracker=True, tracker=True)
        if temp is None:
            return
        mean_pos, orientation = temp
        #print(mean_pt, orientation)
        if world:
            return self.world2orange_pos, self.world2orange
        else:
            return mean_pt, orientation


def main():
    bof = BaselineOrangeFinder()
    
    img = "/home/gabe/ws/ros_ws/src/orange_picking/data/v1_flights/test_v1/real_world_traj_bag/bag0/trial0/staging/image0.png"
    a = Image.open(img)
    a = np.array(a)
    #print(a.shape, np.min(a), np.max(a))

    depth_img = "/home/gabe/ws/ros_ws/src/orange_picking/data/v1_flights/test_v1/real_world_traj_bag/bag0/trial0/staging/depth_image0.npy"
    b = np.load(depth_img)
    #print(b.shape, np.min(b), np.max(b))
    trans = np.array([1., 1., 1.])
    rot = np.array([0.707, 0., .707, 1])
    mean_pt, orientation = bof.process_loc(a, b, trans, rot)

    print(mean_pt, orientation)
    
    # ax = plot_basis(R=np.eye(3), ax_s=2)
    # rot = R.from_quat(orientation).as_dcm()
    # plot_basis(ax, rot, mean_pt, alpha=0.5)
    # plt.show()



if __name__ == "__main__":
    main()