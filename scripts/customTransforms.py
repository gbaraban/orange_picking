import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import PIL.Image as img
from torchvision.transforms import functional

#TODO: Add custom image transformations here

def tensortoPIL(tensor):
    tensor = tensor.cpu()
    if len(tensor.shape) == 3:
        tensor = np.transpose(tensor,[1,2,0])
        if tensor.shape[2] == 1:
            tensor = tensor[:,:,0]
    return img.fromarray(np.uint8(tensor*255))

def PILtotensor(im):
    print(im.size)
    arr = np.array(im.getdata()).reshape(im.size[1],im.size[0],3).transpose([2,0,1])
    return arr/255.0


def saveImage(tensor,name,num_images=1):
    if len(tensor.shape) is 3:
        num_c = tensor.shape[0]
    else:
        num_c = 1
    num_c = num_c/num_images
    if num_images is 1:
        if (num_c == 3) or (num_c == 1):
            pil = tensortoPIL(tensor)
            pil.save(name+".png")
            return
        if (num_c == 2):
            saveImage(tensor[0,:,:],name + "depth")
            saveImage(tensor[1,:,:],name + "seg")
            return
        saveImage(tensor[0:3,:,:],name+"_color")
        saveImage(tensor[3:,:,:],name+"_")
        return
    for ii in range(num_images):
        temp_name = name + "n" + str(ii)
        temp_tensor = tensor[(num_c*ii):(num_c*(ii+1)),:,:]
        saveImage(temp_tensor,temp_name)
    return
    #new_t = PILtotensor(pil)
    #res = new_t - tensor
    #print(res.max())
    #print(res.min())

class CameraIntrinsics:
    def __init__(self):
        self.ppx = 333.42010498046875
        self.ppy = 250.38986206054688
        self.fx = 607.265625
        self.fy = 607.2756958007812

class WaypointPerspective(object):
    def __init__(self, p, bound, relative_flag):
        self.C = np.array([[0,1,0],[0,0,1],[1,0,0]])
        self.cam = CameraIntrinsics()
        self.p = p
        self.bound = bound
        self.rel = relative_flag

    def world2pixel(self,p):
        print("w2p")
        print(p)
        p = np.matmul(self.C,p)
        print(p)
        p = np.array([p[0]*self.cam.fx*p[2],p[1]*self.cam.fy*p[2]]) 
        print(p)
        p = p + np.array([self.cam.ppx,self.cam.ppy])
        print(p)
        return p

    def __call__(self, data):
        image = data["img"]
        saveImage(image,"pre_image")
        points = np.array(data["pts"])
        rot_list = None
        if "rots" in data.keys():
            rot_list = np.array(data["rots"])
        rotated = False
        if np.random.random() > self.p:
            rotated = True
            yaw = 0.5#(np.random.sample()*2*self.bound) - self.bound
            print("yaw: " + str(yaw))
            pitch = 0
            roll = 0
            rot = R.from_euler('ZYX', [yaw,pitch,roll],degrees=False).as_dcm()
            new_points = np.zeros(points.shape)
            new_rots = None
            if rot_list is not None:
                new_rots = np.zeros(rot_list.shape)
            if self.rel:
                new_points[0,:] = np.matmul(rot,points[0,:])
                new_points[1:,:] = points[1:,:]
                if rot_list is not None:
                    new_rots[0,:,:] = np.matmul(rot,rot_list[0,:,:])
                    new_rots[1:,:] = rot_list[1:,:,:]
            else:
                for i in range(len(points)):
                    new_points[i,:] = np.matmul(rot,points[i,:])
                    if rot_list is not None:
                        new_rots[i,:,:] = np.matmul(rot,rot_list[i,:,:])
            start_points = np.zeros((4,2))
            end_points = np.zeros((4,2))
            temp_points = [points[0],
                           points[0] + [0,-0.5,0],
                           points[0] + [0,-0.5,-0.5],
                           points[0] + [0,0,-0.5]]
            for i in range(4):
                start_points[i,:] = self.world2pixel(temp_points[i])
                end_points[i,:] = self.world2pixel(np.matmul(rot,temp_points[i]))
            print(start_points)
            print(end_points)
            pil_image = tensortoPIL(image)
            coeffs = functional._get_perspective_coeffs(start_points, end_points)
            print(coeffs)
            pil_image = functional.perspective(pil_image,start_points,start_points)
            image = PILtotensor(pil_image)
            saveImage(image,"post_image")
            points = new_points
            if rot_list is not None:
                rot_list = new_rots
        data["img"] = image
        data["pts"] = points
        data["rots"] = rot_list
        data["rotated"] = rotated
        exit()
        return data

class sphericalToXYZ(object):
    def transformFunc(self, v):
        r = v[0]
        theta = v[1]
        phi = v[2]
        x = r*np.cos(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.cos(phi)
        z = r*np.sin(phi)
        if len(v) == 3:
            return np.array([x,y,z])
        return np.hstack((x,y,z,v[3:6]))

    def __call__(self,points):
        local_pts = []
        for point in points:
            if (point[0].shape == ()):
                local_pts.append(self.transformFunc(point))
            else:
                local_pts.append(self.__call__(self,point))
        return np.array(local_pts)

class xyzToSpherical(object):
    def transformFunc(self, v):
        x = v[0]
        y = v[1]
        z = v[2]
        r = np.linalg.norm(v[0:3])
        if r < self.r_thesh:
            theta = 0
            phi = 0
        else:
            theta = np.arctan2(y,x)
            phi = np.arctan2(z,r)
        if len(v) == 3:
            return np.array((r,theta,phi))
        return np.hstack((r,theta,phi,v[3:6]))
 
    def __init__(self,r_theshold = 1e-5):
        self.r_thesh = r_theshold

    def __call__(self,points):
        local_pts = []
        for point in points:
            #print(len(point), len(points))
            if (point[0].shape == ()):
                local_pts.append(self.transformFunc(point))
            else:
                local_pts.append(self.__call__(self,point))
        return np.array(local_pts)

class pointToBins(object):
    def __init__(self,min_list,max_list, bins, extra_min = None, extra_max = None):
        self.min_list = min_list
        self.max_list = max_list
        self.bins = bins
        self.extra_min = extra_min
        self.extra_max = extra_max

    def __call__(self,points):
        phase = 0
        if type(points) is dict:
            phase = points['phase']
            points = points['points']

        local_pts = []
        if (phase is None) or (self.extra_min is None) or (self.extra_max is None):
            bound_min = self.min_list
            bound_max = self.max_list
        else:
            if phase > 0:
                bound_min = self.extra_min[phase-1]
                bound_max = self.extra_max[phase-1]
            else:
                bound_min = self.min_list
                bound_max = self.max_list
        for ctr, point in enumerate(points):
            #Convert into bins
            #print(len(point), len(points))
            min_i = np.array(bound_min[ctr]).astype(float)
            max_i = np.array(bound_max[ctr]).astype(float)
            point = np.array(point).astype(float)
            if phase is 3:
                bin_nums = np.zeros(point.shape)
            else:
                bin_nums = (point - min_i)/(max_i-min_i)
                bin_nums_scaled = (bin_nums*self.bins).astype(int)
                bin_nums = np.clip(bin_nums_scaled,a_min=0,a_max=self.bins-1)
            local_pts.append(bin_nums)
        return np.array(local_pts).astype(np.long)

class GaussLabels(object):
    def __init__(self,mean,stdev,bins):
        self.mean = mean
        self.stdev = stdev
        self.bins = bins

    def __call__(self,bin_list):
        label_list = []
        num_points = len(bin_list)
        for bins in bin_list:
            labels = np.zeros((3,self.bins))
            for coord in range(3):#
                for i in range(labels.shape[1]):
                    labels[coord][i] = self.mean * (np.exp((-np.power(bins[coord]-i, 2))/(2 * np.power(self.stdev, 2))))
            label_list.append(labels)
        label_list = np.array(label_list)
        label_list.resize((num_points,3,self.bins))
        return label_list

class RandomHorizontalTrajFlip(object):
    def __init__(self, p=0.5, n_inputs = 6):
        self.p = p
        self.reflect = np.diag((1,-1,1,1))
        #self.reflect = np.zeros((4,4))
        #self.reflect[0,0] = 1
        #self.reflect[1,1] = -1
        #self.reflect[2,2] = 1
        #self.reflect[3,3] = 1
        self.n_inputs = n_inputs

    def __call__(self, data):
        image = data["img"]
        raw_image = data["raw_img"]
        points = np.array(data["pts"])
        rot_list = None
        if "rots" in data.keys():
            rot_list = np.array(data["rots"])

        flipped = False
        if np.random.random() > self.p:
            flipped = True
            image = np.flip(image,axis=2).copy()
            raw_image = np.flip(raw_image,axis=2).copy()
            for i in range(len(points)):
                if rot_list is not None:
                    E = np.zeros((4,4))
                    E[3,3] = 1
                    E[0:3,3] = np.array(points[i,:])
                    E[0:3,0:3] = rot_list[i,:,:]
                    E_flip = np.matmul(np.matmul(self.reflect,E),self.reflect)
                    points[i,:] = E_flip[0:3,3]
                    rot_list[i,:,:] = E_flip[0:3,0:3]
                else:
                    E = np.zeros((4))
                    E[0:3] = np.array(points[i,:])
                    E[3] = 1
                    E_flip = np.matmul(np.matmul(self.reflect,E),self.reflect)
                    points[i,:] = E_flip[0:3]
            points = np.array(points)
            if rot_list is not None:
                rot_list = np.array(rot_list)
        data["img"] = image
        data["raw_img"] = raw_image
        data["pts"] = points
        data["rots"] = rot_list
        data["flipped"] = flipped
        return data
