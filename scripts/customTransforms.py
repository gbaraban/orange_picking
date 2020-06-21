import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

#TODO: Add custom image transformations here

class pointToBins(object):
    def __init__(self,min_list,max_list, bins):
        self.min_list = min_list
        self.max_list = max_list
        self.bins = bins

    def __call__(self,points):
        local_pts = []
        for ctr, point in enumerate(points):
            #Convert into bins
            min_i = np.array(self.min_list[ctr])
            max_i = np.array(self.max_list[ctr])
            point = np.array(point)
            bin_nums = (point - min_i)/(max_i-min_i)
            bin_nums_scaled = (bin_nums*self.bins).astype(int)
            bin_nums = np.clip(bin_nums_scaled,a_min=0,a_max=self.bins-1)
            local_pts.append(bin_nums)
        return np.array(local_pts)

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
		self.reflect = np.zeros((4,4))
		self.reflect[0,0] = -1
		self.reflect[1,1] = 1
		self.reflect[2,2] = 1
		self.reflect[3,3] = 1
		self.n_inputs = n_inputs

	def __call__(self, data):
		image = data["img"]
		points = np.array(data["pts"])
		rot_list = None
		if "rots" in data.keys():
			rot_list = np.array(data["rots"])

		if np.random.random() > self.p:
			image = np.fliplr(image).copy()
			"""
			for i, pt in enumerate(points):
				if self.n_inputs == 6:
					E = np.zeros((4,4))
					E[3,3] = 1
					E[0:3,3] = np.array(pt[0:3])
					E[0:3,0:3] = R.from_euler('zyx', pt[3:]).as_dcm()
					E = np.matmul(self.reflect, E)

					points[i,:3] = list(E[0:3,3])
					points[i,3:] = R.from_dcm(E[0:3,0:3]).as_euler('zyx')

				else:
					E = np.zeros((4))
					E[0:3] = np.array(pt[0:3])
					E[3] = 1
					E = np.matmul(self.reflect, E)

					points[i,:3] = list(E[0:3])
			"""
			for i in range(len(points)):
				if rot_list is not None:
					E = np.zeros((4,4))
					E[3,3] = 1
					E[0:3,3] = np.array(points[i,:])
					E[0:3,0:3] = rot_list[i,:,:]
					E = np.matmul(self.reflect, E)

					points[i,:] = list(E[0:3,3])
					rot_list[i,:,:] = E[0:3,0:3]

				else:
					E = np.zeros((4))
					E[0:3] = np.array(points[i,:])
					E[3] = 1
					E = np.matmul(self.reflect, E)

					points[i,:] = list(E[0:3])


			points = np.array(points)
			if rot_list is not None:
				rot_list = np.array(rot_list)

		if rot_list is None:
			return image, points
		else:
			return image, points, rot_list
