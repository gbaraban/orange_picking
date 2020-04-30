import torch
import numpy as np

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

