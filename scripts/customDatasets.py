import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
import PIL.Image as img
import sys
from torch.utils.data import Dataset, DataLoader
from customTransforms import *
import pickle


class SubSet(Dataset):
    def __init__(self,dataset,idx):
        self.ds = dataset
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self,i):
        return self.ds[self.idx[i]]

class OrangeSimDataSet(Dataset):
    def __init__(self, root_dir, num_images, num_pts, pt_trans, img_trans, dt = 1, reduce_N = True):
        self.point_transform = pt_trans
        self.image_transform = img_trans
        self.num_pts = num_pts
        self.num_images = num_images
        self.dt = dt
        self.num_list = []
        self.traj_list = []
        self.run_dir = root_dir
        self.np_dir = root_dir.rstrip("/") + "_np/"
        self.trial_list = os.listdir(root_dir)
        self.num_samples = 0
        if reduce_N:
            time_window = num_pts*dt
        else:
            time_window = 0
        for trial_dir in self.trial_list:
            with open(self.run_dir+'/'+trial_dir+'/trajdata.pickle','rb') as f:
                data = pickle.load(f)#,encoding='latin1')
                self.traj_list.append(data)
            with open(self.run_dir+"/"+trial_dir+"/metadata.pickle",'rb') as data_f:
                data = pickle.load(data_f)#, encoding='latin1')
                N = data['N']
                tf = data['tf']
                self.h = float(N)/tf
                if reduce_N:
                    reduced_N = int(N - time_window*self.h)
                    self.num_samples += reduced_N
                    self.num_list.append(reduced_N)
                else:
                    self.num_samples += N
                    self.num_list.append(N)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,i):
        idx = i
        trial_idx = 0
        while True:
            if self.num_list[trial_idx] <= idx:
                idx = idx-self.num_list[trial_idx]
                trial_idx += 1
            else:
                break
        trial = self.trial_list[trial_idx]
        image = None
        image_idx = idx
        for ii in range(self.num_images):
            temp_idx = max(0,image_idx - ii)
            temp_image = np.load(self.np_dir+trial+'/image'+str(temp_idx)+'.npy')
            temp_image = np.transpose(temp_image,[2,0,1])
            if self.image_transform:
                temp_image = self.image_transform(temp_image)
            if image is None:
                image = temp_image
            else:
                image = np.concatenate((image,temp_image),axis=2)
        image = image.astype('float32')
        p0 = np.array(self.traj_list[trial_idx][idx][0])
        R0 = np.array(self.traj_list[trial_idx][idx][1])
        points = []
        for pt in range(self.num_pts):
            temp_idx = int(idx + self.h*self.dt*(pt+1))
            p = self.traj_list[trial_idx][temp_idx][0]
            p = np.array(p)
            p = np.matmul(R0.T,p-p0)
            points.append(p)
        if self.point_transform:
           points = self.point_transform(points)
        return {'image':image, 'points':points}
