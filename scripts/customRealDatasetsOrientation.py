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
from scipy.spatial.transform import Rotation as R


class SubSet(Dataset):
    def __init__(self,dataset,idx):
        self.ds = dataset
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self,i):
        return self.ds[self.idx[i]]

class OrangeSimDataSet(Dataset):
    def __init__(self, root_dir, num_images, num_pts, pt_trans, img_trans, dt = 1, reduce_N = True, custom_dataset = None):
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
        self.num_samples_dir = {}
        self.num_samples_dir_size = {}
        self.custom_dataset = custom_dataset

        self.nEvents = []
        self.time_secs = []
        self.time_nsecs = []

        if reduce_N:
            time_window = num_pts*dt
        else:
            time_window = 0
        #for trial_dir in self.trial_list:
        for i, trial_dir in enumerate(self.trial_list):
            trial_subdir = os.listdir(root_dir + "/" + trial_dir)
            if self.custom_dataset is None:
                with open(self.run_dir+'/'+trial_dir+'/orientation_data.pickle','rb') as f:
                    data = pickle.load(f,encoding='latin1')
                    self.traj_list.append(data)
            elif self.custom_dataset == "no_parse":
                with open(self.run_dir+'/'+trial_dir+'/data.pickle_no_parse','rb') as f:
                    data = pickle.load(f,encoding='latin1')
                    self.traj_list.append(data["data"])

            #with open(self.run_dir+"/"+trial_dir+"/metadata.pickle",'rb') as data_f:
            #data = pickle.load(data_f)#, encoding='latin1')
            N = len(trial_subdir) - 3 #data['N']
            tf = 1 #data['tf']
            self.h = 0 #float(N)/tf
            if reduce_N:
                reduced_N = int(N - time_window*self.h)
                self.num_samples += reduced_N
                self.num_list.append(reduced_N)
            else:
                self.num_samples += N
                self.num_list.append(N)

            self.num_samples_dir[i] = {}
            if reduce_N:
                self.num_samples_dir[i]['start'] = self.num_samples - reduced_N
            else:
                self.num_samples_dir[i]['start'] = self.num_samples - N

            if self.custom_dataset == "no_parse":
                self.nEvents.append(data["nEvents"])
                self.time_secs.append(data["time_secs"])
                self.time_nsecs.append(data["time_nsecs"])

            self.num_samples_dir[i]['end'] = self.num_samples
            self.num_samples_dir[i]['size'] = self.num_samples_dir[i]['end'] - self.num_samples_dir[i]['start']
            self.num_samples_dir[i]['data'] = data
            self.num_samples_dir[i]['dir'] = trial_dir
            self.num_samples_dir_size[i] = self.num_samples_dir[i]['size']


    def __len__(self):
        return self.num_samples

    def getTrajLen(self):
        pass

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
            #if self.image_transform:
            #    temp_image = self.image_transform(temp_image)
            if image is None:
                image = temp_image
            else:
                image = np.concatenate((image,temp_image),axis=2)
        image = image.astype('float32')

        if self.custom_dataset is None:
            points = np.array(self.traj_list[trial_idx][idx])
            #print(points)
            #p0 = np.array(self.traj_list[trial_idx][idx][0])
            #R0 = np.array(self.traj_list[trial_idx][idx][1])
            #points = []
            #for pt in range(self.num_pts):
            #    temp_idx = int(idx + self.h*self.dt*(pt+1))
            #    p = self.traj_list[trial_idx][temp_idx][0]
            #    p = np.array(p)
            #    p = np.matmul(R0.T,p-p0)
            #    points.append(p)

            if self.point_transform:
                points = self.point_transform(points)

            #if self.image_transform:
            #    data = {}
            #    data["img"] = image
            #    data["pts"] = points
            #    image, points = self.image_transform(data)

            return {'image':image, 'points':points}

        elif self.custom_dataset == "no_parse":
            point = np.array(self.traj_list[trial_idx][idx])
            p0 = np.array(point[0:3])
            R0 = R.from_euler('zyx', point[3:6]).as_dcm()
            points = []
            h = (float(self.nEvents[trial_idx])/self.time_secs[trial_idx])
            indices = np.floor(np.add(np.array([1, 2, 3]) * h, idx)).astype(int)

            for i in range(len(indices)):
                if (indices[i] >= self.num_list[trial_idx]):
                    if i == 0:
                        indices[i] = idx
                    else:
                        indices[i] = indices[i-1]

            point_list = [p0]
            rot_list = [R0]

            for x, ii in enumerate(indices):
                if (ii < idx):
                        print(idx, ii)
                pt = np.array(self.traj_list[trial_idx][ii])
                p = np.array(pt[0:3])
                Ri = R.from_euler('zyx', pt[3:6]).as_dcm()
                point_list.append(p)
                rot_list.append(Ri)

            flipped = False
            if self.image_transform:
                data = {}
                data["img"] = image
                data["pts"] = point_list
                data["rots"] = rot_list
                image, point_list, rot_list, flipped = self.image_transform(data)

            p0 = np.array(point_list[0])
            R0 = np.array(rot_list[0])

            for ii in range(1,len(point_list)):
                p = np.array(point_list[ii])
                Ri = np.array(rot_list[ii])
                p = list(np.matmul(R0.T,p-p0))
                Ri = np.matmul(R0.T,Ri)
                Ri_zyx = list(R.from_dcm(Ri).as_euler('zyx'))
                p.extend(Ri_zyx)
                points.append(p)

            #points = np.matmul(np.array(R0).T, (np.array(points) - np.array(p0)).T).T

            if self.point_transform:
                points = self.point_transform(points)

            #if self.image_transform:
            #    data = {}
            #    data["img"] = image
            #    data["pts"] = points
            #    image, points = self.image_transform(data)

            return {'image':image, 'points':points, "flipped": flipped}

