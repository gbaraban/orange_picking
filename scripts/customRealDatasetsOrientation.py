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
    def __init__(self, root_dir, num_images, num_pts, pt_trans, img_trans, dt = 1.0, reduce_N = True, custom_dataset = None, input = 1.0, depth=False, seg=False, temp_seg=False,
	seg_only=False, rel_pose=False, gaussian_pts=False, pred_dt=1.0):
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
        if seg or seg_only: #temp_seg
            self.trial_list = os.listdir(root_dir + "../seg_mask_real_np/")
        self.num_samples = 0
        self.num_samples_dir = {}
        self.num_samples_dir_size = {}
        self.custom_dataset = custom_dataset
        self.depth = depth
        self.seg = seg
        self.seg_only = seg_only
        self.mean_seg = np.load("data/mean_imgv2_data_seg_real_world_traj_bag.npy")
        self.relative_pose = rel_pose
        self.pred_dt = pred_dt

        self.nEvents = []
        self.time_secs = []
        self.time_nsecs = []
        self.hz = []
        self.time_multiplier = []

        self.gaussian_pts = gaussian_pts
        self.gaussian_var = [0.012, 0.012, 0.012, 0.04, 0.02, 0.02]
        self.gaussian_limit = [0.025, 0.025, 0.025, 0.08, 0.04, 0.04]

        if reduce_N:
            time_window = num_pts*dt
        else:
            time_window = 0
        #for trial_dir in self.trial_list:
        for i, trial_dir in enumerate(self.trial_list):
            trial_subdir = None
            if self.seg_only:
                trial_subdir = os.listdir(root_dir + "../seg_mask_real_np/" + trial_dir)
            else:
                trial_subdir = os.listdir(root_dir + "/" + trial_dir)
            #if self.seg:
            #    trial_subdir = os.listdir(root_dir + "/../seg_mask_real/")
            #    print(trial_subdir, root_dir)
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
            #N = len(trial_subdir) - 3 #data['N']
            #if self.depth:
            #    N = N/2 #TODO: when we add new channels, this needs to get more complex.
            N = 0
            if trial_subdir is None:
                print("This should not be None")

            for fname in trial_subdir:
                if not self.seg_only:
                    if ("image" in fname) and ("depth" not in fname) and ("seg" not in fname):
                        N += 1
                else:
                    N += 1

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
                self.time_multiplier.append(data["time_multiplier"])
                self.time_secs.append(data["time_secs"])
                self.time_nsecs.append(data["time_nsecs"])

            self.num_samples_dir[i]['end'] = self.num_samples
            self.num_samples_dir[i]['size'] = self.num_samples_dir[i]['end'] - self.num_samples_dir[i]['start']
            self.hz.append(self.num_samples_dir[i]['size']/(data["time_secs"]*data["time_multiplier"]))
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

        time_frac = float(idx)/float(self.num_list[trial_idx])
        trial = self.trial_list[trial_idx]
        image = None
        image_idx = idx
        for ii in range(self.num_images):
            temp_idx = max(0,image_idx - int(ii*self.dt*self.hz[trial_idx]))
            if not self.seg_only:
                temp_image = np.load(self.np_dir+trial+'/image'+str(temp_idx)+'.npy')
            else:
                seg_dir = os.listdir(self.np_dir+"../seg_mask_real_np/"+trial)
                seg_img = seg_dir[temp_idx][:-4]
                temp_image = np.load(self.np_dir+trial+'/image'+ seg_img +'.npy')

            temp_image = np.transpose(temp_image,[2,0,1])
            #print("Just Image", temp_image.shape)
            if self.depth:
                if self.seg_only:
                    temp_depth = np.load(self.np_dir+trial+'/depth_image'+seg_img+'.npy')
                else:
                    temp_depth = np.load(self.np_dir+trial+'/depth_image'+str(temp_idx)+'.npy')
                temp_depth = np.expand_dims(temp_depth,0)
                temp_image = np.concatenate((temp_image,temp_depth),axis=0)
            #print("with depth", temp_image.shape)
            if self.seg and not self.seg_only:
                if os.path.isfile(self.np_dir+"../seg_mask_real_np/"+trial+'/'+str(temp_idx)+'.npy'):
                    temp_seg = np.load(self.np_dir+"../seg_mask_real_np/"+trial+'/'+str(temp_idx)+'.npy')
                else:
                    #print("zero_init", temp_image.shape)
                    #temp_seg = np.zeros((temp_image.shape[1], temp_image.shape[2]))
                    temp_seg = self.mean_seg.copy()
                #print("test", temp_seg.shape)
                temp_seg = np.expand_dims(temp_seg,0)
                temp_image = np.concatenate((temp_image,temp_seg),axis=0)
            elif self.seg_only:
                if os.path.isfile(self.np_dir+"../seg_mask_real_np/"+trial+'/'+seg_img+'.npy'):
                    temp_seg = np.load(self.np_dir+"../seg_mask_real_np/"+trial+'/'+seg_img+'.npy')
                else:
                    print("zero_init for seg only!!!!!", seg_img, trial)
                    #temp_seg = np.zeros((temp_image.shape[1], temp_image.shape[2]))
                    temp_seg = self.mean_seg.copy()
                #print("test", temp_seg.shape)
                temp_seg = np.expand_dims(temp_seg,0)
                temp_image = np.concatenate((temp_image,temp_seg),axis=0)
            #print("with seg", temp_image.shape)
            #if self.image_transform:
            #    temp_image = self.image_transform(temp_image)
            if image is None:
                image = temp_image
            else:
                image = np.concatenate((image,temp_image),axis=0)#NOTE: axis might be wrong.  doublecheck
            #print(image.shape)
        image = image.astype('float32')
        #print("fin img", image.shape)

        if self.custom_dataset is None:
            if self.seg_only:
                points = np.array(self.traj_list[trial_idx][int(seg_img)])
            else:
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
            else:
                points = np.array(points)

            #if self.image_transform:
            #    data = {}
            #    data["img"] = image
            #    data["pts"] = points
            #    image, points = self.image_transform(data)

            return {'image':image, 'points':points, "time_frac": time_frac}

        elif self.custom_dataset == "no_parse":
            point = np.array(self.traj_list[trial_idx][idx])
            p0 = np.array(point[0:3])
            R0 = R.from_euler('ZYX', point[3:6]).as_dcm()
            points = []
            h = (float(self.nEvents[trial_idx])/(self.time_secs[trial_idx]*self.time_multiplier[trial_idx]))
            indices = np.floor(np.add(np.array([1*self.pred_dt, 2*self.pred_dt, 3*self.pred_dt]) * h, idx)).astype(int)

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
                Ri = R.from_euler('ZYX', pt[3:6]).as_dcm()
                point_list.append(p)
                rot_list.append(Ri)

            flipped = False
            if self.image_transform:
                data = {}
                data["img"] = image
                data["pts"] = point_list
                data["rots"] = rot_list
                image, point_list, rot_list, flipped = self.image_transform(data)

            p0 = np.array(point_list[0])#seems unnecessary
            R0 = np.array(rot_list[0])

            for ii in range(1,len(point_list)):
                if (self.relative_pose):
                  prev_p = np.array(point_list[ii-1])
                  prev_R = np.array(rot_list[ii-1])
                else:
                  prev_p = p0
                  prev_R = R0
                p = np.array(point_list[ii])
                Ri = np.array(rot_list[ii])
                p = list(np.matmul(prev_R.T,p-prev_p))
                Ri = np.matmul(prev_R.T,Ri)
                Ri_zyx = list(R.from_dcm(Ri).as_euler('ZYX'))
                p.extend(Ri_zyx)
                points.append(p)

            #points = np.matmul(np.array(R0).T, (np.array(points) - np.array(p0)).T).T

            if self.gaussian_pts:
                #print("Gaussian Noise added")
                for pt_num in range(len(points)):
                    for dof in range(len(points[pt_num])):
                        err = np.min((np.max((-np.random.normal(0.0, self.gaussian_var[dof]), -self.gaussian_limit[dof])), self.gaussian_limit[dof]))
                        #print(err)
                        points[pt_num][dof] += err

            if self.point_transform:
                points = self.point_transform(points)
            else:
                points = np.array(points)

            #if self.image_transform:
            #    data = {}
            #    data["img"] = image
            #    data["pts"] = points
            #    image, points = self.image_transform(data)

            return {'image':image, 'points':points, "flipped": flipped, "time_frac": time_frac}

