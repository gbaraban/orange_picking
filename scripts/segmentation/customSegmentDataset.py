import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
import PIL.Image as img
import sys
from torch.utils.data import Dataset, DataLoader
import pickle


class SubSet(Dataset):
    def __init__(self,dataset,idx):
        self.ds = dataset
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self,i):
        return self.ds[self.idx[i]]


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, seg_dir, channels=3, h=480, w=640):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.img_dir_np = self.img_dir.rstrip("/") + "_np/"
        self.seg_dir_np = self.seg_dir.rstrip("/") + "_np/"
        self.h = h
        self.w = w
        self.channels = channels

        self.trials = os.listdir(self.seg_dir_np)

        self.mean_seg = np.load("data/mean_imgv2_data_seg_real_world_traj_bag.npy")
        self.img_loc = []

        for trial in self.trials:
            trial_loc = self.img_dir_np + "/" + trial
            for img_name in os.listdir(trial_loc):
                if img_name.startswith("image") and img_name.endswith("npy"):
                    self.img_loc.append({"loc": trial, "name": img_name})

        print("Dataset Init Done, Num data points: ", len(self.img_loc)) 


    def __len__(self):
        return len(self.img_loc)

    def __getitem__(self, i):
        in_img = np.load(self.img_dir_np + "/" + self.img_loc[i]["loc"] + "/" + self.img_loc[i]["name"]).astype(np.float32)
        in_img = np.transpose(in_img, [2,0,1])

        img_num = self.img_loc[i]["name"].lstrip("image")

        if os.path.isfile(self.seg_dir + "/" + self.img_loc[i]["loc"] + "/" + img_num):
            out_img = np.load(self.seg_dir + "/" + self.img_loc[i]["loc"] + "/" + img_num).astype(np.uint8)
        else:
            out_img = np.zeros((self.h, self.w), dtype=np.uint8)

        #print(out_img, np.max(out_img))

        return {"image": in_img, "segmented": out_img}
