import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
#from torchvision import datasets, models, transforms
import numpy as np

class SegmentationNet(torch.nn.Module):
    def __init__(self, capacity = 1, retrain_off_seg = False):
        super(SegmentationNet,self).__init__()
        self.w = 640
        self.h = 480
        self.f = capacity
        self.n_classes = 2

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=10, stride=4, padding=5)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()

        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=128, out_channels=self.n_classes, kernel_size=10, stride=4, padding=5)
        #print(self.conv_transpose2)
        self.conv_bridge = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.conv_bridge2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)

        self.output = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1)
        if retrain_off_seg:
            print("Seg layers frozen")
            self.conv1.weight.requires_grad = False
            self.conv1.bias.requires_grad = False
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            #self.relu1.weight.requires_grad = False
            #self.relu1.bias.requires_grad = False

            self.conv2.weight.requires_grad = False
            self.conv2.bias.requires_grad = False
            self.bn2.weight.requires_grad = False
            self.bn2.bias.requires_grad = False
            #self.relu2.weight.requires_grad = False
            #self.relu2.bias.requires_grad = False

            self.conv3.weight.requires_grad = False
            self.conv3.bias.requires_grad = False
            self.bn3.weight.requires_grad = False
            self.bn3.bias.requires_grad = False
            #self.relu3.weight.requires_grad = False
            #self.relu3.bias.requires_grad = False

            self.conv4.weight.requires_grad = False
            self.conv4.bias.requires_grad = False

            self.conv_transpose1.weight.requires_grad = False
            self.conv_transpose1.bias.requires_grad = False
            self.bn4.weight.requires_grad = False
            self.bn4.bias.requires_grad = False
            #self.relu4.weight.requires_grad = False
            #self.relu4.bias.requires_grad = False

            self.conv_transpose2.weight.requires_grad = False
            self.conv_transpose2.bias.requires_grad = False
            self.bn5.weight.requires_grad = False
            self.bn5.bias.requires_grad = False
            #self.relu5.weight.requires_grad = False
            #self.relu5.bias.requires_grad = False

            self.conv_transpose3.weight.requires_grad = False
            self.conv_transpose3.bias.requires_grad = False

            self.conv_bridge.weight.requires_grad = False
            self.conv_bridge.bias.requires_grad = False

            self.conv_bridge2.weight.requires_grad = False
            self.conv_bridge2.bias.requires_grad = False

            self.output.weight.requires_grad = False
            self.output.bias.requires_grad = False


    def forward(self,x):
        print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu1(x)
        #print(x1.shape)
        x2 = self.conv_bridge(x1)
        x = self.conv2(x1)
        x = self.bn2(x)
        x3 = self.relu2(x)
        x4 = self.conv_bridge2(x3)
        #print(x3.shape)
        x = self.conv3(x3)
        #print(x.shape)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        #print(x.shape)
        x = self.conv_transpose1(x)
        x += x4
        x = self.bn4(x)
        x = self.relu4(x)
        #print(x.shape)
        x = self.conv_transpose2(x)
        x += x2
        #print(x.shape)
        x = self.conv_transpose3(x)
        #print(x.shape)
        x = self.output(x)
        return x

if __name__ == "__main__":
    sgNet = SegmentationNet()
    img = np.load("/home/gabe/ws/ros_ws/src/orange_picking/data/depth_data/data/real_world_traj_bag_np/bag0/image0.npy").astype(np.float32)
    img = np.transpose(img, [2,0,1])
    img = img.reshape([1, 3, 480, 640])
    img = torch.tensor(img)
    sgNet.forward(img)
