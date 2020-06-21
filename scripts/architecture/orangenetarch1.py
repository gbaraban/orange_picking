import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

class Block8(torch.nn.Module):
    def __init__(self, in_f, out_f, stride = 1, downsample=None):
        super(Block8,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_f,out_channels=out_f,kernel_size=3,stride=1,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = stride)
        self.conv2 = nn.Conv2d(in_channels=out_f,out_channels=out_f,kernel_size=3,stride=1,padding=1)
        #self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = stride)
        self.bn1 = nn.BatchNorm2d(out_f)
        self.bn2 = nn.BatchNorm2d(out_f)
        self.downsample = downsample
        #self.skip1 = nn.Conv2d(in_channels=in_f,out_channels=out_f,kernel_size=1,stride=4,padding=0)

    def forward(self,x):
        #print('x size: ',x.size())
        iden = x
        y = self.conv1(x)
        y = self.maxpool1(y)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        #y = self.maxpool2(y)
        y = self.bn2(y)
        if self.downsample is not None:
            iden = self.downsample(x)
        #print(y.size(),x.size())
        y += iden
        y = F.relu(y)
        return y

class Block18(torch.nn.Module):
    def __init__(self, in_f, out_f, stride = 1, downsample=None):
        super(Block18,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_f,out_channels=out_f,kernel_size=3,stride=stride,padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_f,out_channels=out_f,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_f)
        self.bn2 = nn.BatchNorm2d(out_f)
        self.downsample = downsample


    def forward(self,x):
        iden = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            iden = self.downsample(x)
        out += iden
        out = F.relu(out)
        return out

def make_layer18(in_size,out_size,stride_length=1):
    blocks = []
    downsample = None
    if in_size != out_size or stride_length != 1:
        downsample = nn.Sequential(
                nn.Conv2d(in_size,out_size,kernel_size=1,stride=stride_length),
                nn.BatchNorm2d(out_size))
    return nn.Sequential(
            Block18(in_size,out_size,stride_length, downsample),
            Block18(out_size,out_size))

def make_layer8(in_size,out_size,stride_length=1):
    blocks = []
    downsample = None
    if in_size != out_size or stride_length != 1:
        downsample = nn.Conv2d(in_size,out_size,kernel_size=1,stride=stride_length)
    return nn.Sequential(
            Block8(in_size,out_size,stride_length, downsample))#,
            #Block8(out_size,out_size))

class OrangeNet8(torch.nn.Module):
    def __init__(self, capacity = 1, num_img = 1, num_pts = 3, bins = 30, mins = None, maxs = None, n_outputs = 3):
        super(OrangeNet8, self).__init__()
        #Parameters
        self.w = 640 #300
        self.h = 380 #200
        self.num_points = num_pts
        self.num_images = num_img
        self.f = capacity#5.0#2.0#1.5#125#1#0.25
        #self.learning_fac_init=0.000001
        #self.reg = False
        self.bins = bins
        self.min = mins
        self.max = maxs
        #Blocks
        #TODO: Add input layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=int(32*self.f),kernel_size=5,stride=2,padding = 10)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.block1 = make_layer8(int(32*self.f),int(32*self.f), stride_length=2)
        self.block2 = make_layer8(int(32*self.f),int(64*self.f), stride_length=2)
        self.block3 = make_layer8(int(64*self.f),int(128*self.f), stride_length=2)
        #in_size = 768*self.f#122880#temp
        in_size = 34944*self.f#122880#temp
        self.fc1 = nn.Linear(int(in_size),int(4096*self.f))
        self.fc2 = nn.Linear(int(4096*self.f),int(2048*self.f))
        self.fc3 = nn.Linear(int(2048*self.f),int(1024*self.f))
        #self.output = nn.Linear(int(1024*self.f),3*self.num_points*self.bins)
        self.output = nn.Linear(int(2048*self.f),n_outputs*self.num_points*self.bins)

    def forward(self,x):
        #x =  self.resnet(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x,1)
        x = F.relu(x)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        #x = self.fc3(x)
        #x = F.relu(x)
        x = self.output(x)
        return x

class OrangeNet18(torch.nn.Module):
    def __init__(self, capacity = 1, num_img = 1, num_pts = 3, bins = 30, mins = None, maxs = None, n_outputs = 3):
        super(OrangeNet18, self).__init__()
        #Parameters
        self.w = 640 #300
        self.h = 380 #200 
        self.num_points = num_pts
        self.num_images = num_img
        self.f = capacity#5.0#2.0#1.5#125#1#0.25
        #self.learning_fac_init=0.000001
        #self.reg = False
        self.bins = bins
        self.min = mins
        self.max = maxs
        #Blocks
        self.conv1 = nn.Conv2d(3,int(64*self.f),kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(int(64*self.f))
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = make_layer18(int(64*self.f),int(64*self.f))
        self.layer2 = make_layer18(int(64*self.f),int(128*self.f),2)
        self.layer3 = make_layer18(int(128*self.f),int(256*self.f),2)
        self.layer4 = make_layer18(int(256*self.f),int(512*self.f),2)
        #Change resnet output here
        #Batch normalization change?
        in_size = int(122880*self.f)#temp
        #in_size = int(2044672*self.f)#temp
        self.fc1 = nn.Linear(int(in_size),int(4096*self.f))
        self.fc2 = nn.Linear(int(4096*self.f),int(2048*self.f))
        self.fc3 = nn.Linear(int(2048*self.f),int(1024*self.f))
        self.output = nn.Linear(int(1024*self.f),n_outputs*self.num_points*self.bins)
    
    def forward(self,x):
        #x =  self.resnet(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.resnet.avgpool(x)
        #print('PreFlat',x.shape)
        x = torch.flatten(x,1)
        #print('Flat',x.shape)
        #x = self.resnet.fc(x)
        x = F.relu(x)#Possible work here
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.output(x)
        return x
