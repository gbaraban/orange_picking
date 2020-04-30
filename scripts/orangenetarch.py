import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

class Block8(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Block8,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_f,out_channels=out_f,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_f,out_channels=out_f,kernel_size=3,stride=2,padding=1)
        self.skip1 = nn.Conv2d(in_channels=in_f,out_channels=out_f,kernel_size=1,stride=4,padding=0)

    def forward(self,x):
        #print('x size: ',x.size())
        y = F.relu(x)
        y = self.conv1(y)
        y = F.relu(y)
        y = self.conv2(y)
        x = self.skip1(x)
        #print(y.size(),x.size())
        return y + x


class OrangeNet8(torch.nn.Module):
    def __init__(self, capacity = 1, num_img = 1, num_pts = 3, bins = 30):
        super(OrangeNet8, self).__init__()
        #Parameters
        self.w = 640 #300
        self.h = 380 #200 
        self.num_points = num_pts
        self.num_images = num_img
        self.f = capacity#5.0#2.0#1.5#125#1#0.25
        self.learning_fac_init=0.000001
        #self.reg = False
        self.bins = bins
        #Blocks
        #TODO: Add input layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32*self.f,kernel_size=5,stride=2,padding = 10)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.block1 = Block8(32*self.f,32*self.f)
        self.block2 = Block8(32*self.f,64*self.f)
        self.block3 = Block8(64*self.f,128*self.f)
        in_size = 768#122880#temp
        self.fc1 = nn.Linear(in_size,4096*self.f)
        self.fc2 = nn.Linear(4096*self.f,2048*self.f)
        self.fc3 = nn.Linear(2048*self.f,1024*self.f)
        self.output = nn.Linear(1024*self.f,3*self.num_points*self.bins)
    
    def forward(self,x):
        #x =  self.resnet(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x,1)
        x = F.relu(x)#Possible work here
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.output(x)
        return x

class OrangeNet18(torch.nn.Module):
    def __init__(self, capacity = 1, num_img = 1, num_pts = 3, bins = 30):
        super(OrangeNet18, self).__init__()
        #Parameters
        self.w = 640 #300
        self.h = 380 #200 
        self.num_points = num_pts
        self.num_images = num_img
        self.f = capacity#5.0#2.0#1.5#125#1#0.25
        self.learning_fac_init=0.000001
        #self.reg = False
        self.bins = bins
        #Blocks
        #TODO: Add input layer
        self.resnet = models.resnet18(pretrained=False)
        #Change resnet output here
        #Batch normalization change?
        in_size = 122880#temp
        self.fc1 = nn.Linear(in_size,4096*self.f)
        self.fc2 = nn.Linear(4096*self.f,2048*self.f)
        self.fc3 = nn.Linear(2048*self.f,1024*self.f)
        self.output = nn.Linear(1024*self.f,3*self.num_points*self.bins)
    
    def forward(self,x):
        #x =  self.resnet(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        #x = self.resnet.avgpool(x)
        x = torch.flatten(x,1)
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
