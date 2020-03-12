import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

class OrangeNet(torch.nn.Module):
    def __init__(self, capacity = 1, num_img = 1, num_pts = 3, bins = 30):
        super(OrangeNet, self).__init__()
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
        self.fc1 = nn.Linear(4096*self.f)
        self.fc2 = nn.Linear(2048*self.f)
        self.fc3 = nn.Linear(1024*self.f)
        self.output = nn.Linear(3*self.num_points*self.bins)
    
    def forward(self,x):
        #Change to using the sublayers of resnet??
        x = self.resnet(x)
        x = F.relu(x)#Possible work here
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.output(x)

    def train(self):
        self.resnet.train()

    def eval(self):
        self.resnet.eval()
