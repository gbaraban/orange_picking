from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from datetime import datetime
import argparse
import signal
import sys
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from customTransforms import *
import pickle
import gc
import random
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
from numpy.linalg import norm


#Global variables
save_path = None
model = None
writer = None

def addTimestamp(input_path):
    return input_path + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def createStampedFolder(folder_path):
    stamped_dir = os.path.join(
            folder_path,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(stamped_dir)
        return stamped_dir
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise # This was not a "directory exist" error..

def save_model(i = 'Last'):
    global save_path
    global model
    if model is not None and save_path is not None:
        print('Saving...')
        name = save_path+'/model' + str(i) + '.pth.tar'
        torch.save(model.state_dict(),name)

def signal_handler(signal, frame):
    global writer
    print('')
    save_model()
    if writer is not None:
        print('Closing Writer...')
        writer.close()
    print('Done')
    sys.exit(0)

def dl_signal(signal,frame):
    #print("DL signal called")
    sys.exit(0)

def dl_init(x):
    signal.signal(signal.SIGINT,dl_signal)

def parseFiles(idx,num_list,run_dir,model,traj_data,real,dataclass,args):
  idx = idx.astype(int)
  trial_idx = 0

  while True:
    if num_list[trial_idx] <= idx:
      idx = idx - num_list[trial_idx]
      trial_idx += 1
    else:
      break

  trial_list = os.listdir(run_dir)
  trial = trial_list[trial_idx]

  #image = None
  #image_idx = idx
  #print(idx)
  #print(image_idx)
  #for ii in range(model.num_images):
  #  temp_idx = max(0,image_idx - ii)
  #  img_run_dir = run_dir.rstrip("/") + "_np"
  #  temp_image = np.load(img_run_dir+'/'+ trial + '/image' + str(temp_idx) + '.npy') #.resize((model.h,model.w))
  #  #temp_image = np.array(temp_image.getdata()).reshape(temp_image.size[0],temp_image.size[1],3)
  #  #temp_image = temp_image[:,:,0:3]/255.0 #Cut out alpha
  #  # temp_image = temp_image/255.0
  #  if image is None:
  #      image = temp_image
  #  else:
  #      image = np.concatenate((image,temp_image),axis=2)
  if False: #TODO bad fix
    points = traj_data[trial_idx][idx]
  else:
    #p0 = np.array(traj_data[trial_idx][idx][0])
    #R0 = np.array(traj_data[trial_idx][idx][1])
    points = []

    if args.real == 0:
        p0 = np.array(traj_data[trial_idx][idx][0])
        R0 = np.array(traj_data[trial_idx][idx][1])

        if dataclass.custom_dataset is None:
            for pt in range(dataclass.num_pts):
                temp_idx = int(idx + dataclass.h[trial]*dataclass.dt*(pt+1))
                p = traj_data[trial_idx][temp_idx][0]
                Ri = traj_data[trial_idx][temp_idx][1]
                p = np.array(p)
                p = list(np.matmul(R0.T,p-p0))
                Ri = np.matmul(R0.T,Ri)
                Ri_zyx = list(R.from_dcm(Ri).as_euler('ZYX'))
                p.extend(Ri_zyx)
                points.append(p)
        elif dataclass.custom_dataset == "Run18":
            indices = np.floor(np.add(np.array([1, 2, 3]) * dataclass.h[trial], idx)).astype(int)
            """
            for x, ii in enumerate(indices):
                if ii >= num_list[trial_idx]:
                    delta = idx if (x == 0) else indices[x-1]
                    dt = np.floor(((num_list[trial_idx] -1) - delta)/(3-x)).astype(int)
                    z = 3 - x
                    while (dt < 1) and z != 0:
                        dt = np.floor(((num_list[trial_idx] - 1) - delta)/z).astype(int)
                        z -= 1
                    for j in range(0,z):
                        indices[j+x] = delta + ((j+1)*dt)

                    delta = 3 - (z+x)

                    for j in range(0, delta):
                        indices[x+z+j] = num_list[trial_idx] - 1

                    break
            """
            for i in range(len(indices)):
                if (indices[i] >= num_list[trial_idx]):
                    if i == 0:
                        indices[i] = idx
                    else:
                        indices[i] = indices[i-1]

            for x, ii in enumerate(indices):
                if (ii < idx):
                        print(idx, ii)
                p = np.array(traj_data[trial_idx][ii][0])
                Ri = np.array(traj_data[trial_idx][ii][1])
                p = list(np.matmul(R0.T,p-p0))
                Ri = np.matmul(R0.T,Ri)
                Ri_zyx = list(R.from_dcm(Ri).as_euler('ZYX'))
                p.extend(Ri_zyx)
                points.append(p)

    else:
        if dataclass.custom_dataset is None:
            #print(trial_idx, idx, dataclass.num_list[trial_idx])
            points = np.array(dataclass.traj_list[trial_idx][idx])
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

            #if self.image_transform:
            #    data = {}
            #    data["img"] = image
            #    data["pts"] = points
            #    image, points = self.image_transform(data)

            #return {'image':image, 'points':points}

        elif dataclass.custom_dataset == "no_parse":
            point = np.array(dataclass.traj_list[trial_idx][idx])
            p0 = np.array(point[0:3])
            R0 = R.from_euler('ZYX', point[3:6]).as_dcm()
            points = []
            h = (float(dataclass.nEvents[trial_idx])/dataclass.time_secs[trial_idx])
            indices = np.floor(np.add(np.array([1, 2, 3]) * h, idx)).astype(int)

            for i in range(len(indices)):
                if (indices[i] >= dataclass.num_list[trial_idx]):
                    if i == 0:
                        indices[i] = idx
                    else:
                        indices[i] = indices[i-1]

            point_list = [p0]
            rot_list = [R0]

            for x, ii in enumerate(indices):
                if (ii < idx):
                        print(idx, ii)
                #print(ii, dataclass.num_list[trial_idx])
                pt = np.array(dataclass.traj_list[trial_idx][ii])
                p = np.array(pt[0:3])
                Ri = R.from_euler('ZYX', pt[3:6]).as_dcm()
                point_list.append(p)
                rot_list.append(Ri)

            #flipped = False
            #if self.image_transform:
            #    data = {}
            #    data["img"] = image
            #    data["pts"] = point_list
            #    data["rots"] = rot_list
            #    image, point_list, rot_list, flipped = self.image_transform(data)

            p0 = np.array(point_list[0])
            R0 = np.array(rot_list[0])

            for ii in range(1,len(point_list)):
                p = np.array(point_list[ii])
                Ri = np.array(rot_list[ii])
                #print(R0)
                #print(p0)
                #print(p)
                p = list(np.matmul(R0.T,p-p0))
                Ri = np.matmul(R0.T,Ri)
                Ri_zyx = list(R.from_dcm(Ri).as_euler('ZYX'))

                p.extend(Ri_zyx)
                points.append(p)

            #points = np.matmul(np.array(R0).T, (np.array(points) - np.array(p0)).T).T

  local_pts = []
  ctr = 0
  #print(points)
  #exit()
  #print(points)
  for point in points:
    #Convert into bins
    min_i = np.array(model.min[ctr])
    max_i = np.array(model.max[ctr])
    bin_nums = (point - min_i)/(max_i-min_i)
    bin_nums_scaled = (bin_nums*model.bins).astype(int)
    #for temp in range(3):
        #if (bin_nums_scaled[temp] > model.bins-1) or (bin_nums_scaled[temp] < 0):
            #print(str(point) + ' is out of bounds: ' + str(min_i) + ' ' + str(max_i))
    bin_nums = np.clip(bin_nums_scaled,a_min=0,a_max=model.bins-1)
    ctr += 1
    labels = np.zeros((6,model.bins))    
    mean = 1
    stdev = 1E-5
    for j in range(len(bin_nums)):
      for i in range(labels.shape[1]):
        labels[j][i] = mean * (np.exp((-np.power(bin_nums[j]-i, 2))/(2 * np.power(stdev, 2))))
    local_pts.append(labels)
  local_pts = np.array(local_pts)
  local_pts.resize((model.num_points,6,model.bins))
  return local_pts

def loadData(idx,num_list,run_dir,model,traj_data,real,dataclass,args):
  waypoints_x = []
  waypoints_y = []
  waypoints_z = []
  waypoints_r = []
  waypoints_p = []
  waypoints_yaw = []

  for ii in idx:
    waypoint = parseFiles(ii,num_list,run_dir,model,traj_data,real,dataclass,args)
    #images.append(np.array(image)) #- mean_image))
    waypoints_x.append(waypoint[:,0,:])
    waypoints_y.append(waypoint[:,1,:])
    waypoints_z.append(waypoint[:,2,:])
    waypoints_yaw.append(waypoint[:,3,:])
    waypoints_p.append(waypoint[:,4,:])
    waypoints_r.append(waypoint[:,5,:])

  waypoints_x = np.array(waypoints_x)
  waypoints_y = np.array(waypoints_y)
  waypoints_z = np.array(waypoints_z)
  waypoints_yaw = np.array(waypoints_yaw)
  waypoints_p = np.array(waypoints_p)
  waypoints_r = np.array(waypoints_r)

  return np.array(waypoints_x).reshape(-1,model.num_points,model.bins), np.array(waypoints_y).reshape(-1,model.num_points,model.bins), np.array(waypoints_z).reshape(-1,model.num_points,model.bins), np.array(waypoints_yaw).reshape(-1,model.num_points,model.bins), np.array(waypoints_p).reshape(-1,model.num_points,model.bins), np.array(waypoints_r).reshape(-1,model.num_points,model.bins)

def skew2axis(mat):
	return [-mat[1,2], mat[0,2], -mat[0,1]]

def angle_dist(true_angle, pred_angle):
	Rot0 = R.from_euler('ZYX', [true_angle[0], true_angle[1], true_angle[2]]) #[np.pi/3, np.pi/6, np.pi/4])
	R0 = Rot0.as_dcm()

	Rot1 = R.from_euler('ZYX', [pred_angle[0], pred_angle[1], pred_angle[2]]) #[np.pi/3 + 1, np.pi/6, np.pi/4])
	R1 = Rot1.as_dcm()

	R2 = np.matmul(R0.T,R1)
	logR2 = logm(R2)
	dist = norm(skew2axis(logR2))

	return dist

def acc_metric(args,logits,point_batch,yaw_only,phase,ranges,sphere_flag = False):
    #TODO: Add diff accuracy for angles
    softmax = nn.Softmax(dim=0)
    shape = logits.size()#.item()
    acc_list = []
    extra_acc_list = [[] for i in range(args.num_controls-1)]
    logits = logits.detach()
    for pt in range(shape[2]):
        batch_list = []
        extra_batch_list = [[] for i in range(args.num_controls-1)]
        ang_d = []
        extra_ang_d = [[] for i in range(args.num_controls-1)]
        for ii in range(shape[0]):
            coord_list = []
            if phase[ii] == 0:
                bin_min = args.min
                bin_max = args.max
                n_bins = args.bins
            else:
                bin_min = args.extra_mins[phase[ii]-1]
                bin_max = args.extra_maxs[phase[ii]-1]
                n_bins = args.bins
                                
            if sphere_flag:
                r_theta_phi = [0.,0.,0.,0.,0.,0.]
                r_theta_phi_true = [0.,0.,0.,0.,0.,0.]
                for coord in range(3):
                    prob = np.array(softmax(logits[ii,coord,pt,ranges[phase[ii]]]))#.numpy()
                    idx = np.argmax(prob)
                    true_idx = np.array(point_batch[ii,pt,coord])
                    bin_size = (bin_max[pt][coord] - bin_min[pt][coord])/float(n_bins)
                    r_theta_phi[coord] = idx*bin_size + bin_min[pt][coord]
                    r_theta_phi_true[coord] = true_idx*bin_size + bin_min[pt][coord]
                (xyz,xyz_true) = sphericalToXYZ()([r_theta_phi,r_theta_phi_true])
                d = (xyz - xyz_true)
                if phase[ii] == 0:
                    batch_list.append(np.linalg.norm(d))
                else:
                    extra_batch_list[phase[ii]-1].append(np.linalg.norm(d))

            else:
                for coord in range(3):
                    prob = np.array(softmax(logits[ii,coord,pt,ranges[phase[ii]]]))#.numpy()
                    max_pred = np.argmax(prob)
                    #true_pred = np.argmax(point_batch[ii,pt,coord,:])
                    true_pred = np.array(point_batch[ii,pt,coord])
                    bin_size = (bin_max[pt][coord] - bin_min[pt][coord])/float(n_bins)
                    d = (true_pred - max_pred)*bin_size
                    coord_list.append(d)
                d = np.vstack([coord_list[0],coord_list[1],coord_list[2]])
                if phase[ii] == 0:
                    batch_list.append(np.linalg.norm(d))
                else:
                    extra_batch_list[phase[ii]-1].append(np.linalg.norm(d))                    

            if not yaw_only:
                true_angle = []
                pred_angle = []
                for coord in range(3,6):
                    prob = np.array(softmax(logits[ii,coord,pt,ranges[phase[ii]]]))#.numpy()
                    max_pred = np.argmax(prob)
                    #true_pred = np.argmax(point_batch[ii,pt,coord,:])
                    true_pred = np.array(point_batch[ii,pt,coord])
                    bin_size = (bin_max[pt][coord] - bin_min[pt][coord])/float(n_bins)
                    true_angle.append(true_pred * bin_size)
                    pred_angle.append(max_pred * bin_size)
                if phase[ii] == 0:
                    ang_d.append(angle_dist(true_angle, pred_angle))
                else:
                    extra_ang_d[phase[ii]-1].append(angle_dist(true_angle, pred_angle))

            else:
                prob = np.array(softmax(logits[ii,3,pt,ranges[phase[ii]]]))#.numpy()
                max_pred = np.argmax(prob)
                #true_pred = np.argmax(point_batch[ii,pt,coord,:])
                true_pred = np.array(point_batch[ii,pt,3])
                bin_size = (bin_max[pt][coord] - bin_min[pt][coord])/float(n_bins)
                true_angle = (true_pred * bin_size)
                pred_angle = (max_pred * bin_size)

                if phase[ii] == 0:
                    ang_d.append(norm((true_angle - pred_angle)))
                else:
                    extra_ang_d[phase[ii]-1].append(norm((true_angle - pred_angle)))

        if len(batch_list) != 0:
            batch_mean = np.mean(batch_list)
            batch_mean = [batch_mean, np.mean(ang_d)]
        else:
            batch_mean = [0., 0.]
        acc_list.append(batch_mean)
        for i in range(len(extra_batch_list)):
            if len(extra_batch_list[i]) != 0:
                temp_batch_mean = np.mean(extra_batch_list[i])
                temp_batch_mean = [temp_batch_mean, np.mean(extra_ang_d[i])]
            else:
                temp_batch_mean = [0., 0.]
            extra_acc_list[i].append(temp_batch_mean)

    if args.num_controls > 1:
        return acc_list, extra_acc_list
    return acc_list

def acc_metric_regression(args, logits, point_batch, sphere_flag = False):
    acc_list = []
    shape = logits.size()
    logits = logits.detach()
    if sphere_flag:
        logits = sphericalToXYZ()(logits)
    for pt in range(shape[1]):
        batch_list = []
        ang_d = 0
        for ii in range(shape[0]):
            coord_list = []
            for coord in range(3):
                pred = logits[ii,pt,coord]
                d = pred - point_batch[ii, pt, coord]
                coord_list.append(d)
            d = np.vstack([coord_list[0],coord_list[1],coord_list[2]])
            batch_list.append(np.linalg.norm(d))

            true_angle = []
            pred_angle = []
            for coord in range(3,6):
                pred = logits[ii,pt,coord]
                true_angle.append(point_batch[ii, pt, coord])
                pred_angle.append(pred)
            ang_d = angle_dist(true_angle, pred_angle)
        
        batch_mean = np.mean(batch_list)
        batch_mean = [batch_mean, ang_d]
        acc_list.append(batch_mean)
    return acc_list

def get_state_data(state_config, data):
    if state_config == 1:
        # print("State 1")
        orange_pose_data = data["orange_pose"]
        # print("RP State")
        rp_data = data["rp"]
        states_data = torch.cat((orange_pose_data, rp_data), 1)
        return states_data

    elif state_config == 2:
        body_v_data = data["body_v"]
        orange_pose_data = data["orange_pose"]
        rp_data = data["rp"]
        states_data = torch.cat((body_v_data, orange_pose_data, rp_data), 1)
        return states_data
    else:
        return None

def phase_accuracy(actual, predicted):
    softmax = nn.Softmax(dim=1)
    actual_phases = np.array(actual)
    predicted_phases = softmax(predicted).to('cpu').detach().numpy()
    max_pred = np.argmax(predicted_phases, axis=1)
    accuracy = np.sum((max_pred == actual_phases).astype(np.float64))/actual_phases.shape[0]
    return accuracy

def main():
    signal.signal(signal.SIGINT,signal_handler)
    global model
    global save_path
    global writer
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data folder')
    parser.add_argument('--load', help='model to load')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--resample', action='store_true', help='resample data')
    parser.add_argument('--gpu', help='gpu to use')
    parser.add_argument('--num_images', type=int, default=1, help='number of input images')
    parser.add_argument('--batch_size', type=int, default=65, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=65, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='batch size')
    parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
    parser.add_argument('--capacity', type=float, default=1, help='network capacity')
    parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
    parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
    parser.add_argument('--bins', type=int, default=30, help='number of bins per coordinate')
    parser.add_argument('-j', type=int, default=8, help='number of loader workers')
    parser.add_argument('--traj', type=int, default=1, help='train trajectories')
    parser.add_argument('--real', type=int, default=0, help='real world imgs (0: sim data, 1: orange tracking data, else: normal real world data')
    parser.add_argument('--val', type=float, default=0.10, help='validation percentage')
    parser.add_argument('--resnet18', type=int, default=0, help='ResNet18')
    parser.add_argument('--yaw_only', type=int, default=0, help='yaw only')
    parser.add_argument('--custom', type=str, default="", help='custom parser: Run18/no_parse')
    parser.add_argument('--test_arch', type=int, default=100, help='testing architectures')
    parser.add_argument('--train', type=int, default=1, help='train or test')
    parser.add_argument('--data_aug_flip',type=int,default=1, help='Horizontal flipping on/off')
    parser.add_argument('--real_test',type=int,default=0,help='self.h = 380/480')
    parser.add_argument('--freeze',type=str,default="",help='layer to freeze while training, linear or conv')
    parser.add_argument('--plot_data',type=int,default=0,help="plot data for accuracy")
    parser.add_argument('--input_size',type=float,default=1,help='input size change')
    parser.add_argument('--use_sampler',type=bool,default=False,help='use adaptive resampling (EXPERIMENTAL)')
    parser.add_argument('--use_error_sampler',type=bool,default=False,help='adaptive error based sampling')
    parser.add_argument('--custom_loss',type=bool,default=False,help='custom loss for training')
    parser.add_argument('--depth',type=bool,default=False,help='use depth channel')
    parser.add_argument('--seg',type=bool,default=False,help='use segmentation channel')
    parser.add_argument('--temp_seg',type=bool,default=False,help='use segmentation channel')
    parser.add_argument('--seg_only',type=bool,default=False,help='use segmentation channel')
    parser.add_argument('--save_variables',type=int,default=20,help="save after every x epochs")
    parser.add_argument('--mean_seg',type=str,default='data/depth_data/data/mean_seg.npy',help='mean segmentation image') # data/depth_data/data/mean_seg.npy #data/mean_imgv2_data_seg_Run24.npy
    parser.add_argument('--segload', help='segment model to load')
    parser.add_argument('--retrain_off_seg',type=bool,default=False,help='retrain segmentation off')
    parser.add_argument('--relative_pose', type=bool, default=False,help='relative pose of points')
    parser.add_argument('--regression', type=bool, default=False,help='Use regression loss (MSE)')
    parser.add_argument('--spherical', type=bool, default=False,help='Use spherical coordinates')
    parser.add_argument('--pred_dt',type=float, default=1.0, help="Space between predicted points")
    parser.add_argument('--image_dt',type=float,default=1.0, help="Space between images provided to network")
    parser.add_argument('--states',type=int,default=0,help="states to use with network")
    parser.add_argument('--num_controls',type=int,default=1,help="states to use with network")


    args = parser.parse_args()

    if args.custom == "":
        args.custom = None

    if args.plot_data == 0:
        args.plot_data = False
    else:
        args.plot_data = True

    if args.yaw_only == 0:
        args.yaw_only = False
    else:
        args.yaw_only = True

    if args.traj == 0:
        args.traj = False
    else:
        args.traj = True

    if args.real == 0:
        from customDatasetsOrientation import OrangeSimDataSet, SubSet
        #args.real = False
    elif args.real == 1:
        from customDatasetv1 import OrangeSimDataSet, SubSet
    else:
        from customRealDatasetsOrientation import OrangeSimDataSet, SubSet
        #args.real = True

    if args.test_arch == 100:
        if args.num_controls > 1:
            from orangenetarchmulticontrol import OrangeNet8, OrangeNet18
        elif args.states != 0:
            from orangenetarchstates import OrangeNet8, OrangeNet18
        elif args.states == 0 and args.num_controls == 1:
            from orangenetarch import OrangeNet8, OrangeNet18

    else:
        import importlib
        i = importlib.import_module('architecture.orangenetarch' + str(args.test_arch))
        OrangeNet8 = i.OrangeNet8
        OrangeNet18 = i.OrangeNet18

    if args.temp_seg:
        from segmentation.segmentnetarch import SegmentationNet
        segmodel = SegmentationNet(retrain_off_seg=args.retrain_off_seg)

    if args.real_test == 0:
        args.real_test = False
    else:
        args.real_test = True

    if args.relative_pose:
      #Change these
      print("Relative Pose")
      if args.spherical:
          args.min = [(0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2), (0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2), (0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2)]
          args.max = [(0.7, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2), (0.7, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2), (0.7, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2)]
      else:
          if args.real == 0:
              args.min = [(-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -0.5, -0.25, -0.5, -0.1, -0.1)]
              args.max = [(1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.0, 0.5, 0.25, 0.5, 0.1, 0.1)]
          elif args.real == 1:
              #args.min = [(-0.25, -0.5, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.5, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.5, -0.25, -np.pi/2, -0.1, -0.1)]
              #args.max = [(0.75, 0.75, 0.25, np.pi/2, 0.1, 0.1), (0.75, 0.75, 0.25, np.pi/2, 0.1, 0.1), (0.75, 0.75, 0.25, np.pi/2, 0.1, 0.1)]
              #args.min = [(-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1)]
              #args.max = [(0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1), (0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1), (0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1)]
              args.min = [(-0.10, -0.15, -0.10, -0.25, -0.075, -0.075), (-0.10, -0.15, -0.10, -0.25, -0.075, -0.075), (-0.10, -0.15, -0.10, -0.25, -0.075, -0.075)]
              args.max = [(0.30, 0.15, 0.10, 0.25, 0.075, 0.075), (0.30, 0.15, 0.10, 0.25, 0.075, 0.075), (0.30, 0.15, 0.10, 0.25, 0.075, 0.075)]

          else:
              #args.min = [(0.,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0.,-1.,-0.15,-np.pi,-np.pi/2,-np.pi),(0.,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0.,-2.,-0.3,-np.pi,-np.pi/2,-np.pi),(0.,-3.,-0.5,-np.pi,-np.pi/2,-np.pi)]
              args.min = [(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi)]
              args.max = [(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.,0.5,0.2,np.pi,np.pi,np.pi)]
    else:
      #args.min = [(0.,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0.,-1.,-0.15,-np.pi,-np.pi/2,-np.pi),(0.,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0.,-2.,-0.3,-np.pi,-np.pi/2,-np.pi),(0.,-3.,-0.5,-np.pi,-np.pi/2,-np.pi)]
      #args.max = [(1.,0.5,0.1,np.pi,np.pi/2,np.pi),(2.,1.,0.15,np.pi,np.pi/2,np.pi),(4.,1.5,0.2,np.pi,np.pi/2,np.pi),(6.,2.,0.3,np.pi,np.pi/2,np.pi),(7.,0.3,0.5,np.pi,np.pi/2,np.pi)]
      if args.spherical:
          args.min = [(0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2), (0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2), (0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2)]
          args.max = [(0.7, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2), (1.2, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2), (1.7, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2)]
      else:
          if args.real == 0:
              args.min = [(-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -1.0, -0.4, -np.pi/4, -0.1, -0.1), (-0.1, -1.25, -0.6, -np.pi/2, -0.1, -0.1)]
              args.max = [(1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.75, 1.0, 0.4, np.pi/4, 0.1, 0.1), (2.5, 1.25, 0.6, np.pi/2, 0.1, 0.1)]
          elif args.real == 1:
              args.min = [(-0.1, -0.4, -0.1, -np.pi/2, -0.1, -0.1), (-0.2, -0.8, -.15, -np.pi/2, -0.1, -0.1), (-0.3, -1.2, -0.25, -np.pi/2, -0.1, -0.1)]
              args.max = [(0.5, 0.5, 0.2, np.pi/2, 0.1, 0.1), (1.0, 0.8, 0.4, np.pi/2, 0.1, 0.1), (1.5, 1.2, 0.55, np.pi/2, 0.1, 0.1)]
          else:
              args.min = [(-0.5,-0.5,-0.2,-np.pi,-np.pi,-np.pi),(-0.75,-0.75,-0.5,-np.pi,-np.pi,-np.pi),(-1.0,-1.0,-0.75,-np.pi,-np.pi,-np.pi)]
              args.max = [(1.,0.5,0.2,np.pi,np.pi,np.pi),(1.5,1.0,0.5,np.pi,np.pi,np.pi),(2.0,1.0,0.75,np.pi,np.pi,np.pi)]

    if args.num_controls > 1:
        args.extra_mins = [[(-0.05, -0.05, -0.075, -0.10, -0.03, -0.03), (-0.05, -0.05, -0.075, -0.10, -0.03, -0.03), (-0.05, -0.05, -0.075, -0.10, -0.03, -0.03)]]
        args.extra_maxs = [[(0.15, 0.05, 0.075, 0.10, 0.03, 0.03), (0.15, 0.05, 0.075, 0.10, 0.03, 0.03), (0.15, 0.05, 0.075, 0.10, 0.03, 0.03)]]
    else:
        args.extra_mins = None
        args.extra_maxs = None

    #args.traj = False
    #Data Transforms
    pt_trans = []
    if not args.regression:
        if args.spherical:
            pt_trans.append(xyzToSpherical())
        pt_trans.append(pointToBins(args.min,args.max,args.bins,extra_min=args.extra_mins,extra_max=args.extra_maxs))
    if len(pt_trans) > 0:
        pt_trans = transforms.Compose(pt_trans)
    else:
        pt_trans = None

    #TODO ,GaussLabels(1,1e-10,args.bins)])

    if args.train == 1 and args.data_aug_flip == 1:
        print("Image transform set")
        img_trans = transforms.Compose([RandomHorizontalTrajFlip(p=0.5)])
    else:
        img_trans = None

    sampler_n_epochs = 2
    #Load Mean image
    print("Test")
    data_loc = copy.deepcopy(args.data)
    data_loc_name = data_loc.strip("..").strip(".").strip("/").replace("/", "_")
    mean_img_loc = data_loc + "/../mean_imgv2_" + data_loc_name + '.npy'
    if not (os.path.exists(mean_img_loc)):
        print('mean image file not found', mean_img_loc)
        return 0
        #mean_image = compute_mean_image(train_indices, data_loc, model)#TODO:Add this in
        #np.save(mean_img_loc, mean_image)
    else:
        print('mean image file found')
        mean_image = np.load(mean_img_loc)
    #mean_image = np.zeros((model.w, model.h, 3))
    #img_trans = None 
    #Create dataset class
    seg_mean_image = torch.tensor(np.load(args.mean_seg))

    #TOCHECK
    if args.real == 1:
        dataclass = OrangeSimDataSet(args.data, args.num_images, args.num_pts, pt_trans, img_trans, depth=args.depth, rel_pose=args.relative_pose, pred_dt = args.pred_dt, img_dt=args.image_dt)
    else:
        dataclass = OrangeSimDataSet(args.data, args.num_images, args.num_pts, pt_trans, img_trans, custom_dataset=args.custom, input=args.input_size, depth=args.depth, seg=args.seg, temp_seg=args.temp_seg, seg_only=args.seg_only, rel_pose=args.relative_pose, pred_dt = args.pred_dt, dt=args.image_dt)

    #Break up into validation and training
    #val_perc = 0.07
    val_perc = args.val
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not args.traj:
        print("No traj")
        if args.resample:
            rand_idx = np.random.choice(len(dataclass),size=len(dataclass),replace=True)
        else:
            rand_idx = np.random.permutation(len(dataclass))
        val_idx = np.ceil(len(dataclass)*val_perc).astype(int)
        train_idx = len(dataclass) - val_idx
        val_idx = rand_idx[-val_idx:]
        train_idx = rand_idx[:train_idx]

    else:
        print("Traj")
        val_order = np.ceil(len(dataclass.num_samples_dir)*val_perc).astype(int)
        val_indices = []
        train_indices = []
        val_data = {}
        val_data["order"] = np.array(random.sample(list(dataclass.num_samples_dir.keys()), k=val_order))
        #print(val_data["order"])

        for x in val_data["order"]:
            val_indices.extend(list(range(dataclass.num_samples_dir[x]['start'], dataclass.num_samples_dir[x]['end'])))
            val_data[x] = dataclass.num_samples_dir[x]

        for i, x in enumerate(list(dataclass.num_samples_dir.keys())):
            if x not in val_data["order"]:
                train_indices.extend(list(range(dataclass.num_samples_dir[x]['start'], dataclass.num_samples_dir[x]['end'])))

        val_idx = len(val_indices)
        train_idx = len(dataclass) - val_idx

        random.shuffle(train_indices)

        val_idx = np.array(val_indices)
        train_idx = np.array(train_indices)
        
        #fopen = open('val_data_xyz.pickle', 'wb')
        #pickle.dump(val_data, fopen, pickle.HIGHEST_PROTOCOL)

    train_data = SubSet(dataclass,train_idx)
    #print(np.min(train_idx), np.max(train_idx))
    error_weights = [0.0 for i in range(len(train_data))]
    val_data = SubSet(dataclass,val_idx)

    train_loc = {}
    for i, id in enumerate(train_idx):
        train_loc[id] = i
    #hist1 = np.histogram(train_idx, 10, range=(0, 5000))
    #print(hist1)
    #hist2 = np.histogram(val_idx, 10, range=(0,5000))
    #print(hist2)

    #Create DataLoaders
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.j, worker_init_fn=dl_init)
    weight_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.j, worker_init_fn=dl_init)
    val_loader = DataLoader(val_data,batch_size=args.val_batch_size,shuffle=False,num_workers=args.j)
    print ('Training Samples: ' + str(len(train_data)))
    print ('Validation Samples: ' + str(len(val_data)))

    #Create Model
    if args.yaw_only:
        n_outputs = 4
    else:
        n_outputs = 6

    base_n_channels = 3
    n_channels = base_n_channels

    if args.depth == 0:
        args.depth = False
    else:
        args.depth = True

    if args.depth:
        n_channels += 1
    if args.seg or args.seg_only or args.temp_seg:
        if args.seg and not args.real:
            n_channels += 2
        else:
            n_channels += 1

    state_count = 0

    if not args.resnet18:
        if args.states == 0 and args.num_controls == 1:
            model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=args.real_test,retrain_off=args.freeze,input=args.input_size, num_channels = n_channels, real=args.real)
        else:
            if args.states == 1:
                state_count += 6 + 2
            elif args.states == 2:
                state_count += 6 + 6 + 2
            if args.num_controls > 1:
                model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=args.real_test,retrain_off=args.freeze,input=args.input_size, num_channels = n_channels, real=args.real, state_count = state_count, num_controls=args.num_controls, extra_mins=args.extra_mins, extra_maxs=args.extra_maxs)
            else:
                model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=args.real_test,retrain_off=args.freeze,input=args.input_size, num_channels = n_channels, real=args.real, state_count = state_count)
    else:
        model = OrangeNet18(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=args.real_test,input=args.input_size, num_channels = n_channels)

    if args.load:
        if os.path.isfile(args.load):
            checkpoint = torch.load(args.load, map_location=torch.device('cuda:'+args.gpu))
            model.load_state_dict(checkpoint)
            print("Loaded Checkpoint: ",args.load)
        else:
            print('No checkpoint found at: ',args.load)

    if args.segload:
        if os.path.isfile(args.segload):
            checkpoint = torch.load(args.segload, map_location=torch.device('cuda:'+args.gpu))
            segmodel.load_state_dict(checkpoint)
            print("Loaded Checkpoint: ",args.segload)
        else:
            print('No checkpoint found at: ',args.segload)

    #CUDA Check
    #torch.backends.cudnn.enabled = False #TODO:FIX THIS
    use_cuda = torch.cuda.is_available()
    print('Cuda Flag: ',use_cuda)
    if use_cuda:
        if args.gpu:
            device = torch.device('cuda:'+str(args.gpu))
            model = model.to(device)
            if args.temp_seg:
                segmodel = segmodel.to(device)
        else:
            device = torch.device('cuda')
            model = model.to(device)
            if args.temp_seg:
                segmodel = segmodel.to(device)
            if (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model)
                if args.temp_seg:
                    segmodel = nn.DataParallel(segmodel)
    else:
        device = torch.device('cpu')
    print('Device: ', device)
    if args.temp_seg:
        print('Model Device: ', next(model.parameters()).device, next(segmodel.parameters()).device)
    else:
        print('Model Device: ', next(model.parameters()).device)

    #Create Optimizer
    learning_rate = args.learning_rate
    learn_rate_decay = np.power(1e-3,1/float(args.epochs))#0.9991#10 / args.epochs
    if args.temp_seg:
        optimizer = optim.AdamW(list(model.parameters()) + list(segmodel.parameters()), lr = args.learning_rate, weight_decay=1e-2)
    else:
        optimizer = optim.AdamW(list(model.parameters()), lr = args.learning_rate, weight_decay=1e-2)
    #optimizer = optim.SGD(list(model.parameters()) + list(segmodel.parameters()), lr=args.learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=learn_rate_decay)
    #ReduceLROnPlateau is an interesting idea
    loss_mult = torch.tensor([[0.8, 0.5, 0.3],[0.8, 0.5, 0.3]]).to(device)

    #Save Parameters
    save_variables_divider = args.save_variables #5 #10
    log_path = './model/logs'
    save_path = createStampedFolder(os.path.join(log_path, 'variable_log'))
    tensorboard_path = addTimestamp(os.path.join(log_path, 'tensorboard_'))
    #if args.traj:
    #    val_path = addTimestamp(os.path.join(log_path, 'validation_'))
    plot_data_path = addTimestamp(os.path.join(log_path, 'plot_data_'))
    print("Saving models at: ", save_path)
    print ('Training...')
    writer = SummaryWriter(tensorboard_path)
    #val_writer = SummaryWriter(val_path)
    #graph_writer = Su
    os.makedirs(plot_data_path)

    #print ('Writers Set Up')

    #iters = 0
    if args.traj and False: #TODO Does nothing significant rn 
        plotting_data = dict()
        data_loc = copy.deepcopy(args.data)
        val_outputs_x, val_outputs_y, val_outputs_z, val_outputs_yaw, val_outputs_p, val_outputs_r = loadData(val_idx,dataclass.num_list,data_loc,model,dataclass.traj_list,args.real,dataclass,args)
        print(len(val_idx))
        plotting_data['idx'] = range(len(val_idx))

        if not args.yaw_only:
            plotting_data['truth'] = [val_outputs_x[plotting_data['idx']],
                                      val_outputs_y[plotting_data['idx']],
                                      val_outputs_z[plotting_data['idx']],
                                      val_outputs_yaw[plotting_data['idx']],
                                      val_outputs_p[plotting_data['idx']],
                                      val_outputs_r[plotting_data['idx']]]
        else:
            plotting_data['truth'] = [val_outputs_x[plotting_data['idx']],
                                      val_outputs_y[plotting_data['idx']],
                                      val_outputs_z[plotting_data['idx']],
                                      val_outputs_yaw[plotting_data['idx']]]


        plotting_data['data'] = list()
        #plotting_data['foc_l'] = args.cam_coord
        plotting_data['min'] = model.min
        plotting_data['max'] = model.max
        plotting_data['bins'] = model.bins
        for ii in plotting_data['idx']:
            plotting_data['data'].append([])

    #print(plotting_data)
    #loss = nn.CrossEntropyLoss()
    since = time.time()
    #gc.collect()
    #for obj in gc.get_objects():
    #    try:
    #        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #            print(type(obj), obj.size())
    #    except:
    #            pass
    #model = model.to('cpu')

    seg_mean_image = seg_mean_image.to(device)
    ranges = np.array([np.arange(i*args.bins, (i+1)*args.bins) for i in range(args.num_controls)])
    for epoch in range(args.epochs):            
        
        print('\n\nEpoch: ', epoch)
        #Train
        #gc.collect()
        epoch_acc = [[[], []], [[], []], [[], []]]
        extra_epoch_acc = [[[[], []], [[], []], [[], []]]]
        acc_total = [[0., 0.], [0., 0.], [0., 0.]]
        extra_acc_total = [[[0., 0.], [0., 0.], [0., 0.]] for i in range(args.num_controls-1)]
        extra_elements = [0. for i in range(args.num_controls-1)]
        train_phase_accuracy = 0.
        elements = 0.
        loader = train_loader
        if args.use_sampler:
            loader = weight_loader

        if args.use_error_sampler:
            if not np.sum(np.array(error_weights)) == 0:
                loader = weight_loader

        for ctr, batch in enumerate(loader):
            # print('Batch: ',ctr)
            #image_batch = batch['image'].to(device).float()
            model = model.to(device)
            point_batch = batch['points'] #.to(device)
            optimizer.zero_grad()
            if args.temp_seg:
                segmodel.train()
            model.train()
            with torch.set_grad_enabled(True):
                batch_imgs = batch['image']
                batch_imgs = batch_imgs.to(device)
                #model = model.to(device)
                classifier_loss = None

                if args.temp_seg:
                    segmodel = segmodel.to(device)
                    batch_images = None
                    for img_num in range(args.num_images):
                        seglogits = segmodel(batch_imgs[:, img_num*base_n_channels:(img_num+1)*base_n_channels, :, :])
                        seglogits = seglogits.view(-1,2,segmodel.h,segmodel.w)
                        segimages = (torch.max(seglogits, 1).indices).type(torch.FloatTensor).to(device)
                        #k = 0
                        #floc = "test_segs"
                        #os.makedirs(floc)
                        #for i in range(segimages.shape[0]):
                        #    img = segimages[i, :, :]

                        #    img = (np.array(img) * 255).astype(np.uint8)
                        #    #print(img.shape, np.max(img), np.min(img), np.sum(np.where(img==255, 1, 0)))
                        #    im = Image.fromarray(img.astype(np.uint8))
                        #    im.save(floc + "/output" + str(k + i) + ".png")

                        #exit(0)
                        # print(segimages.shape, seg_mean_image.shape)
                        segimages -= seg_mean_image
                        segimages = torch.reshape(segimages, (segimages.shape[0], 1, segimages.shape[1], segimages.shape[2]))
                        t_batch_imgs =  torch.cat((batch_imgs[:, img_num*(n_channels-1):(img_num+1)*(n_channels-1), :, :], segimages), 1)
                        if batch_images is None:
                            batch_images = t_batch_imgs
                        else:
                            batch_images = torch.cat((batch_images, t_batch_imgs), 1)
                else:
                    batch_images = batch_imgs
                
                if args.states == 0:
                    logits = model(batch_images)
                else:
                    states = get_state_data(args.states, batch)
                    states = states.to(device)
                    logits = model(batch_images, states)

                #print(logits.shape)
                #exit(0)
                #del batch_imgs
                #del batch
                #classifier = logits[:, 0]
                #logits = logits[:, :3600]
                if args.temp_seg:
                    segmodel = segmodel.to('cpu')
                b_size = logits.shape[0]

                if args.num_controls > 1:
                    classifier = logits[:, :model.classifier]
                    classifier = classifier.reshape(-1, model.classifier).to(device)
                    logits = logits[:, model.classifier:]
                    classifier_loss = F.cross_entropy(classifier, batch['phase'].long().to(device))
                    temp_accuracy = phase_accuracy(batch['phase'], classifier)
                    train_phase_accuracy = ((elements * train_phase_accuracy) + (b_size * temp_accuracy))/(elements + b_size)
                    phase = np.array(batch['phase'].to('cpu'))
                else:
                    phase = np.zeros(b_size)

                if not args.regression:
                    if not args.yaw_only:
                        logits = logits.view(-1,6,model.num_points,model.bins*args.num_controls)
                    else:
                        logits = logits.view(-1,4,model.num_points,model.bins*args.num_controls)

                    loss_x = [0. for t in range(model.num_points)]
                    loss_y = [0. for t in range(model.num_points)]
                    loss_z = [0. for t in range(model.num_points)]

                    if not args.yaw_only:
                        loss_r = [0. for t in range(model.num_points)]
                        loss_p = [0. for t in range(model.num_points)]

                    loss_yaw = [0. for t in range(model.num_points)]

                    
                    point_batch = point_batch.to(device)
                    elements_accessed = np.arange(b_size)
                    for temp in range(model.num_points):
                        #print(logits[:,0,temp,:].shape)
                        #print(((point_batch)[:,temp,0]).shape)
                        #exit()        
                        loss_x[temp] += F.cross_entropy(logits[elements_accessed,0,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,0])
                        loss_y[temp] += F.cross_entropy(logits[elements_accessed,1,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,1])
                        loss_z[temp] += F.cross_entropy(logits[elements_accessed,2,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,2])
                        loss_yaw[temp] += F.cross_entropy(logits[elements_accessed,3,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,3])

                        if not args.yaw_only:
                            loss_p[temp] += F.cross_entropy(logits[elements_accessed,4,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,4])
                            loss_r[temp] += F.cross_entropy(logits[elements_accessed,5,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,5])

                    point_batch = point_batch.to('cpu')
                    logits = logits.to('cpu')
                    if args.use_error_sampler:
                        arg2 = 6
                        if not args.yaw_only:
                            logits = logits.view(-1,6,model.num_points,model.bins)
                        else:
                            arg2 = 4
                            logits = logits.view(-1,4,model.num_points,model.bins)
                        idx_batch = batch['idx'].to('cpu')
                        #label_batch = batch['time_frac']
                        #print(logits.shape)
                        #print(idx_batch)
                        for ii, idx in enumerate(idx_batch):
                            temp_logits = (logits.cpu()[ii]).view(1, arg2, model.num_points,model.bins)
                            temp_point_batch = (point_batch.cpu()[ii]).view(1, model.num_points, arg2)
                            acc_list = acc_metric(args, temp_logits, temp_point_batch, args.yaw_only,sphere_flag = args.spherical)
                            #ele = np.where(train_idx == np.int(idx))
                            #print(idx, ele)
                            #print(train_idx)
                            #if len(ele) != 1:
                            #    print(idx, ele)
                            #print(idx)
                            eps = 1e-6
                            error_weights[int(train_loc[int(idx)])] = np.sum(np.array(acc_list)) + eps
                            #print(error_weights[batch['idx'][ii]])

                    acc_list = acc_metric(args,logits,point_batch,args.yaw_only,phase,ranges,sphere_flag = args.spherical)
                    if args.num_controls > 1:
                        acc_list, extra_acc_list = acc_list

                    #del point_batch
                    #del logits

                    batch_loss = None
                    for t in range(model.num_points):
                        if args.custom_loss:
                            if batch_loss is None:
                                batch_loss = loss_mult[0,t]*loss_x[t]
                            else:
                                batch_loss += loss_mult[0,t]*loss_x[t]
                            batch_loss += loss_mult[0,t]*loss_y[t]
                            batch_loss += loss_mult[0,t]*loss_z[t]
                            batch_loss += loss_mult[1,t]*loss_yaw[t]
                            if not args.yaw_only:
                                batch_loss += loss_mult[1,t]*loss_p[t]
                                batch_loss += loss_mult[1,t]*loss_r[t]

                        else:
                            if batch_loss is None:
                                batch_loss = loss_x[t]
                            else:
                                batch_loss += loss_x[t]
                            batch_loss += loss_y[t]
                            batch_loss += loss_z[t]
                            batch_loss += loss_yaw[t]
                            if not args.yaw_only:
                                batch_loss += loss_p[t]
                                batch_loss += loss_r[t]
                   
                    if args.num_controls > 1:
                        batch_loss += classifier_loss
 
                else:
                    logits = logits.view(-1,model.num_points,6)
                    logits = logits.to(device).double()
                    point_batch = point_batch.to(device).double()
                    # print((point_batch)[:,0, 0])
                    # s0, s1, s2 = len(point_batch), len(point_batch[0]), len(point_batch[0][0])
                    # pb = torch.Tensor(s0, s1, s2)
                    # print(s0, s1, s2)
                    # list_pb = []
                    # for ele_s0 in range(s0):
                    #     list_list_pb = torch.Tensor(s1, s2)
                    #     torch.cat(point_batch[ele_s0], out=list_list_pb)
                    #     list_pb.append(list_list_pb)
                    # print(pb.shape)
                    # torch.cat(list_pb, out=pb)
                    if args.spherical:
                        temp_transform = sphericalToXYZ()
                        batch_loss = F.mse_loss(temp_transform(logits),point_batch)#TODO: make sure this works
                    else:
                        batch_loss = F.mse_loss(logits, point_batch)
                    acc_list = acc_metric_regression(args, logits.cpu(), point_batch.cpu(),sphere_flag = args.spherical)

                #print(batch_loss)
                #batch_loss.to_cpu()
                """lamda = torch.tensor(1.).to(device)
                l2_reg = torch.tensor(0.)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                l2_reg = l2_reg.to(device)
                batch_loss += lamda * l2_reg"""
                batch_loss = batch_loss.to(device)
                # batch_imgs = batch_imgs.to(device)
                #print(dir())
                #print(optimizer)
                # logits = logits.to(device)
                # point_batch = point_batch.to(device)
                model = model.to(device)
                #optimizer = optimizer.to('cpu')
                loss_mult = loss_mult.to(device)
                if not args.regression:
                    for i in range(model.num_points):
                        loss_x[i] = loss_x[i].to(device)
                        loss_y[i] = loss_y[i].to(device)
                        loss_z[i] = loss_z[i].to(device)
                        loss_yaw[i] = loss_yaw[i].to(device)
                        if not args.yaw_only:
                            loss_p[i] = loss_p[i].to(device)
                            loss_r[i] = loss_r[i].to(device)

                #print(batch_loss.device, l2_reg.device, lamda.device, logits.device, point_batch.device)
                batch_loss.backward()
                optimizer.step()
                model = model.to('cpu')

                del batch_imgs, point_batch, logits, batch_images, batch
                if args.states != 0:
                    del states

                if not args.regression:
                    writer.add_scalar('train_loss',batch_loss,ctr+epoch*len(train_loader))
                    writer.add_scalar('train_loss_x',torch.sum(torch.tensor(loss_x)),ctr+epoch*len(train_loader))
                    writer.add_scalar('train_loss_y',torch.sum(torch.tensor(loss_y)),ctr+epoch*len(train_loader))
                    writer.add_scalar('train_loss_z',torch.sum(torch.tensor(loss_z)),ctr+epoch*len(train_loader))
                    writer.add_scalar('train_loss_yaw',torch.sum(torch.tensor(loss_yaw)),ctr+epoch*len(train_loader))
                    if not args.yaw_only:
                        writer.add_scalar('train_loss_p',torch.sum(torch.tensor(loss_p)),ctr+epoch*len(train_loader))
                        writer.add_scalar('train_loss_r',torch.sum(torch.tensor(loss_r)),ctr+epoch*len(train_loader))

                for ii, acc in enumerate(acc_list):
                    writer.add_scalar('train_acc_'+str(ii),acc[0],ctr+epoch*len(train_loader))
                #print('Cross-Entropy Loss: ',batch_loss.item(),[loss_x.item(),loss_y.item(),loss_z.item()])
                #print('Training Accuracy: ',acc_list)
                #print(type(acc_list))

                for ii in range(args.num_controls):
                    temp_ctr = len(np.argwhere(phase==ii))
                    for i in range(len(acc_total)):
                        for j in range(2):
                            if ii == 0:
                                if elements + temp_ctr == 0.:
                                    continue
                                acc_total[i][j] = ((elements * acc_total[i][j]) + (temp_ctr * acc_list[i][j]))/(elements + temp_ctr)
                                #epoch_acc[i][j].append(acc_list[i][j])
                            else:
                                if extra_elements[ii-1] + temp_ctr == 0.:
                                    continue
                                extra_acc_total[ii-1][i][j] = ((extra_elements[ii-1] * extra_acc_total[ii-1][i][j]) + (temp_ctr * extra_acc_list[ii-1][i][j]))/(extra_elements[ii-1] + temp_ctr)                               

                    if ii == 0:
                        elements += temp_ctr
                    else:
                        extra_elements[ii-1] += temp_ctr

                #print(b_size)

        #Validation
        #print("Reach here")
        print('Training Accuracy: ',acc_total)
        if args.num_controls > 1:
            print("Train Phase Loss:", classifier_loss)
            print('Training Phase Accuracy:', train_phase_accuracy)
            print("Training Extra:", extra_acc_total)
        #exit()
        if args.yaw_only:
            val_loss = [0.,0.,0.,0.]
        else:
            val_loss = [0.,0.,0.,0.,0.,0.]

        val_acc = np.zeros((model.num_points,2))
        if args.num_controls > 1:
            extra_val_acc = np.zeros((args.num_controls-1,model.num_points,2))
        # if not args.yaw_only:
        #     resnet_output = np.zeros((0, 6, args.num_pts, model.bins))
        # else:
        #     resnet_output = np.zeros((0, 4, args.num_pts, model.bins))

        val_elements = [0. for i in range(args.num_controls)]


        val_phase_accuracy = 0.

        for batch in val_loader:
            val_classifier_loss = None
            with torch.set_grad_enabled(False):
                image_batch = batch['image'].to(device)
                model = model.to(device)
                if args.temp_seg:
                    segmodel = segmodel.to(device)
                    batch_images = None
                    for img_num in range(args.num_images):
                        seglogits = segmodel(image_batch[:, img_num*base_n_channels:(img_num+1)*base_n_channels, :, :])
                        seglogits = seglogits.view(-1,2,segmodel.h,segmodel.w)
                        segimages = (torch.max(seglogits, 1).indices).type(torch.FloatTensor).to(device)
                        #k = 0
                        #floc = "test_segs"
                        #os.makedirs(floc)
                        #for i in range(segimages.shape[0]):
                        #    img = segimages[i, :, :]

                        #    img = (np.array(img) * 255).astype(np.uint8)
                        #    #print(img.shape, np.max(img), np.min(img), np.sum(np.where(img==255, 1, 0)))
                        #    im = Image.fromarray(img.astype(np.uint8))
                        #    im.save(floc + "/output" + str(k + i) + ".png")

                        #exit(0)
                        segimages -= seg_mean_image
                        segimages = torch.reshape(segimages, (segimages.shape[0], 1, segimages.shape[1], segimages.shape[2]))
                        t_batch_imgs =  torch.cat((image_batch[:, img_num*(n_channels-1):(img_num+1)*(n_channels-1), :, :], segimages), 1)
                        if batch_images is None:
                            batch_images = t_batch_imgs
                        else:
                            batch_images = torch.cat((batch_images, t_batch_imgs), 1)
                    segmodel = segmodel.to('cpu')
                else:
                    batch_images = image_batch

                #print(segimages.shape, batch_imgs.shape)
                # model = model.to(device)
                model.eval()
                # logits = model(batch_images)
                if args.states == 0:
                    logits = model(batch_images)
                else:
                    states = get_state_data(args.states, batch)
                    states = states.to(device)
                    logits = model(batch_images, states)

                del image_batch
                b_size = logits.shape[0]

                if args.num_controls > 1:
                    classifier = logits[:, :model.classifier].reshape(-1, model.classifier).to(device)
                    logits = logits[:, model.classifier:]
                    val_classifier_loss = F.cross_entropy(classifier, batch['phase'].long().to(device))
                    temp_accuracy = phase_accuracy(batch['phase'], classifier)
                    val_phase_accuracy += temp_accuracy
                    phase = np.array(batch['phase'].to('cpu'))
                else:
                    phase = np.zeros(b_size)

                if not args.regression:
                    if not args.yaw_only:
                        logits = logits.view(-1,6,model.num_points,model.bins*args.num_controls)
                    else:
                        logits = logits.view(-1,4,model.num_points,model.bins*args.num_controls)

                    loss_x = [0. for t in range(model.num_points)]
                    loss_y = [0. for t in range(model.num_points)]
                    loss_z = [0. for t in range(model.num_points)]

                    if not args.yaw_only:
                        loss_r = [0. for t in range(model.num_points)]
                        loss_p = [0. for t in range(model.num_points)]

                    loss_yaw = [0. for t in range(model.num_points)]

                    
                    point_batch = batch['points'].to(device)
                    elements_accessed = np.arange(b_size)
                    for temp in range(model.num_points):                     
                        val_loss[0] += F.cross_entropy(logits[elements_accessed,0,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,0])
                        val_loss[1] += F.cross_entropy(logits[elements_accessed,1,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,1])
                        val_loss[2] += F.cross_entropy(logits[elements_accessed,2,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,2])
                        val_loss[3] += F.cross_entropy(logits[elements_accessed,3,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,3])

                        if not args.yaw_only:
                            val_loss[4] += F.cross_entropy(logits[elements_accessed,4,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,4])
                            val_loss[5] += F.cross_entropy(logits[elements_accessed,5,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,5])

                    val_acc_list = acc_metric(args,logits.cpu(),point_batch.cpu(), args.yaw_only,phase,ranges,sphere_flag = args.spherical)

                    if args.num_controls > 1:
                        val_acc_list, extra_val_acc_list = val_acc_list
    
                else:
                    logits = logits.view(-1,model.num_points,6)
                    logits = logits.to(device).double()
                    point_batch = batch['points'].to(device).double()
                    # s0, s1, s2 = len(point_batch), len(point_batch[0]), len(point_batch[0][0])
                    # pb = torch.Tensor(s0, s1, s2)
                    # list_pb = []
                    # for ele_s0 in range(s0):
                    #     list_list_pb = torch.Tensor(s1, s2)
                    #     torch.cat(point_batch[ele_s0], out=list_list_pb)
                    #     list_pb.append(list_list_pb)
                    # torch.cat(list_pb, out=pb)
                    # batch_loss = F.mse_loss(logits, pb)
                    val_loss = F.mse_loss(logits, point_batch)
                    val_acc_list = acc_metric_regression(args, logits.cpu(), point_batch.cpu(),sphere_flag = args.spherical)
                logits = logits.to('cpu')
                #print(logits.shape)
                #resnet_output = np.concatenate((resnet_output,logits), axis=0)
                #print(resnet_output.shape)

                for phase_num in range(args.num_controls):
                    phase_i = len(np.argwhere(phase==phase_num))
                    val_elements[phase_num] += phase_i
                    if phase_num == 0:
                        for ii, acc in enumerate(val_acc_list):
                            for jj, acc_j in enumerate(acc):
                                val_acc[ii][jj] += phase_i * acc_j
                                epoch_acc[ii][jj].append(acc_j)
                    else:
                        for ii, acc in enumerate(extra_val_acc_list[phase_num-1]):
                            for jj, acc_j in enumerate(acc):
                                extra_val_acc[phase_num-1][ii][jj] += phase_i * acc_j
                                extra_epoch_acc[phase_num-1][ii][jj].append(acc_j)

                del point_batch
                
                if args.states != 0:
                    del states

                model = model.to('cpu')
        
        if val_elements[0] != 0.:
            val_acc = val_acc/val_elements[0]
        if args.num_controls > 1:
            val_phase_accuracy /= len(val_loader)
            for i in range(len(extra_val_acc)):
                phase_i = val_elements[i+1]
                if phase_i != 0:
                    extra_val_acc[i] /= phase_i

        if args.traj and False:
            for ii in plotting_data['idx']:
                plotting_data['data'][ii].append(resnet_output[ii,:,:,:])

            if args.plot_data:
                with open(plot_data_path+'/data.pickle','wb') as f:
                    pickle.dump(plotting_data,f,pickle.HIGHEST_PROTOCOL)
        if not args.regression:
            if not args.yaw_only:
                val_loss = [val_loss[temp].item()/len(val_loader) for temp in range(6)]
            else:
                val_loss = [val_loss[temp].item()/len(val_loader) for temp in range(4)]
            
            if val_classifier_loss is None:
                writer.add_scalar('val_loss',sum(val_loss),(epoch+1)*len(train_loader))
            else:
                writer.add_scalar('val_loss',sum(val_loss)+val_classifier_loss,(epoch+1)*len(train_loader))
            writer.add_scalar('val_loss_x',val_loss[0],(epoch+1)*len(train_loader))
            writer.add_scalar('val_loss_y',val_loss[1],(epoch+1)*len(train_loader))
            writer.add_scalar('val_loss_z',val_loss[2],(epoch+1)*len(train_loader))
            writer.add_scalar('val_loss_yaw',val_loss[3],(epoch+1)*len(train_loader))
            if not args.yaw_only:
                writer.add_scalar('val_loss_p',val_loss[4],(epoch+1)*len(train_loader))
                writer.add_scalar('val_loss_r',val_loss[5],(epoch+1)*len(train_loader))
        for ii, acc in enumerate(val_acc):
            writer.add_scalar('val_acc_'+str(ii),acc[0],(epoch+1)*len(train_loader))
        print('Val Cross-Entropy Loss: ', val_loss)
        print('Val Accuracy: ',val_acc)
        if args.num_controls > 1:
            print("Val Phase Loss:", val_classifier_loss)
            print("Val Phase Accuracy:", val_phase_accuracy)
            print("Val Extra Accuracy:", extra_val_acc)
        #NOTE: Experimental --- might be bad
        if args.use_error_sampler and epoch % sampler_n_epochs == 0:
            #print(type(error_weights))
            #print(len(error_weights))
            #print(error_weights[0])
            temp_err = np.where(np.array(error_weights) == 0.0, 1, 0)
            print(np.sum(temp_err))
            print(np.sum(np.array(error_weights)))
            #print(error_weights)
            weight_sampler = WeightedRandomSampler(error_weights, len(train_data))
            weight_loader = DataLoader(train_data,batch_size=args.batch_size,sampler=weight_sampler,num_workers=args.j, worker_init_fn=dl_init)

        if args.use_sampler:
            #Measure all training samples 
            time_bin_num = 100 #Make into arg (maybe)
            time_sample_cnt = [0 for ii in range(time_bin_num)]
            time_sample_metric = [None for ii in range(time_bin_num)]
            #Populate cnt and metrics from full data
            for batch in train_loader:
                with torch.set_grad_enabled(False):
                    image_batch = batch['image'].to(device)
                    model = model.to(device)
                    model.eval()
                    logits = model(image_batch)
                    del image_batch
                    arg2 = 6
                    if not args.yaw_only:
                        logits = logits.view(-1,6,model.num_points,model.bins)
                    else:
                        arg2 = 4
                        logits = logits.view(-1,4,model.num_points,model.bins)
                    point_batch = batch['points']
                    label_batch = batch['time_frac']
                    #print(logits.shape)
                    for ii, label in enumerate(label_batch):
                        label = int(label*time_bin_num)
                        if label >= time_bin_num:
                            label = time_bin_num - 1
                        temp_logits = (logits.cpu()[ii]).view(1, arg2, model.num_points,model.bins)
                        temp_point_batch = (point_batch.cpu()[ii]).view(1, model.num_points, arg2)
                        acc_list = acc_metric(args, temp_logits, temp_point_batch, args.yaw_only,sphere_flag = args.spherical)
                        total_acc = np.sum(np.array(acc_list)) #TODO: This can be changed to a weighted sum later #TODO: Check
                        time_sample_cnt[label] += 1
                        if time_sample_metric[label] is None:
                            time_sample_metric[label] = 0
                        time_sample_metric[label] += total_acc
                    logits = logits.to('cpu')
                    del point_batch
                    model = model.to('cpu')
            #Normalize metric by count
            times = []
            avg_metric = []
            for ii in range(time_bin_num):
                if time_sample_metric[ii] is not None:
                    times.append(ii*(1.0/time_bin_num))
                    avg_metric.append(time_sample_metric[ii]/time_sample_cnt[ii])
            print("Avg Metrics",times,avg_metric)
            #Make weight list for sampler #TODO: once this is working, see if there's a more efficient way
            weights = []
            for ii in range(len(train_data)):
                time_frac = train_data[ii]['time_frac']
                label = int(time_frac*time_bin_num)
                if label >= time_bin_num:
                    label = time_bin_num - 1
                weights.append(avg_metric[label])
            #Create new sampler
            #print(type(weights))
            #print(len(weights))
            #print(weights[0])
            #exit(0)
            weight_sampler = WeightedRandomSampler(weights, len(train_data))
            weight_loader = DataLoader(train_data,batch_size=args.batch_size,sampler=weight_sampler,num_workers=args.j, worker_init_fn=dl_init)

        #Adjust LR
        scheduler.step()
        print('Learning Rate Set to: ',scheduler.get_lr())
        # Save variables
        if ((epoch + 1) % save_variables_divider == 0 or (epoch == 0) or (epoch == args.epochs - 1)):
            print("Saving variables")
            save_model(epoch)

        for ii in range(args.num_controls):
            print("Phase:", ii)
            if ii == 0:
                epoch_acc = np.array(epoch_acc)
                mean_str = ""
                var_str = ""
                for i in range(len(epoch_acc)):
                    for j in range(len(epoch_acc[i])):
                        mean_str += (str(np.mean(epoch_acc[i][j])) + "\t")
                        var_str += (str(np.std(epoch_acc[i][j])) + "\t")
                print(mean_str)
                print(var_str)
            else:
                extra_epoch_acc = np.array(extra_epoch_acc)
                mean_str = ""
                var_str = ""
                for i in range(len(extra_epoch_acc[ii-1])):
                    for j in range(len(extra_epoch_acc[ii-1][i])):
                        mean_str += (str(np.mean(extra_epoch_acc[ii-1][i][j])) + "\t")
                        var_str += (str(np.std(extra_epoch_acc[ii-1][i][j])) + "\t")
                print(mean_str)
                print(var_str)

    writer.close()
    print("Done")

if __name__ == '__main__':
    #print("asdf")
    main()
