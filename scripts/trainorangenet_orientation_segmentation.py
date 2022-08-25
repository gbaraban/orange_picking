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

def acc_metric_regression(args, logits, point_batch, phases, sphere_flag = False):
    acc_list = []
    extra_acc_list = [[] for i in range(args.num_controls-1)]
    shape = logits.size()
    logits = logits.detach()
    if sphere_flag:
        logits = sphericalToXYZ()(logits)
    for pt in range(args.num_pts):
        batch_list = []
        extra_batch_list = [[] for i in range(args.num_controls-1)]
        ang_d = []
        extra_ang_d = [[] for i in range(args.num_controls-1)]
        for ii in range(shape[0]):
            phase = phases[ii]
            coord_list = []
            for coord in range(3):
                pred = logits[ii,(pt*6)+coord]
                d = pred - point_batch[ii,(pt*6)+coord]
                coord_list.append(d)
            d = np.vstack([coord_list[0],coord_list[1],coord_list[2]])
            if phase > 0:
                extra_batch_list[phase-1].append(np.linalg.norm(d))
            else:
                batch_list.append(np.linalg.norm(d))

            true_angle = []
            pred_angle = []
            for coord in range(3,6):
                pred = logits[ii,(pt*6)+coord]
                true_angle.append(point_batch[ii, (pt*6)+coord])
                pred_angle.append(pred)
            if phase > 0:
                extra_ang_d[phase-1].append(angle_dist(true_angle,pred_angle))
            else:
                ang_d.append(angle_dist(true_angle, pred_angle))

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
        return acc_list,extra_acc_list
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
    elif state_config == 3:
        magnet_data = data["magnet"]
        orange_pose_data = data["orange_pose"]
        rp_data = data["rp"]
        states_data = torch.cat((magnet_data, orange_pose_data, rp_data), 1)
        return states_data

    elif state_config == 4:
        rp_data = data["rp"]
        states_data = torch.Tensor(rp_data)
        return states_data

    elif state_config == 5:
        orange_pose_data = data["orange_pose"]
        states_data = torch.Tensor(orange_pose_data)
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

def run_seg(segmodel,batch):
    dev = batch.device
    seglogits = segmodel(batch)
    segimages = (torch.max(seglogits, 1).indices).type(torch.FloatTensor).to(dev)
    segimages = torch.reshape(segimages, (segimages.shape[0], 1, segimages.shape[1], segimages.shape[2]))
    return segimages
 
def main():
    signal.signal(signal.SIGINT,signal_handler)
    global model
    global save_path
    global writer
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data folder')
    parser.add_argument('--load', help='model to load')
    parser.add_argument('--epochs', type=int, default=1000, help='max number of epochs to train for/but also controls how fast learning rate decays')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', help='gpu to use')
    parser.add_argument('--num_images', type=int, default=1, help='number of input images')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=100, help='batch size for validation')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='starting learning rate')
    parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
    parser.add_argument('--capacity', type=float, default=1, help='network capacity')
    parser.add_argument('--bins', type=int, default=100, help='number of bins per coordinate')
    parser.add_argument('-j', type=int, default=8, help='number of loader workers')
    parser.add_argument('--traj', type=int, default=1, help='train trajectories')
    parser.add_argument('--val', type=float, default=0.10, help='validation percentage')
    parser.add_argument('--resnet18', type=int, default=0, help='ResNet18 or ResNet8')
    parser.add_argument('--yaw_only', type=int, default=0, help='yaw only')
    parser.add_argument('--custom', type=str, default="no_parse", help='custom parser: Run18/no_parse (used in sim data and normal real world, Run18/no_parse allows to read unparsed future waypoint data)')
    parser.add_argument('--train', type=int, default=1, help='train or test')
    parser.add_argument('--data_aug_flip',type=int,default=1, help='Horizontal flipping on/off')
    parser.add_argument('--freeze',type=str,default="",help='layer to freeze while training, linear or conv')
    parser.add_argument('--plot_data',type=int,default=0,help="plot data for accuracy")
    parser.add_argument('--input_size',type=float,default=1,help='input size change')
    parser.add_argument('--custom_loss',type=bool,default=False,help='custom loss for training')
    parser.add_argument('--depth',type=bool,default=False,help='use depth channel')
    parser.add_argument('--seg',default=True,help='use segmentation channel, use this')
    parser.add_argument('--seg_only',type=bool,default=False,help='use segmentation channel, don\'t use mostly')
    parser.add_argument('--save_variables',type=int,default=2,help="save after every x epochs")
    parser.add_argument('--mean_seg',type=str,default='useful_models/mean_imgv2_data_data_collection4_real_world_traj_bag.npy',help='mean segmentation image, two options for real and sim world in comments') # data/depth_data/data/mean_seg.npy #data/mean_imgv2_data_seg_Run24.npy
    parser.add_argument('--segload', type=str, default="useful_models/model_seg145.pth.tar", help='segment model to load')
    parser.add_argument('--retrain_off_seg',type=bool,default=True,help='retrain of segmentation off')
    parser.add_argument('--relative_pose', type=bool, default=True,help='relative pose of points')
    parser.add_argument('--regression', type=bool, default=False,help='Use regression loss (MSE)')
    parser.add_argument('--spherical', type=bool, default=False,help='Use spherical coordinates')
    parser.add_argument('--pred_dt',type=float, default=1.0, help="Space between predicted points")
    parser.add_argument('--extra_dt', nargs="+", type=float, default=[0.25, 0.25, 0], help='pred_dt for extra phases')
    parser.add_argument('--image_dt',type=float,default=1.0, help="Space between images provided to network")
    parser.add_argument('--states',type=int,default=1,help="states to use with network, options: (1, 2, 3), check get_states function to see which version uses which states")
    parser.add_argument('--num_controls',type=int,default=4,help="to divide data into multiple scenarios, to teach a different control strategy for each")
    parser.add_argument('--reduce_n', type=bool, default=True,help='Remove last few events from staging dataset')
    parser.add_argument('--relabel', type=bool, default=False,help='Relabel last few events from staging')
    parser.add_argument('--resets', type=bool, default=True,help='Reset data used for training')
    parser.add_argument('--body_v_dt', type=float, default=None,help='Average body v over longer dts')
    parser.add_argument('--remove_hover', nargs="+", type=float, default=None,help='Threshold to remove equilibrium staging points')
    parser.add_argument('--perspective', default=0, help='apply random perspective')
    parser.add_argument('--print', default=5, help='Print the first x datapoints, for introspection')
    parser.add_argument('--phase_end', default=False, help='Predict the end of the phase')
    parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz/overwritten later')
    parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz/ overwritten later')
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

    if args.reduce_n == 0:
        args.reduce_n = False
    else:
        args.reduce_n = True

    if args.relabel == 0:
        args.relabel = False
    else:
        args.relabel = True

    if args.states in [3]:
        args.magnet = True
    else:
        args.magnet = False

    if args.remove_hover is not None:
        args.remove_hover = tuple(args.remove_hover) 

    if args.depth == 0:
        args.depth = False
    else:
        args.depth = True

    from customDatasetv1 import OrangeSimDataSet, SubSet

    if args.num_controls > 1:
        from orangenetarchmulticontrol import OrangeNet8, OrangeNet18
    elif args.states != 0:
        from orangenetarchstates import OrangeNet8, OrangeNet18
    elif args.states == 0 and args.num_controls == 1:
        from orangenetarch import OrangeNet8, OrangeNet18

    #Set min and max
    if args.spherical:
        args.min = [(0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2), (0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2), (0.0, -np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2)]
        args.max = [(0.7, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2), (0.7, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2), (0.7, np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2)]
    else:
        stage_min = [(0, -0.10, -0.1, -0.20, -0.1, -0.1), (0, -0.13, -0.1, -0.20, -0.1, -0.1), (0, -0.14, -0.1, -0.20, -0.1, -0.1)]
        stage_max = [(0.75, 0.1, 0.1, 0.20, 0.1, 0.1), (0.75, 0.13, 0.1, 0.20, 0.1, 0.1), (0.75, 0.14, 0.1, 0.20, 0.1, 0.1)]
        final_min = [(-0.01, -0.01, -0.01, -0.01, -0.03, -0.03), (-0.01, -0.01, -0.01, -0.01, -0.1, -0.1), (-0.01, -0.01, -0.01, -0.01, -0.1, -0.1)]
        final_max = [(0.04, 0.01, 0.03, 0.01, 0.03, 0.04), (0.04, 0.01, 0.03, 0.01, 0.04, 0.04), (0.04, 0.01, 0.03, 0.01, 0.04, 0.04)]
        reset_min = [(-0.05, -0.01, -0.03, -0.04, -0.03, -0.03), (-0.05, -0.01, -0.03, -0.04, -0.1, -0.1), (-0.05, -0.01, -0.04, -0.01, -0.1, -0.1)]
        reset_max = [(0.0, 0.01, 0.0, 0.04, 0.03, 0.04), (0.0, 0.01, 0.0, 0.04, 0.04, 0.04), (0.0, 0.01, 0.0, 0.04, 0.04, 0.04)]
        grip_min = [(0,0,0,0,0,0),(0,0,0,0,0,0),(0,0,0,0,0,0)]
        grip_max = [(0,0,0,0,0,0),(0,0,0,0,0,0),(0,0,0,0,0,0)]
        args.min = stage_min
        args.max = stage_max
        args.extra_mins = [final_min,reset_min,grip_min]
        args.extra_maxs = [final_max,reset_max,grip_max]
    """
    if args.relative_pose:
      #Change these
      print("Relative Pose")
          if args.real == 0:
              args.min = [(-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -0.5, -0.25, -0.5, -0.1, -0.1), (-0.1, -0.5, -0.25, -0.5, -0.1, -0.1)]
              args.max = [(1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.0, 0.5, 0.25, 0.5, 0.1, 0.1), (1.0, 0.5, 0.25, 0.5, 0.1, 0.1)]
          elif args.real == 1:
              #args.min = [(-0.25, -0.5, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.5, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.5, -0.25, -np.pi/2, -0.1, -0.1)]
              #args.max = [(0.75, 0.75, 0.25, np.pi/2, 0.1, 0.1), (0.75, 0.75, 0.25, np.pi/2, 0.1, 0.1), (0.75, 0.75, 0.25, np.pi/2, 0.1, 0.1)]
              #args.min = [(-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1), (-0.25, -0.25, -0.25, -np.pi/2, -0.1, -0.1)]
              #args.max = [(0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1), (0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1), (0.45, 0.45, 0.25, np.pi/2, 0.1, 0.1)]
              if args.pred_dt <= 0.5:
                  print("Pred DT: 0.5")
                  args.min = [(-0.03, -0.1, -0.04, -0.12, -0.04, -0.04), (-0.03, -0.1, -0.04, -0.12, -0.04, -0.04), (-0.03, -0.1, -0.04, -0.12, -0.04, -0.04)]
                  args.max = [(0.20, 0.1, 0.04, 0.12, 0.04, 0.04), (0.20, 0.1, 0.04, 0.12, 0.04, 0.04), (0.20, 0.1, 0.04, 0.12, 0.04, 0.04)]
              else:
                  args.min = [(-0.10, -0.20, -0.10, -0.25, -0.075, -0.075), (-0.10, -0.15, -0.10, -0.25, -0.075, -0.075), (-0.10, -0.15, -0.10, -0.25, -0.075, -0.075)]
                  args.max = [(0.30, 0.20, 0.10, 0.25, 0.075, 0.075), (0.30, 0.15, 0.10, 0.25, 0.075, 0.075), (0.30, 0.15, 0.10, 0.25, 0.075, 0.075)]

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
        if (args.extra_dt == [0.25] or (len(args.extra_dt) == 1 and args.extra_dt[0] <= 0.25)):
            print("Extra dt 0.25")
            if args.mix_labels:
                args.extra_mins = [[(-0.030, -0.05, -0.01, -0.08, -0.02, -0.025), (-0.030, -0.05, -0.01, -0.08, -0.02, -0.025), (-0.030, -0.05, -0.01, -0.08, -0.02, -0.025)]]
                args.extra_maxs = [[(0.035, 0.05, 0.03, 0.08, 0.02, 0.025), (0.035, 0.05, 0.03, 0.08, 0.02, 0.025), (0.035, 0.05, 0.03, 0.08, 0.02, 0.025)]]
            else:
                #args.extra_mins = [[(-0.030, -0.05, -0.01, -0.08, -0.02, -0.025), (-0.030, -0.05, -0.01, -0.08, -0.02, -0.025), (-0.030, -0.05, -0.01, -0.08, -0.02, -0.025)]]
                #args.extra_maxs = [[(0.035, 0.05, 0.03, 0.08, 0.02, 0.025), (0.035, 0.05, 0.03, 0.08, 0.02, 0.025), (0.035, 0.05, 0.03, 0.08, 0.02, 0.025)]]
                args.extra_mins = [[(-0.05, -0.05, -0.075, -0.10, -0.03, -0.03), (-0.05, -0.05, -0.075, -0.10, -0.03, -0.03), (-0.05, -0.05, -0.075, -0.10, -0.03, -0.03)]]
                args.extra_maxs = [[(0.15, 0.05, 0.075, 0.10, 0.03, 0.03), (0.15, 0.05, 0.075, 0.10, 0.03, 0.03), (0.15, 0.05, 0.075, 0.10, 0.03, 0.03)]]
                #args.extra_mins = [[(-0.01, -0.02, -0.01, -0.03, -0.02, -0.02), (-0.01, -0.02, -0.01, -0.03, -0.02, -0.02), (-0.01, -0.02, -0.01, -0.03, -0.02, -0.02)]]
                #args.extra_maxs = [[(0.03, 0.02, 0.03, 0.03, 0.02, 0.02), (0.03, 0.02, 0.03, 0.03, 0.02, 0.02), (0.03, 0.02, 0.03, 0.03, 0.02, 0.02)]]
        elif args.extra_dt == [0.5] or (len(args.extra_dt) == 1 and args.extra_dt[0] <= 0.5):
            print("Extra dt 0.5")
            args.extra_mins = [[(-0.05, -0.02, -0.01, -0.05, -0.03, -0.03), (-0.05, -0.02, -0.01, -0.05, -0.03, -0.03), (-0.05, -0.02, -0.01, -0.05, -0.03, -0.03)]]
            extra_maxs = [[( 0.05,  0.02,  0.035,  0.05,  0.03,  0.03), ( 0.05,  0.02,  0.035,  0.05,  0.03,  0.03), ( 0.05,  0.02,  0.035,  0.05,  0.03,  0.03)]]
        else:
            args.extra_mins = [[(-0.05, -0.05, -0.075, -0.10, -0.03, -0.03), (-0.05, -0.05, -0.075, -0.10, -0.03, -0.03), (-0.05, -0.05, -0.075, -0.10, -0.03, -0.03)]]
            args.extra_maxs = [[(0.15, 0.05, 0.075, 0.10, 0.03, 0.03), (0.15, 0.05, 0.075, 0.10, 0.03, 0.03), (0.15, 0.05, 0.075, 0.10, 0.03, 0.03)]]

        if len(args.extra_dt) == 2:
            args.extra_mins.append([(-0.1, -0.05, -0.03, -0.075, -0.03, -0.03), (-0.1, -0.05, -0.03, -0.075, -0.03, -0.03), (-0.1, -0.05, -0.03, -0.075, -0.03, -0.03)])
            args.extra_maxs.append([(0.01, 0.05, 0.01, 0.075, 0.03, 0.03), (0.01, 0.05, 0.01, 0.075, 0.03, 0.03), (0.01, 0.05, 0.01, 0.075, 0.03, 0.03)])
    else:
        args.extra_mins = None
        args.extra_maxs = None
        """
    if args.phase_end:
        args.num_pts = 1
        stage_min = [(0, -0.7, -0.2, -0.20, -0.1, -0.1)]
        stage_max = [(2.5, 0.7, 0.2, 0.20, 0.1, 0.1)]
        final_min = [(0, -0.05, -0.01, -0.1, -0.05, -0.04)]
        final_max = [(0.25, 0.05, 0.2, 0.1, 0.05, 0.04)]
        reset_min = [(-0.30, -0.1, -0.20, -0.2, -0.06, -0.04)]
        reset_max = [(0.0, 0.1, 0.0, 0.2, 0.06, 0.04)]
        grip_min = [(0,0,0,0,0,0)]
        grip_max = [(0,0,0,0,0,0)]
        args.min = stage_min
        args.max = stage_max
        args.extra_mins = [final_min,reset_min,grip_min]
        args.extra_maxs = [final_max,reset_max,grip_max]

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

    img_trans = []
    if args.train == 1 and args.data_aug_flip == 1:
        img_trans.append(RandomHorizontalTrajFlip(p=0.5))
    args.perspective = float(args.perspective)
    if args.perspective != 0:
        img_trans.append(WaypointPerspective(0.5,args.perspective,args.relative_pose))
    if len(img_trans) > 0:
        print("Image transform set")
        img_trans = transforms.Compose(img_trans)
    else:
        img_trans = None

    sampler_n_epochs = 2
    #Load Mean image
    print("Test")
    mean_img_loc = args.data + '/mean_color_image.npy'
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

    #TOCHECK
    dataclass = OrangeSimDataSet(args.data, args.num_images, args.num_pts, pt_trans, img_trans, depth=args.depth, rel_pose=args.relative_pose, pred_dt = args.pred_dt, img_dt=args.image_dt, reduce_N=args.reduce_n, use_resets=args.resets, extra_dt=args.extra_dt, relabel=args.relabel, mix_labels=False, body_v_dt = args.body_v_dt, remove_hover = args.remove_hover, use_magnet=args.magnet)

    #Break up into validation and training
    #val_perc = 0.07
    val_perc = args.val
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.traj:
        print("No traj")
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

    #train_loc = {}
    #for i, id in enumerate(train_idx):
    #    train_loc[id] = i
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

    if args.depth:
        n_channels += 1
    if (args.seg is True):
        n_channels += 1

    state_count = 0

    if not args.resnet18:
        if args.states == 0 and args.num_controls == 1:
            model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=False,retrain_off=args.freeze,input=args.input_size, num_channels = n_channels, real=1)
        else:
            if args.states == 1:
                state_count += 6 + 2
            elif args.states == 2:
                state_count += 6 + 6 + 2
            elif args.states == 3:
                state_count += 3 + 6 + 2
            elif args.states == 4:
                state_count += 2
            elif args.states == 5:
                state_count += 6
            if args.num_controls > 1:
                model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=False,retrain_off=args.freeze,input=args.input_size, num_channels = n_channels, real=1, state_count = state_count, num_controls=args.num_controls, extra_mins=args.extra_mins, extra_maxs=args.extra_maxs)
            else:
                model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=False,retrain_off=args.freeze,input=args.input_size, num_channels = n_channels, real=1, state_count = state_count)
    else:
        model = OrangeNet18(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=False,input=args.input_size, num_channels = n_channels)

    if args.load:
        if os.path.isfile(args.load):
            checkpoint = torch.load(args.load, map_location=torch.device('cuda:'+args.gpu))
            model.load_state_dict(checkpoint)
            print("Loaded Checkpoint: ",args.load)
        else:
            print('No checkpoint found at: ',args.load)


    #CUDA Check
    #torch.backends.cudnn.enabled = False #TODO:FIX THIS
    use_cuda = torch.cuda.is_available()
    print('Cuda Flag: ',use_cuda)
    if use_cuda:
        if args.gpu:
            device = torch.device('cuda:'+str(args.gpu))
            model = model.to(device)
        else:
            device = torch.device('cuda')
            model = model.to(device)
            if (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    print('Device: ', device)

    if args.seg is True:
        print("Segmentation Enabled")
        seg_mean_image = np.transpose(np.load(args.mean_seg),[2,0,1])
        seg_mean_image = torch.tensor(seg_mean_image)
        seg_mean_image = seg_mean_image.to(device)
        from segmentation.segmentnetarch import SegmentationNet
        segmodel = SegmentationNet(retrain_off_seg=args.retrain_off_seg, mean_image = seg_mean_image)
        segmodel = segmodel.to(device)
        if not args.gpu:
            if (torch.cuda.device_count() > 1):
                    segmodel = nn.DataParallel(segmodel)
        if args.segload:
            if os.path.isfile(args.segload):
                checkpoint = torch.load(args.segload, map_location=torch.device('cuda:'+args.gpu))
                segmodel.load_state_dict(checkpoint)
                print("Loaded Checkpoint: ",args.segload)
            else:
                print('No checkpoint found at: ',args.segload)

    if args.seg is True:
        print('Model Device: ', next(model.parameters()).device, next(segmodel.parameters()).device)
    else:
        print('Model Device: ', next(model.parameters()).device)


    #Create Optimizer
    learning_rate = args.learning_rate
    learn_rate_decay = np.power(1e-3,1/float(args.epochs))#0.9991#10 / args.epochs
    if not args.retrain_off_seg:
        optimizer = optim.AdamW(list(model.parameters()) + list(segmodel.parameters()), lr = args.learning_rate, weight_decay=1e-2)
    else:
        optimizer = optim.AdamW(list(model.parameters()), lr = args.learning_rate, weight_decay=1e-2)
    #optimizer = optim.SGD(list(model.parameters()) + list(segmodel.parameters()), lr=args.learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=learn_rate_decay)
    #ReduceLROnPlateau is an interesting idea
    #loss_mult = torch.tensor([[0.8, 0.5, 0.3],[0.8, 0.5, 0.3]]).to(device)
    loss_mult = torch.tensor([[1, 1, 1],[1, 1, 1]]).to(device)

    #Save Parameters
    save_variables_divider = args.save_variables #5 #10
    log_path = '/mnt/samsung/gabe/save_models'
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
    """if args.traj and False: #TODO Does nothing significant rn 
        plotting_data = dict()
        data_loc = copy.deepcopy(args.data)
        val_outputs_x, val_outputs_y, val_outputs_z, val_outputs_yaw, val_outputs_p, val_outputs_r = loadData(val_idx,dataclass.num_list,data_loc,model,dataclass.traj_list,args.real,dataclass,args)
        #print(len(val_idx))
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
    """
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

    ranges = np.array([np.arange(i*args.bins, (i+1)*args.bins) for i in range(args.num_controls)])

    print_ctr = int(args.print)

    for epoch in range(args.epochs):            
        
        print('\n\nEpoch: ', epoch)
        #Train
        #gc.collect()
        epoch_acc = [[[], []] for _ in range(args.num_pts)]
        extra_epoch_acc = [[[[], []] for _ in range(args.num_pts)] for i in range(args.num_controls-1)]
        acc_total = [[0., 0.] for _ in range(args.num_pts)]
        extra_acc_total = [[[0., 0.] for  p in range(args.num_pts)] for i in range(args.num_controls-1)]
        extra_elements = [0. for i in range(args.num_controls-1)]
        train_phase_accuracy = 0.
        elements = 0.
        loader = train_loader

        for ctr, batch in enumerate(loader):
            #image_batch = batch['image'].to(device).float()
            model = model.to(device)
            point_batch = batch['points'] #.to(device)
            optimizer.zero_grad()
            model.train()
            with torch.set_grad_enabled(True):
                #model = model.to(device)
                classifier_loss = None

                if args.seg is True:
                    imgs = batch['image'].to(device)
                    raw_imgs = batch['raw_image'].to(device)
                    segmodel = segmodel.to(device)
                    batch_images = None
                    for img_num in range(args.num_images):
                        start_raw_idx = img_num*(base_n_channels)
                        end_raw_idx = (img_num+1)*(base_n_channels)
                        start_idx = img_num*(n_channels)
                        end_idx = (img_num+1)*(n_channels)
                        segimages = run_seg(segmodel,raw_imgs[:, start_raw_idx:end_raw_idx, :, :])
                        t_batch_imgs =  torch.cat((imgs[:, start_idx:end_idx, :, :], segimages), 1)
                        if batch_images is None:
                            batch_images = t_batch_imgs
                        else:
                            batch_images = torch.cat((batch_images, t_batch_imgs), 1)
                else:
                    batch_images = batch['image'].to(device)

                if print_ctr > 0:
                    for i in range(print_ctr):
                        print("Printing: " + str(i))
                        saveImage(batch_images[i,:,:,:],args.data + "/batch" + str(i))
                        r = get_state_data(args.states,batch)
                        if r is not None:
                            states = r[i]
                            print("States: " + str(i))
                            print(states)
                        if args.num_controls > 1:
                            print("Phase: " + str(i))
                            print(batch['phase'][i])
                        print("Points: " + str(i))
                        print(point_batch[i])
                    print_ctr = 0


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
                if args.seg is True:
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
                    phase = np.zeros(b_size).astype(int)

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
                        loss_x[temp] += F.cross_entropy(logits[elements_accessed,0,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,0])
                        loss_y[temp] += F.cross_entropy(logits[elements_accessed,1,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,1])
                        loss_z[temp] += F.cross_entropy(logits[elements_accessed,2,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,2])
                        loss_yaw[temp] += F.cross_entropy(logits[elements_accessed,3,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,3])

                        if not args.yaw_only:
                            loss_p[temp] += F.cross_entropy(logits[elements_accessed,4,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,4])
                            loss_r[temp] += F.cross_entropy(logits[elements_accessed,5,temp,ranges[phase].T].T,(point_batch)[elements_accessed,temp,5])

                    point_batch = point_batch.to('cpu')
                    logits = logits.to('cpu')

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
                    logits = logits.to(device).double()
                    point_batch = point_batch.to(device).double()
                    if not args.yaw_only:
                        logits = logits.view(-1,args.num_controls,model.num_points*6)
                        temp_point_batch = point_batch.view(-1,model.num_points*6)
                    else:
                        logits = logits.view(-1,args.num_controls,model.num_points*4)
                        temp_point_batch = point_batch.view(-1,model.num_points*4)
                    elements_accessed = np.arange(b_size)
                    batch_loss = 0
                    if (args.num_controls > 1):
                        batch_loss += classifier_loss
                    if args.spherical:
                        temp_transform = sphericalToXYZ()
                        batch_loss += F.mse_loss(temp_transform(logits[elements_accessed,phase,:]),temp_point_batch)
                    else:
                        batch_loss += F.mse_loss(logits[elements_accessed,phase,:],temp_point_batch)

                    #if args.spherical:
                    #    temp_transform = sphericalToXYZ()
                    #    batch_loss = F.mse_loss(temp_transform(logits),point_batch)#TODO: make sure this works
                    #else:
                    #    batch_loss = F.mse_loss(logits, point_batch)
                    acc_list = acc_metric_regression(args, logits[elements_accessed,phase,:].cpu(), temp_point_batch.cpu(),phase,sphere_flag = args.spherical)
                    if args.num_controls > 1:
                        acc_list, extra_acc_list = acc_list

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

                del point_batch, logits, batch_images, batch
                if args.states != 0:
                    del states

                writer.add_scalar('train_loss',batch_loss,ctr+epoch*len(train_loader))
                if not args.regression:
                    writer.add_scalar('train_loss_x',torch.sum(torch.tensor(loss_x)),ctr+epoch*len(train_loader))
                    writer.add_scalar('train_loss_y',torch.sum(torch.tensor(loss_y)),ctr+epoch*len(train_loader))
                    writer.add_scalar('train_loss_z',torch.sum(torch.tensor(loss_z)),ctr+epoch*len(train_loader))
                    writer.add_scalar('train_loss_yaw',torch.sum(torch.tensor(loss_yaw)),ctr+epoch*len(train_loader))
                    if not args.yaw_only:
                        writer.add_scalar('train_loss_p',torch.sum(torch.tensor(loss_p)),ctr+epoch*len(train_loader))
                        writer.add_scalar('train_loss_r',torch.sum(torch.tensor(loss_r)),ctr+epoch*len(train_loader))

                for ii, acc in enumerate(acc_list):
                    writer.add_scalar('train_acc_'+str(ii),acc[0],ctr+epoch*len(train_loader))

                for ii in range(args.num_controls):
                    temp_ctr = len(np.argwhere(phase==ii))
                    for i in range(len(acc_total)):
                        for j in range(2):
                            if ii == 0:
                                if elements + temp_ctr == 0.:
                                    continue
                                acc_total[i][j] = ((elements * acc_total[i][j]) + (temp_ctr * acc_list[i][j]))/(elements + temp_ctr)
                            else:
                                if extra_elements[ii-1] + temp_ctr == 0.:
                                    continue
                                extra_acc_total[ii-1][i][j] = ((extra_elements[ii-1] * extra_acc_total[ii-1][i][j]) + (temp_ctr * extra_acc_list[ii-1][i][j]))/(extra_elements[ii-1] + temp_ctr)                               

                    if ii == 0:
                        elements += temp_ctr
                    else:
                        extra_elements[ii-1] += temp_ctr

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
                if args.seg is True:
                    imgs = batch['image'].to(device)
                    raw_imgs = batch['raw_image'].to(device)
                    segmodel = segmodel.to(device)
                    batch_images = None
                    for img_num in range(args.num_images):
                        start_raw_idx = img_num*(base_n_channels)
                        end_raw_idx = (img_num+1)*(base_n_channels)
                        start_idx = img_num*(n_channels)
                        end_idx = (img_num+1)*(n_channels)
                        segimages = run_seg(segmodel,raw_imgs[:, start_raw_idx:end_raw_idx, :, :])
                        t_batch_imgs =  torch.cat((imgs[:, start_idx:end_idx, :, :], segimages), 1)
                        if batch_images is None:
                            batch_images = t_batch_imgs
                        else:
                            batch_images = torch.cat((batch_images, t_batch_imgs), 1)
                else:
                    batch_images = batch['image'].to(device)

                #print(segimages.shape, batch_imgs.shape)
                # model = model.to(device)
                model = model.to(device)
                model.eval()
                # logits = model(batch_images)
                if args.states == 0:
                    logits = model(batch_images)
                else:
                    states = get_state_data(args.states, batch)
                    states = states.to(device)
                    logits = model(batch_images, states)

                b_size = logits.shape[0]

                if args.num_controls > 1:
                    classifier = logits[:, :model.classifier].reshape(-1, model.classifier).to(device)
                    logits = logits[:, model.classifier:]
                    val_classifier_loss = F.cross_entropy(classifier, batch['phase'].long().to(device))
                    temp_accuracy = phase_accuracy(batch['phase'], classifier)
                    val_phase_accuracy += temp_accuracy
                    phase = np.array(batch['phase'].to('cpu'))
                else:
                    phase = np.zeros(b_size).astype(int)

                point_batch = batch['points'].to(device)
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
                    logits = logits.to(device).double()
                    point_batch = point_batch.to(device).double()
                    if not args.yaw_only:
                        logits = logits.view(-1,args.num_controls,model.num_points*6)
                        temp_point_batch = point_batch.view(-1,model.num_points*6)
                    else:
                        logits = logits.view(-1,args.num_controls,model.num_points*4)
                        temp_point_batch = point_batch.view(-1,model.num_points*6)
                    elements_accessed = np.arange(b_size)
                    val_loss = [0]
                    if (args.num_controls > 1):
                        val_loss[0] += val_classifier_loss
                    if args.spherical:
                        temp_transform = sphericalToXYZ()
                        val_loss[0] += F.mse_loss(temp_transform(logits[elements_accessed,phase,:]),temp_point_batch)
                    else:
                        val_loss[0] += F.mse_loss(logits[elements_accessed,phase,:],temp_point_batch)

                    val_acc_list = acc_metric_regression(args, logits[elements_accessed,phase,:].cpu(), temp_point_batch.cpu(),phase,sphere_flag = args.spherical)
                    if args.num_controls > 1:
                        val_acc_list, extra_val_acc_list = val_acc_list
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

                del point_batch, logits, batch_images, batch
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

        if val_classifier_loss is None:
            writer.add_scalar('val_loss',sum(val_loss),(epoch+1)*len(val_loader))
        else:
            writer.add_scalar('val_loss',sum(val_loss)+val_classifier_loss,(epoch+1)*len(val_loader))
        if not args.regression:
            if not args.yaw_only:
                val_loss = [val_loss[temp].item()/len(val_loader) for temp in range(6)]
            else:
                val_loss = [val_loss[temp].item()/len(val_loader) for temp in range(4)]
            writer.add_scalar('val_loss_x',val_loss[0],(epoch+1)*len(val_loader))
            writer.add_scalar('val_loss_y',val_loss[1],(epoch+1)*len(val_loader))
            writer.add_scalar('val_loss_z',val_loss[2],(epoch+1)*len(val_loader))
            writer.add_scalar('val_loss_yaw',val_loss[3],(epoch+1)*len(val_loader))
            if not args.yaw_only:
                writer.add_scalar('val_loss_p',val_loss[4],(epoch+1)*len(val_loader))
                writer.add_scalar('val_loss_r',val_loss[5],(epoch+1)*len(val_loader))
        for ii, acc in enumerate(val_acc):
            writer.add_scalar('val_acc_'+str(ii),acc[0],(epoch+1)*len(val_loader))
        print('Val Cross-Entropy Loss: ', val_loss)
        print('Val Accuracy: ',val_acc)
        if args.num_controls > 1:
            print("Val Phase Loss:", val_classifier_loss)
            print("Val Phase Accuracy:", val_phase_accuracy)
            print("Val Extra Accuracy:", extra_val_acc)
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
