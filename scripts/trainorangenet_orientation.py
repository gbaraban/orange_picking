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
import PIL.Image as img
import signal
import sys
from torch.utils.data import Dataset, DataLoader
from customTransforms import *
#from orangenetarch import *
import pickle
#from customRealDatasets import *
#print("summ")
#from torch.utils.tensorboard import SummaryWriter 
#print("gc")
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

def parseFiles(idx,num_list,run_dir,model,traj_data,real,dataclass):
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
  if real:
    points = traj_data[trial_idx][idx]
  else:
    p0 = np.array(traj_data[trial_idx][idx][0])
    R0 = np.array(traj_data[trial_idx][idx][1])
    points = []
    if dataclass.custom_dataset is None:
        for pt in range(dataclass.num_pts):
            temp_idx = int(idx + dataclass.h[trial]*dataclass.dt*(pt+1))
            p = traj_data[trial_idx][temp_idx][0]
            Ri = traj_data[trial_idx][temp_idx][1]
            p = np.array(p)
            p = list(np.matmul(R0.T,p-p0))
            Ri = np.matmul(R0.T,Ri)
            Ri_zyx = list(R.from_dcm(Ri).as_euler('zyx'))
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
            Ri_zyx = list(R.from_dcm(Ri).as_euler('zyx'))
            p.extend(Ri_zyx)
            points.append(p)



  local_pts = []
  ctr = 0
  #print(points)
  #exit()
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

def loadData(idx,num_list,run_dir,model,traj_data,real,dataclass):
  waypoints_x = []
  waypoints_y = []
  waypoints_z = []
  waypoints_r = []
  waypoints_p = []
  waypoints_yaw = []

  for ii in idx:
    waypoint = parseFiles(ii,num_list,run_dir,model,traj_data,real,dataclass)
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
	Rot0 = R.from_euler('zyx', [true_angle[0], true_angle[1], true_angle[2]]) #[np.pi/3, np.pi/6, np.pi/4])
	R0 = Rot0.as_dcm()

	Rot1 = R.from_euler('zyx', [pred_angle[0], pred_angle[1], pred_angle[2]]) #[np.pi/3 + 1, np.pi/6, np.pi/4])
	R1 = Rot1.as_dcm()

	R2 = np.matmul(R0.T,R1)
	logR2 = logm(R2)
	dist = norm(skew2axis(logR2))

	return dist

def acc_metric(args,logits,point_batch,yaw_only):
    #TODO: Add diff accuracy for angles
    softmax = nn.Softmax(dim=0)
    shape = logits.size()#.item()
    acc_list = []
    logits = logits.detach()
    for pt in range(shape[2]):
        batch_list = []
        ang_d = 0
        for ii in range(shape[0]):
            coord_list = []
            for coord in range(3):
                prob = np.array(softmax(logits[ii,coord,pt,:]))#.numpy()
                max_pred = np.argmax(prob)
                #true_pred = np.argmax(point_batch[ii,pt,coord,:])
                true_pred = np.array(point_batch[ii,pt,coord])
                bin_size = (args.max[pt][coord] - args.min[pt][coord])/args.bins
                d = (true_pred - max_pred)*bin_size
                coord_list.append(d)
            d = np.vstack([coord_list[0],coord_list[1],coord_list[2]])
            batch_list.append(np.linalg.norm(d))

            if not yaw_only:
                true_angle = []
                pred_angle = []
                for coord in range(3,6):
                    prob = np.array(softmax(logits[ii,coord,pt,:]))#.numpy()
                    max_pred = np.argmax(prob)
                    #true_pred = np.argmax(point_batch[ii,pt,coord,:])
                    true_pred = np.array(point_batch[ii,pt,coord])
                    bin_size = (args.max[pt][coord] - args.min[pt][coord])/args.bins
                    true_angle.append(true_pred * bin_size)
                    pred_angle.append(max_pred * bin_size)

                ang_d = angle_dist(true_angle, pred_angle)

            else:
                prob = np.array(softmax(logits[ii,3,pt,:]))#.numpy()
                max_pred = np.argmax(prob)
                #true_pred = np.argmax(point_batch[ii,pt,coord,:])
                true_pred = np.array(point_batch[ii,pt,3])
                bin_size = (args.max[pt][coord] - args.min[pt][coord])/args.bins
                true_angle = (true_pred * bin_size)
                pred_angle = (max_pred * bin_size)

                ang_d = norm((true_angle - pred_angle))

        batch_mean = np.mean(batch_list)
        batch_mean = [batch_mean, ang_d]
        acc_list.append(batch_mean)
    return acc_list

def main():
    signal.signal(signal.SIGINT,signal_handler)
    global model
    global save_path
    global writer
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data folder')
    parser.add_argument('--load', help='model to load')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--resample', action='store_true', help='resample data')
    parser.add_argument('--gpu', help='gpu to use')
    parser.add_argument('--num_images', type=int, default=1, help='number of input images')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='batch size')
    parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
    parser.add_argument('--capacity', type=float, default=1, help='network capacity')
    parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
    parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
    parser.add_argument('--bins', type=int, default=30, help='number of bins per coordinate')
    parser.add_argument('-j', type=int, default=4, help='number of loader workers')
    parser.add_argument('--traj', type=int, default=0, help='train trajectories')
    parser.add_argument('--real', type=int, default=0, help='real world imgs')
    parser.add_argument('--val', type=float, default=0.10, help='validation percentage')
    parser.add_argument('--resnet18', type=int, default=0, help='ResNet18')
    parser.add_argument('--yaw_only', type=int, default=0, help='yaw only')
    parser.add_argument('--custom', type=str, default="", help='custom parser')
    parser.add_argument('--test_arch', type=int, default=100, help='testing architectures')
    parser.add_argument('--train', type=int, default=1, help='train or test')
    parser.add_argument('--data_aug_flip',type=int,default=1, help='Horizontal flipping on/off')
    parser.add_argument('--real_test',type=int,default=0,help='self.h = 380/480')
    parser.add_argument('--freeze',type=str,default="",help='layer to freeze while training, linear or conv')
    args = parser.parse_args()

    if args.custom == "":
        args.custom = None

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
        args.real = False
    else:
        from customRealDatasetsOrientation import OrangeSimDataSet, SubSet
        args.real = True

    if args.test_arch == 100:
        from orangenetarch import OrangeNet8, OrangeNet18
    else:
        import importlib
        i = importlib.import_module('architecture.orangenetarch' + str(args.test_arch))
        OrangeNet8 = i.OrangeNet8
        OrangeNet18 = i.OrangeNet18

    if args.real_test == 0:
        args.real_test = False
    else:
        args.real_test = True

    args.min = [(0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0,-1,-0.15,-np.pi,-np.pi/2,-np.pi),(0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0,-2,-0.3,-np.pi,-np.pi/2,-np.pi),(0,-3,-0.5,-np.pi,-np.pi/2,-np.pi)]
    args.max = [(1,0.5,0.1,np.pi,np.pi/2,np.pi),(2,1,0.15,np.pi,np.pi/2,np.pi),(4,1.5,0.2,np.pi,np.pi/2,np.pi),(6,2,0.3,np.pi,np.pi/2,np.pi),(7,0.3,0.5,np.pi,np.pi/2,np.pi)]
    #args.traj = False
    #Data Transforms
    pt_trans = transforms.Compose([pointToBins(args.min,args.max,args.bins)])#,GaussLabels(1,1e-10,args.bins)])

    if args.train == 1 and args.data_aug_flip == 1:
        img_trans = transforms.Compose([RandomHorizontalTrajFlip(p=0.5)])
    else:
        img_trans = None

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
    dataclass = OrangeSimDataSet(args.data, args.num_images, args.num_pts, pt_trans, img_trans, custom_dataset=args.custom)

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
        val_order = np.ceil(len(dataclass.num_samples_dir_size)*val_perc).astype(int)
        val_indices = []
        train_indices = []
        val_data = {}
        val_data["order"] = np.array(random.sample(list(dataclass.num_samples_dir_size.keys()), k=val_order))
        #print(val_data["order"])

        for x in val_data["order"]:
            val_indices.extend(list(range(dataclass.num_samples_dir[x]['start'], dataclass.num_samples_dir[x]['end'])))
            val_data[x] = dataclass.num_samples_dir[x]

        for i, x in enumerate(list(dataclass.num_samples_dir_size.keys())):
            if x not in val_data["order"]:
                train_indices.extend(list(range(dataclass.num_samples_dir[x]['start'], dataclass.num_samples_dir[x]['end'])))

        val_idx = len(val_indices)
        train_idx = dataclass.num_samples - val_idx

        random.shuffle(train_indices)

        val_idx = np.array(val_indices)
        train_idx = np.array(train_indices)

        fopen = open('val_data_xyz.pickle', 'wb')
        pickle.dump(val_data, fopen, pickle.HIGHEST_PROTOCOL)

    train_data = SubSet(dataclass,train_idx)
    val_data = SubSet(dataclass,val_idx)

    #Create DataLoaders
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.j, worker_init_fn=dl_init)
    val_loader = DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.j)
    print ('Training Samples: ' + str(len(train_data)))
    print ('Validation Samples: ' + str(len(val_data)))

    #Create Model
    if args.yaw_only:
        n_outputs = 4
    else:
        n_outputs = 6

    if not args.resnet18:
        model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=args.real_test,retrain_off=args.freeze)
    else:
        model = OrangeNet18(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=n_outputs,real_test=args.real_test)

    if args.load:
        if os.path.isfile(args.load):
            checkpoint = torch.load(args.load)
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
    print('Model Device: ', next(model.parameters()).device)
    
    #Create Optimizer
    learning_rate = args.learning_rate
    learn_rate_decay = np.power(1e-3,1/float(args.epochs))#0.9991#10 / args.epochs
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=learn_rate_decay)
    #ReduceLROnPlateau is an interesting idea

    #Save Parameters
    save_variables_divider = 10
    log_path = './model/logs'
    save_path = createStampedFolder(os.path.join(log_path, 'variable_log'))
    tensorboard_path = addTimestamp(os.path.join(log_path, 'tensorboard_'))
    #if args.traj:
    #    val_path = addTimestamp(os.path.join(log_path, 'validation_'))
    plot_data_path = addTimestamp(os.path.join(log_path, 'plot_data_'))

    print ('Training...')
    writer = SummaryWriter(tensorboard_path)
    #val_writer = SummaryWriter(val_path)
    #graph_writer = Su
    os.makedirs(plot_data_path)

    #print ('Writers Set Up')

    #iters = 0
    if args.traj:
        plotting_data = dict()
        data_loc = copy.deepcopy(args.data)
        val_outputs_x, val_outputs_y, val_outputs_z, val_outputs_yaw, val_outputs_p, val_outputs_r = loadData(val_idx,dataclass.num_list,data_loc,model,dataclass.traj_list,args.real,dataclass)
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
    for epoch in range(args.epochs):
        model = model.to(device)
        print('Epoch: ', epoch)
        #Train
        #gc.collect()
        acc_total = [[0., 0.], [0., 0.], [0., 0.]]
        elements = 0.
        for ctr, batch in enumerate(train_loader):
            #print('Batch: ',ctr)
            #image_batch = batch['image'].to(device).float()
            point_batch = batch['points']#.to(device)
            optimizer.zero_grad()
            model.train()
            #for obj in gc.get_objects():
            #    try:
            #        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #            print(type(obj), obj.size())
            #    except:
            #        pass
            with torch.set_grad_enabled(True):
                batch_imgs = batch['image']
                batch_imgs = batch_imgs.to(device)
                #model = model.to(device)
                logits = model(batch_imgs)
                #print(logits.size())
                del batch_imgs
                del batch
                if not args.yaw_only:
                    logits = logits.view(-1,6,model.num_points,model.bins)
                else:
                    logits = logits.view(-1,4,model.num_points,model.bins)
                #logits_x = logits[:,0,:,:]
                #logits_y = logits[:,1,:,:]
                #logits_z = logits[:,2,:,:]
                loss_x = 0
                loss_y = 0
                loss_z = 0

                if not args.yaw_only:
                    loss_r = 0
                    loss_p = 0

                loss_yaw = 0

                b_size = logits.shape[0]
                point_batch = point_batch.to(device)
                for temp in range(model.num_points):
                    loss_x += F.cross_entropy(logits[:,0,temp,:],(point_batch)[:,temp,0])
                    loss_y += F.cross_entropy(logits[:,1,temp,:],(point_batch)[:,temp,1])
                    loss_z += F.cross_entropy(logits[:,2,temp,:],(point_batch)[:,temp,2])
                    loss_yaw += F.cross_entropy(logits[:,3,temp,:],(point_batch)[:,temp,3])

                    if not args.yaw_only:
                        loss_p += F.cross_entropy(logits[:,4,temp,:],(point_batch)[:,temp,4])
                        loss_r += F.cross_entropy(logits[:,5,temp,:],(point_batch)[:,temp,5])

                point_batch = point_batch.to('cpu')
                logits = logits.to('cpu')
                acc_list = acc_metric(args,logits,point_batch,args.yaw_only)
                del point_batch
                del logits

                if not args.yaw_only:
                    batch_loss = loss_x + loss_y + loss_z + loss_yaw + loss_p + loss_r
                else:
                    batch_loss = loss_x + loss_y + loss_z + loss_yaw

                batch_loss.backward()
                optimizer.step()
                #model = model.to('cpu')

                writer.add_scalar('train_loss',batch_loss,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                writer.add_scalar('train_loss_x',loss_x,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                writer.add_scalar('train_loss_y',loss_y,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                writer.add_scalar('train_loss_z',loss_z,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                writer.add_scalar('train_loss_yaw',loss_yaw,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                if not args.yaw_only:
                    writer.add_scalar('train_loss_p',loss_p,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                    writer.add_scalar('train_loss_r',loss_r,ctr+epoch*len(train_loader))#TODO: possibly loss.item

                for ii, acc in enumerate(acc_list):
                    writer.add_scalar('train_acc_'+str(ii),acc[0],ctr+epoch*len(train_loader))
                #print('Cross-Entropy Loss: ',batch_loss.item(),[loss_x.item(),loss_y.item(),loss_z.item()])
                #print('Training Accuracy: ',acc_list)
                #print(type(acc_list))

                for i in range(len(acc_total)):
                    for j in range(2):
                        acc_total[i][j] = ((elements * acc_total[i][j]) + (b_size * acc_list[i][j]))/(elements + b_size)

                elements += b_size

                #print(b_size)

                #image_batch = point_batch = logits = loss_x = loss_y = loss_z = batch_loss = None
            #del image_batch, point_batch, logits, loss_x, loss_y, loss_z, batch_loss
            #image_batch = image_batch.cpu()
            #point_batch = point_batch.cpu()
            #torch.cuda.empty_cache()
        #Validation
        #print("Reach here")
        print('Training Accuracy: ',acc_total)
        #exit()
        if args.yaw_only:
            val_loss = [0.,0.,0.,0.]
        else:
            val_loss = [0.,0.,0.,0.,0.,0.]

        val_acc = np.zeros((model.num_points,2))
        if not args.yaw_only:
            resnet_output = np.zeros((0, 6, args.num_pts, model.bins))
        else:
            resnet_output = np.zeros((0, 4, args.num_pts, model.bins))

        for batch in val_loader:
            with torch.set_grad_enabled(False):
                image_batch = batch['image'].to(device)
                model = model.to(device)
                model.eval()
                logits = model(image_batch)
                del image_batch
                if not args.yaw_only:
                    logits = logits.view(-1,6,model.num_points,model.bins)
                else:
                    logits = logits.view(-1,4,model.num_points,model.bins)
                logits_x = logits[:,0,:,:]
                logits_y = logits[:,1,:,:]
                logits_z = logits[:,2,:,:]
                logits_yaw = logits[:,3,:,:]
                if not args.yaw_only:
                    logits_p = logits[:,4,:,:]
                    logits_r = logits[:,5,:,:]

                point_batch = batch['points'].to(device)
                for temp in range(model.num_points):
                    val_loss[0] += F.cross_entropy(logits_x[:,temp,:],point_batch[:,temp,0])
                    val_loss[1] += F.cross_entropy(logits_y[:,temp,:],point_batch[:,temp,1])
                    val_loss[2] += F.cross_entropy(logits_z[:,temp,:],point_batch[:,temp,2])
                    val_loss[3] += F.cross_entropy(logits_yaw[:,temp,:],point_batch[:,temp,3])
                    if not args.yaw_only:
                        val_loss[4] += F.cross_entropy(logits_p[:,temp,:],point_batch[:,temp,4])
                        val_loss[5] += F.cross_entropy(logits_r[:,temp,:],point_batch[:,temp,5])

                val_acc_list = acc_metric(args,logits.cpu(),point_batch.cpu(), args.yaw_only)
                logits = logits.to('cpu')
                #print(logits.shape)
                resnet_output = np.concatenate((resnet_output,logits), axis=0)
                #print(resnet_output.shape)

                for ii, acc in enumerate(val_acc_list):
                    for jj, acc_j in enumerate(acc):
                        val_acc[ii][jj] += acc_j
                del point_batch
                model = model.to('cpu')

        if args.traj:
            for ii in plotting_data['idx']:
                plotting_data['data'][ii].append(resnet_output[ii,:,:,:])

            with open(plot_data_path+'/data.pickle','wb') as f:
                pickle.dump(plotting_data,f,pickle.HIGHEST_PROTOCOL)
        if not args.yaw_only:
            val_loss = [val_loss[temp].item()/len(val_loader) for temp in range(6)]
        else:
            val_loss = [val_loss[temp].item()/len(val_loader) for temp in range(4)]
        val_acc = val_acc/len(val_loader)
        writer.add_scalar('val_loss',sum(val_loss),(epoch+1)*len(train_loader))#TODO: possibly loss.item
        writer.add_scalar('val_loss_x',val_loss[0],(epoch+1)*len(train_loader))#TODO: possibly loss.item
        writer.add_scalar('val_loss_y',val_loss[1],(epoch+1)*len(train_loader))#TODO: possibly loss.item
        writer.add_scalar('val_loss_z',val_loss[2],(epoch+1)*len(train_loader))#TODO: possibly loss.item
        writer.add_scalar('val_loss_yaw',val_loss[3],(epoch+1)*len(train_loader))#TODO: possibly loss.item
        if not args.yaw_only:
            writer.add_scalar('val_loss_p',val_loss[4],(epoch+1)*len(train_loader))#TODO: possibly loss.item
            writer.add_scalar('val_loss_r',val_loss[5],(epoch+1)*len(train_loader))#TODO: possibly loss.item
        for ii, acc in enumerate(val_acc):
            writer.add_scalar('val_acc_'+str(ii),acc[0],(epoch+1)*len(train_loader))
        print('Val Cross-Entropy Loss: ',sum(val_loss),val_loss)
        print('Val Accuracy: ',acc_list)
        #Adjust LR
        scheduler.step()
        print('Learning Rate Set to: ',scheduler.get_lr())
        # Save variables
        if ((epoch + 1) % save_variables_divider == 0 or (epoch == 0) or (epoch == args.epochs - 1)):
            print("Saving variables")
            save_model(epoch)
    writer.close()
    print("Done")

if __name__ == '__main__':
    print("asdf")
    main()
