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
import argparse
import PIL.Image as img
import signal
import sys
import torch.utils.data import Dataset, DataLoader
from orangeTransforms import *
from torch.utils.tensorboard import SummaryWriter


class SubSet(Dataset):
    def __init__(self,dataset,idx):
        self.ds = dataset
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self,i):
        return self.ds[self.idx[i]]

class OrangeDataSet(Dataset):
    def __init__(self, root_dir, num_images, num_pts, pt_trans, img_trans, dt = 1, reduce_N = True):
        self.point_transform = pt_trans
        self.image_transform = img_trans
        self.num_pts = num_pts
        self.num_images = num_images
        self.dt = dt
        self.num_list = []
        self.traj_list = []
        self.run_dir = root_dir
        self.trial_list = os.listdir(root_dir)
        self.num_samples = 0
        if reduce_N:
            time_window = num_pts*dt
        else:
            time_window = 0
        for trial_dir in trial_list:
            with open(run_dir+'/'+trial_dir+'/data.pickle','rb') as f:
                data = pickle.load(f,encoding='latin1')
                self.traj_list.append(data)
            with open(run_dir+"/"+trial_dir+"/metadata.pickle",'rb') as data_f:
                data = pickle.load(data_f)#, encoding='latin1')
                N = data['N']
                if reduce_N:
                    tf = data['tf']
                    h = float(N)/tf
                    reduced_N = int(N - time_window*h)
                    self.num_samples += reduced_N
                    self.num_list.append(reduced_N)
                else:
                    self.num_samples += N
                    self.num_list.append(N)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,i):
        idx = i
        trial_idx = 0
        while True:
            if num_list[trial_idx] <= idx:
                idx = idx-num_list[trial_idx]
                trial_idx += 1
            else:
                break
        trial = self.trial_list[trial_idx]
        image = None
        image_idx = idx
        for ii in range(self.num_images):
            temp_idx = max(0,image_idx - ii)
            img_run_dir = run_dir.rstrip("/") + "_np"
            temp_image = np.load(imag_run_dir+'/'+trial+'/image'+str(temp_idx)+'.npy')
            if self.image_transform:
                temp_image = self.image_transform(temp_image)
            if image is None:
                image = temp_image
            else:
                image = np.concatenate((image,temp_image),axis=2)
        points = self.traj_list[trial_idx][idx]
        if self.point_transform:
           points = self.point_transform(points)
        return {'image':image, 'points':points}

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
    name = save_path+'/model' + str(i) + '.pth.tar'
    torch.save(model.state_dict(),save_path)

def signal_handler(signal, frame):
    print('Closing....')
    save_model()
    writer.close()
    print('Done')
    sys.exit(0)

def acc_metric(model,logits,point_batch):
    shape = logits.size().item()
    acc_list = []
    for pt in range(shape[2]):
        batch_list = []
        for ii in range(shape[0]):
            coord_list = []
            for coord in range(3):
                prob = nn.Softmax(logits[ii,coord,pt,:]).numpy()
                max_pred = np.argmax(prob)
                true_pred = np.argmax(point_batch[ii,pt,coord,:])
                bin_size = (model.max_list[pt][coord] - model.min_list[pt][coord])/model.bins
                d = (true_pred - max_pred)*bin_size
                coord_list.append(d)
            d = np.vstack([coord_list[0],coord_list[1],coord_list[2]])
            batch_list.append(np.linalg.norm(d))
        batch_mean = np.mean(batch_list)
        acc_list.append(batch_mean)
    return acc_list

def main():
    signal.signal(signal.SIGINT,signal_handler)
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
    args = parser.parse_args()
    args.min = [(0,-0.5,-0.1),(0,-1,-0.15),(0,-1.5,-0.2),(0,-2,-0.3),(0,-3,-0.5)]
    args.max = [(1,0.5,0.1),(2,1,0.15),(4,1.5,0.2),(6,2,0.3),(7,0.3,0.5)]
    
    #Data Transforms
    pt_trans = transforms.Compose([pointToBins(args.min,args.max,args.bins),GaussLabels(1,0,args.bins)])
    #Load Mean image
    data_loc = copy.deepcopy(args.data)
    data_loc_name = data_loc.strip("..").strip(".").strip("/").replace("/", "_")
    mean_img_loc = data_loc + "../mean_imgv2_" + data_loc_name + '.npy' 
    if not (os.path.exists(mean_img_loc)):
        print('mean image file not found')
        mean_image = compute_mean_image(train_indices, data_loc, model)
        np.save(mean_img_loc, mean_image)
    else:
        print('mean image file found')
        mean_image = np.load(mean_img_loc)
  # mean_image = np.zeros((model.w, model.h, 3))
    img_trans = None #TODO: Change this
    #Create dataset class
    dataclass = OrangeDataSet(args.data, args.num_images, args.num_pts, pt_trans, img_trans)
    
    #Break up into validation and training
    val_perc = 0.07
    np.random.seed(args.seed)
    if args.resample:
        rand_idx = np.random.choice(len(dataclass),size=len(dataclass),replace=True)
    else:
        rand_idx = np.random.permutation(len(dataclass))
    val_idx = np.ceil(len(dataclass)*val_perc).astype(int)
    train_idx = len(dataclass) - val_idx
    val_idx = rand_idx[-val_idx:]
    train_idx = rand_idx[:train_idx]
    train_data = SubSet(dataclass,train_idx)
    val_data = SubSet(dataclass,val_idx)
    
    #Create DataLoaders
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.j)
    val_loader = DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.j)
    print ('Training Samples: ' + str(len(train_data))
    print ('Validation Samples: ' + str(len(val_data))

    #Create Model
    model = OrangeNet(args.capacity,args.num_images,args.num_pts,args.bins)
    if args.load:
        if os.path.isfile(args.load):
            checkpoint = torch.load(args.load)
            model.load_state_dict(checkpoint)
            print("Loaded Checkpoint: ",args.load)
        else:
            print('No checkpoint found at: ',args.load)
    #CUDA Check
    use_cuda = torch.cuda.is_available()
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
    
    #Create Optimizer
    learning_rate = args.learning_rate
    learn_rate_decay = 100 / num_epochs
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=learn_rate_decay)
    #ReduceLROnPlateau is an interesting idea

    #Save Parameters
    save_variables_divider = 10
    log_path = './model/logs'
    save_path = createStampedFolder(os.path.join(log_path, 'variable_log'))
    tensorboard_path = addTimestamp(os.path.join(log_path, 'tensorboard_'))
    #val_path = addTimestamp(os.path.join(log_path, 'validation_'))
    plot_data_path = addTimestamp(os.path.join(log_path, 'plot_data_'))

    print ('Training...')
    writer = SummaryWriter(tensorboard_path)
    #val_writer = SummaryWriter(val_path)
    #graph_writer = Su
    #os.makedirs(plot_data_path)

    #print ('Writers Set Up')

    #iters = 0
    #plotting_data = dict()
    #plotting_data['idx'] = range(5)
    #plotting_data['truth'] = [val_outputs_x[plotting_data['idx']],
    #                          val_outputs_y[plotting_data['idx']],
    #                          val_outputs_z[plotting_data['idx']]]
    #plotting_data['data'] = list()
    #plotting_data['foc_l'] = args.cam_coord
    #plotting_data['min'] = model.min
    #plotting_data['max'] = model.max
    #plotting_data['bins'] = model.bins
    #for ii in plotting_data['idx']:
    #  plotting_data['data'].append([])
    #print(plotting_data)
    loss = nn.CrossEntropyLoss()
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        #Train
        for ctr, batch in enumerate(train_loader):
            image_batch = batch['image'].to(device)
            point_batch = batch['points'].to(device)
            optimizer.zero_grad()
            model.train()
            with torch.set_grad_enabled(True):
                logits = model(image_batch)
                logits = logits.view(-1,3,model.num_points,model.bins)
                logits_x = logits[:,0,:,:]
                logits_y = logits[:,1,:,:]
                logits_z = logits[:,2,:,:]
                #TODO: Check loss syntax
                loss_x = sum([loss(logits_x[:,temp,:],point_batch[:,temp,0]) for temp in range(model.num_points)])
                loss_y = sum([loss(logits_y[:,temp,:],point_batch[:,temp,1]) for temp in range(model.num_points)])
                loss_z = sum([loss(logits_z[:,temp,:],point_batch[:,temp,2]) for temp in range(model.num_points)])
                batch_loss = loss_x + loss_y + loss_z
                batch_loss.backward()
                optimizer.step()
                writer.add_scalar('train_loss',batch_loss,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                writer.add_scalar('train_loss_x',loss_x,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                writer.add_scalar('train_loss_y',loss_y,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                writer.add_scalar('train_loss_z',loss_z,ctr+epoch*len(train_loader))#TODO: possibly loss.item
                acc_list = acc_metric(logits,point_batch)
                for ii, acc in enumerate(acc_list):
                    writer.add_scalar('train_acc_'+str(ii),acc,ctr+epoch*len(train_loader))
        #Validation
        val_loss = [0,0,0]
        val_acc = np.zeros(model.num_points)
        for batch in val_loader:
            with torch.set_grad_enabled(False):
                image_batch = batch['image'].to(device)
                point_batch = batch['points'].to(device)
                model.eval()
                logits = model(image_batch)
                logits = logits.view(-1,3,model.num_points,model.bins)
                logits_x = logits[:,0:model.bins]
                logits_y = logits[:,model.bins:2*model.bins]
                logits_z = logits[:,2*model.bins:3*model.bins]
                #TODO: Check loss syntax
                val_loss[0] += sum([loss(logits_x[:,temp,:],point_batch[:,temp,0]) for temp in range(model.num_points)])
                val_loss[1] += sum([loss(logits_y[:,temp,:],point_batch[:,temp,1]) for temp in range(model.num_points)])
                val_loss[2] += sum([loss(logits_y[:,temp,:],point_batch[:,temp,2]) for temp in range(model.num_points)])
                val_acc_list = acc_metric(logits,point_batch)
                for ii, acc in enumerate(val_acc_list):
                    val_acc[ii] += acc
        val_loss = [val_loss[temp]/len(val_loader) for temp in range(3)]
        val_acc = val_acc/len(val_loader)
        writer.add_scalar('val_loss',sum(val_loss),(epoch+1)*len(train_loader))#TODO: possibly loss.item
        writer.add_scalar('val_loss_x',val_loss[0],(epoch+1)*len(train_loader))#TODO: possibly loss.item
        writer.add_scalar('val_loss_y',val_loss[1],(epoch+1)*len(train_loader))#TODO: possibly loss.item
        writer.add_scalar('val_loss_z',val_loss[2],(epoch+1)*len(train_loader))#TODO: possibly loss.item
        for ii, acc in enumerate(val_acc):
            writer.add_scalar('val_acc_'+str(ii),acc,(epoch+1)*len(train_loader))
        #Adjust LR
        scheduler.step()
        print('Learning Rate Set to: ',scheduler.get_lr())
        # Save variables
        if ((epoch + 1) % save_variables_divider == 0 or (epoch == 0) or (epoch == num_epochs - 1)):
            print("Saving variables")
            save_model(epoch)
    writer.close()
    print("Done")

if __name__ == '__main__':
    main()
