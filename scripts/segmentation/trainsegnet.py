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
import PIL.Image as Image
import signal
import sys
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle
#print("summ")
#from torch.utils.tensorboard import SummaryWriter 
#print("gc")
import gc
import random
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
from numpy.linalg import norm
from customSegmentDataset import *

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
        name = save_path+'/model_seg' + str(i) + '.pth.tar'
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


def val_2_img(val, loc, epoch, k, imgs, seg, mean):
    #print("val: ", val.shape)
    floc = loc + "/epoch" + str(epoch) + "/"
    if not os.path.isdir(floc):
        os.makedirs(floc)

    images = torch.max(val, 1).indices
    #print("max shape: ", images.shape)
    for i in range(images.shape[0]):
        img = images[i, :, :]
        #print("single img: ", img.shape)
        img = (np.array(img) * 255).astype(np.uint8)
        #print(img.shape, np.max(img), np.min(img), np.sum(np.where(img==255, 1, 0)))
        im = Image.fromarray(img.astype(np.uint8))
        im.save(floc + "/output" + str(k + i) + ".png")

        c_img = np.array(imgs[i, :, :, :])
        c_img = np.transpose(c_img, [1, 2, 0])
        c_img += mean
        c_img *= 255
        #print(c_img.shape, np.max(c_img), np.min(c_img))
        c_im = Image.fromarray(c_img.astype(np.uint8))
        c_im.save(floc + "/img" + str(k+i) + ".png")

        s_img = np.array(seg[i, :, :])*255
        #print(s_img.shape, np.max(s_img), np.min(s_img))
        s_im = Image.fromarray(s_img.astype(np.uint8))
        s_im.save(floc + "/seg" + str(k+i) + ".png")
        #exit(0)


def main():
    signal.signal(signal.SIGINT,signal_handler)
    global model
    global save_path
    global writer
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='image data folder')
    parser.add_argument('seg', help='seg data folder')
    parser.add_argument('--load', help='model to load')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--resample', action='store_true', help='resample data')
    parser.add_argument('--gpu', help='gpu to use')
    #parser.add_argument('--num_images', type=int, default=1, help='number of input images')
    parser.add_argument('--batch_size', type=int, default=75, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='batch size')
    #parser.add_argument('--capacity', type=float, default=1, help='network capacity')
    parser.add_argument('--test_arch', type=int, default=100, help='testing architectures')
    parser.add_argument('--train',type=bool, default=False, help='add train aug') #TODO AUGMENTATION A LOT
    parser.add_argument('--val', type=float, default=0.10, help='validation percentage')
    parser.add_argument('-j', type=int, default=4, help='number of loader workers')
    parser.add_argument('--save_variables',type=int,default=2,help="save after every x epochs")
    parser.add_argument('--save_outputs',type=int,default=5,help="save after every x epochs")
    args = parser.parse_args()

    num_channels = 3
    mean = np.load("data/depth_data/data/mean_imgv2_data_real_world_traj_bag.npy")
    print(mean.shape)
    if args.test_arch == 100:
        from segmentnetarch import SegmentationNet
    else:
        import importlib
        i = importlib.import_module('architecture.segmentnetarch' + str(args.test_arch))
        SegmentationNet = i.SegmentationNet

    if args.train:
        print("Image transform set")
        #img_trans = transforms.Compose([RandomHorizontalTrajFlip(p=0.5)])
    else:
        img_trans = None

    dataclass = SegmentationDataset(args.img, args.seg)

    val_perc = args.val
    np.random.seed(args.seed)
    random.seed(args.seed)

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
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.j, worker_init_fn=dl_init)
    #weight_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.j, worker_init_fn=dl_init)
    val_loader = DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.j)

    print ('Training Samples: ' + str(len(train_data)))
    print ('Validation Samples: ' + str(len(val_data)))

    model = SegmentationNet()

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

    learning_rate = args.learning_rate
    learn_rate_decay = np.power(1e-3,1/float(args.epochs))#0.9991#10 / args.epochs
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay=1e-2)
    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=learn_rate_decay)

    save_variables_divider = args.save_variables
    log_path = './model/segmentation/logs'
    save_path = createStampedFolder(os.path.join(log_path, 'variable_log'))
    tensorboard_path = addTimestamp(os.path.join(log_path, 'tensorboard_'))
    image_path = addTimestamp(os.path.join(log_path, 'image_'))
    print(image_path)
    os.makedirs(image_path)

    print ('Training...')
    writer = SummaryWriter(tensorboard_path)
    #val_writer = SummaryWriter(val_path)
    #graph_writer = Su
    #os.makedirs(plot_data_path)

    since = time.time()

    for epoch in range(args.epochs):
        model = model.to(device)
        print('Epoch: ', epoch)

        loader = train_loader
        total_loss = torch.Tensor([0.]).to(device)

        for ctr, batch in enumerate(loader):
            seg_batch = batch['segmented'].long().to(device)
            optimizer.zero_grad()
            model.train()
            with torch.set_grad_enabled(True):
                batch_imgs = batch['image']
                batch_imgs = batch_imgs.to(device)
                #model = model.to(device)
                logits = model(batch_imgs)
                #logits = torch.transpose(torch.transpose(logits, 1, 2), 2, 3)
                #print(logits.shape)
                logits = logits.view(-1,2,model.h,model.w)
                #print(logits.shape)
                loss = 0.
                seg_batch = seg_batch.long()
                #print(seg_batch.shape)
                loss = F.cross_entropy(logits, seg_batch)

                seg_batch = seg_batch.to('cpu')
                logits = logits.to('cpu')

                total_loss[0] += loss

                loss.backward()
                optimizer.step()

                writer.add_scalar('train_loss',loss,ctr+epoch*len(train_loader))

        print("Training Loss: ", total_loss.item())

        val_loss = torch.Tensor([0.]).to(device)

        for ctr, batch in enumerate(val_loader):
            with torch.set_grad_enabled(False):
                seg_batch = batch['segmented'].long().to(device)
                model = model.to(device)
                model.eval()
                batch_imgs = batch['image']
                batch_imgs = batch_imgs.to(device)
                #model = model.to(device)
                logits = model(batch_imgs)
                #logits = logits.view(-1,model.h*model.w)
                #logits = torch.transpose(torch.transpose(logits, 1, 2), 2, 3)
                logits = logits.view(-1,2, model.h,model.w)

                loss = 0.
                seg_batch = seg_batch.long()

                loss = F.cross_entropy(logits, seg_batch)

                seg_batch = seg_batch.to('cpu')
                logits = logits.to('cpu')

                val_loss[0] += loss

                #loss.backward()
                #optimizer.step()
                if epoch % args.save_outputs == 0:
                    batch_imgs = batch_imgs.to('cpu')
                    seg_batch = seg_batch.to('cpu')
                    val_2_img(logits, image_path, epoch, ctr*logits.shape[0], batch_imgs, seg_batch, mean)

                writer.add_scalar('val_loss',loss,ctr+epoch*len(train_loader))

        print("Validation Loss: ", val_loss.item())
        writer.add_scalar('val_loss',val_loss,(epoch+1)*len(train_loader))
        scheduler.step()
        print('Learning Rate Set to: ',scheduler.get_lr())
        # Save variables
        if ((epoch + 1) % save_variables_divider == 0 or (epoch == 0) or (epoch == args.epochs - 1)):
            print("Saving variables")
            save_model(epoch)

    writer.close()
    print("Done")


if __name__ == '__main__':
    main()

