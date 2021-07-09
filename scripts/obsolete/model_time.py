import numpy as np
#from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img
from scipy.spatial.transform import Rotation as R
from orangenetarch import *
#from trainorangenet_orientation import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import argparse
#from plotting.parsetrajfile import *
#from orangesimulation import *
import pickle
import os
#import gcophrotor
#from torch.utils.tensorboard import SummaryWriter
from customTransforms import *
import time
import random


def load_run18(args):
	from customDatasetsOrientation import OrangeSimDataSet, SubSet
	custom = "Run18"
	traj = True
	data = "./data/Run20/"
	val_perc = 0 #TODO: adjust this for number of training trajectories, we are using train traj, so we want to adjust (1-val_perc)

	pt_trans = transforms.Compose([pointToBins(args.min, args.max, args.bins)])
	img_trans = transforms.Compose([RandomHorizontalTrajFlip()])

	dataclass = OrangeSimDataSet(data, args.num_images, args.num_pts, pt_trans, img_trans, custom_dataset=custom)

	val_order = np.ceil(len(dataclass.num_samples_dir_size)*val_perc).astype(int)
	#val_indices = []
	#print("Val data size: " +  str(val_order))
	train_indices = []
	val_data = {}
	val_data["order"] = np.array(random.sample(list(dataclass.num_samples_dir_size.keys()), k=val_order))
	#print(val_data["order"])
	#print("Total size: " + str(len(list(dataclass.num_samples_dir_size.keys()))))

	#for x in val_data["order"]:
	#    val_indices.extend(list(range(dataclass.num_samples_dir[x]['start'], dataclass.num_samples_dir[x]['end'])))
	#    val_data[x] = dataclass.num_samples_dir[x]
	train_set_size = 0
	#print(val_data["order"])
	for i, x in enumerate(list(dataclass.num_samples_dir_size.keys())):
		if x not in val_data["order"]:
			train_indices.extend(list(range(dataclass.num_samples_dir[x]['start'], dataclass.num_samples_dir[x]['end'])))
			#train_set_size += 1
			#print(x)

	#val_idx = len(val_indices)
	train_idx =  len(train_indices) #dataclass.num_samples - val_idx
	print("Train size: " + str(train_idx))
	#print(train_set_size)
	random.shuffle(train_indices)

	#val_idx = np.array(val_indices)
	train_idx = np.array(train_indices)

	train_data = SubSet(dataclass,train_idx)

	return train_data

def eval_time(args,model,device,mean_image=None):

	train_data = load_run18(args)
	train_loader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=1)
	time_proc_list = []
	time_infer_list = []
	mean_image = mean_image.transpose(2,0,1)
	for ctr, batch in enumerate(train_loader):
		t_out1 = time.time()
		image_arr = batch['image']

		if mean_image is None:
			mean_subtracted = (image_arr)
		else:
			mean_subtracted = (image_arr-mean_image)
		image_tensor = torch.tensor(mean_subtracted)

		image_tensor = image_tensor.to(device,dtype=torch.float)

		#image_tensor = image_tensor.permute(2,0,1).unsqueeze(0)
		t_in1 = time.time()
		logits = model(image_tensor)
		logits = logits.cpu()
		logits = logits.view(1,model.outputs,model.num_points,model.bins).detach().numpy()
		t_in2 = time.time()
		predict = np.argmax(logits,axis=3)
		#print("Predict: ", predict)
		goal = []
		for pt in range(model.num_points):
			point = []
			for coord in range(model.outputs):
				bin_size = (model.max[pt][coord] - model.min[pt][coord])/float(model.bins)
				point.append(model.min[pt][coord] + bin_size*predict[0,coord,pt])
			goal.append(np.array(point))
		goal = np.array(goal)
		t_out2 = time.time()
		time_proc_list.append(t_out2 - t_out1)
		time_infer_list.append(t_in2 - t_in1)

	print("Total proc time: ", np.average(np.array(time_proc_list)))
	print("Total infer time: ", np.average(np.array(time_infer_list)))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('load', help='model to load')
	parser.add_argument('--gpu', help='gpu to use')

	parser.add_argument('--num_images', type=int, default=1, help='number of input images')
	parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
	parser.add_argument('--capacity', type=float, default=1, help='network capacity')
	parser.add_argument('--bins', type=int, default=30, help='number of bins per coordinate')
	parser.add_argument('--outputs', type=int, default=6, help='number of coordinates')
	parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
	parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
	parser.add_argument('--resnet18', type=int, default=0, help='ResNet18')
	parser.add_argument('--test_size', type=int, default=500, help="No of imgs to test")

	parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_dummy_data.npy', help='Mean Image')

	args = parser.parse_args()
	args.min = [(0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0,-1,-0.15,-np.pi,-np.pi/2,-np.pi),(0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0,-2,-0.3,-np.pi,-np.pi/2,-np.pi),(0,-3,-0.5,-np.pi,-np.pi/2,-np.pi)]
	args.max = [(1,0.5,0.1,np.pi,np.pi/2,np.pi),(2,1,0.15,np.pi,np.pi/2,np.pi),(4,1.5,0.2,np.pi,np.pi/2,np.pi),(6,2,0.3,np.pi,np.pi/2,np.pi),(7,0.3,0.5,np.pi,np.pi/2,np.pi)]

	if not args.resnet18:
		model = OrangeNet8(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=args.outputs)
	else:
		model = OrangeNet18(args.capacity,args.num_images,args.num_pts,args.bins,args.min,args.max,n_outputs=args.outputs)

	model.min = args.min
	model.max = args.max

	if os.path.isfile(args.load):
		print(args.load)
		checkpoint = torch.load(args.load,map_location=torch.device('cuda'))
		model.load_state_dict(checkpoint)
		model.eval()
		print("Loaded Model: ",args.load)
	else:
		print("No checkpoint found at: ", args.load)
		return

	if not (os.path.exists(args.mean_image)):
		print('mean image file not found', args.mean_image)
		return 0
	else:
		print('mean image file found')
		mean_image = np.load(args.mean_image)

	if args.gpu:
		device = torch.device('cuda:'+str(args.gpu))
		model = model.to(device)
	else:
		device = torch.device('cpu')
		model = model.to(device)

	eval_time(args,model,device,mean_image)

if __name__ == "__main__":
	main()
