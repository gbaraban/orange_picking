import numpy as np
from mlagents_envs.environment import UnityEnvironment
import PIL.Image as img
from scipy.spatial.transform import Rotation as R
from orangenetarch import *
from trainorangenet_orientation import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import argparse
from plotting.parsetrajfile import *
from orangesimulation import *
import pickle
import os
import gcophrotor
from torch.utils.tensorboard import SummaryWriter
from customTransforms import *
from orangeimages import *

class DAggerSet(Dataset):
    def __init__(self,batch_list,pt_trans):
        self.data = batch_list
        self.pt_trans = pt_trans
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        #TODO: doublecheck format here
        value = self.data[i]
        img = np.transpose(np.array(value[0]), [2,0,1])
        #img = np.array(img).astype('float32')
        points = self.pt_trans(value[1])
        #print(points)
        return {'image':img,'points':points,"flipped":False}

def DAggerCompare(x,goals,ref_goals,cyl_o,cyl_h=1.6,cyl_r=0.6):
    #TODO: adjust gamma
    gamma = 1
    if len(x) == 6:
        r = R.from_euler('zyx',x[3:6])
    elif ((len(x) == 2) or (len(x) is 4)):
        r = R.from_matrix(np.array(x[1]))
        temp_x = x
        x = list(temp_x[0])
        x.extend(r.as_euler('zyx'))

    for g in goals:
        g_trans = r.apply(g[0:3]) + x[0:3]
        if (g_trans[2] < cyl_o[2] + cyl_h) and (g_trans[2] > cyl_o[2]):
            dx = (cyl_o[0] - g_trans[0])
            dy = (cyl_o[1] - g_trans[2])
            if cyl_r*cyl_r > (dx*dx + dy*dy):
                return float('inf')
    metric = 0
    for (g,ref) in zip(goals,ref_goals):
        metric += np.linalg.norm(g[0:3]-ref[0]) #distance
        g_dir = R.from_euler('zyx',g[3:6]).apply([1,0,0])
        ref_dir = np.array(ref[1])[:,0]
        metric += gamma*(1 - np.dot(g_dir,ref_dir))
    return metric

def wp_from_traj(expert_path,t,tf=15,goal_times=[1,2,3]):
    N = len(expert_path)-1#Because gcop makes N+1 x trajectories
    #if N == 0:
    #    N = 1
    h = float(N)/float(tf)
    #if h == 0:
    #    print(N) 
    goal_idx = [int((t+(temp*h))) for temp in goal_times]
    waypoints = []
    for idx in goal_idx:
        if idx >= N:
            waypoints.append(expert_path[N-1])
        elif idx < 0:
            waypoints.append(expert_path[0])
        else:
            waypoints.append(expert_path[idx])
    #print(waypoints)
    return waypoints

def transform_local(points,x):
    #transform points from gcop to torch
    torch_points = []
    if len(x) == 6:
        r_inv = R.from_euler('zyx',x[3:6]).inv()
    elif ((len(x) == 2) or (len(x) is 4)):
        r = R.from_matrix(np.array(x[1]))
        temp_x = x
        x = list(temp_x[0])
        x.extend(r.as_euler('zyx'))
        r_inv = r.inv()

    for p in points:
        pos = r_inv.apply(np.array(p[0]) - np.array(x[0:3]))
        rot = (r_inv*R.from_dcm(p[1])).as_euler('zyx')
        torch_points.append(np.hstack((pos,rot)))
    return torch_points

def list_to_ds(batch,pt_trans,save_path = None,name = None):
    if save_path is not None:
        with open(save_path + name,'wb') as f:
            pickle.dump(batch,f,pickle.HIGHEST_PROTOCOL)
    return DAggerSet(batch, pt_trans)

def load_run18(args,model):
    from customDatasetsOrientation import OrangeSimDataSet, SubSet
    custom = "Run18"
    traj = True
    data = "./data/Run20/"
    val_perc = 0.9 #TODO: adjust this for number of training trajectories, we are using train traj, so we want to adjust (1-val_perc)

    pt_trans = transforms.Compose([pointToBins(model.min, model.max, model.bins)])
    img_trans = transforms.Compose([RandomHorizontalTrajFlip()])

    dataclass = OrangeSimDataSet(data, model.num_images, model.num_pts, pt_trans, img_trans, custom_dataset=custom)

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
    #print("Train size: " + str(train_idx))
    #print(train_set_size)
    random.shuffle(train_indices)

    #val_idx = np.array(val_indices)
    train_idx = np.array(train_indices)

    train_data = SubSet(dataclass,train_idx)
    #val_data = SubSet(dataclass,val_idx)

    #Create DataLoaders
    #train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.j, worker_init_fn=dl_init)
    #exit()
    return train_data


def retrain_model(args, model,loader,device,ctr="Latest",writer=None,save_path=None,epochs=100,learning_rate=5e-5):
    #Run epochs of training on model
    #Include tensorboard output
    #Create Optimizer
    learn_rate_decay = np.power(1e-3,1/float(epochs)) #0.9991#10 / args.epochs
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=learn_rate_decay)
    #Learn Rate for Dagger is an open question
    for epoch in range(epochs):
        acc_total = [[0., 0.], [0., 0.], [0., 0.]]
        elements = 0.
        print(str(epoch+1) + " out of " + str(epochs))
        for ctr, batch in enumerate(loader):
            point_batch = batch['points']
            optimizer.zero_grad()
            model.train()
            with torch.set_grad_enabled(True):
                batch_imgs = batch['image']
                batch_imgs = batch_imgs.to(device)
                logits = model(batch_imgs)
                del batch_imgs
                del batch
                logits = logits.view(-1,model.outputs,model.num_points,model.bins)
                loss_x = loss_y = loss_z = loss_yaw = loss_r = loss_p = 0
                b_size = logits.shape[0]
                point_batch = point_batch.to(device)
                for temp in range(model.num_points):
                    loss_x += F.cross_entropy(logits[:,0,temp,:],(point_batch)[:,temp,0])
                    loss_y += F.cross_entropy(logits[:,1,temp,:],(point_batch)[:,temp,1])
                    loss_z += F.cross_entropy(logits[:,2,temp,:],(point_batch)[:,temp,2])
                    loss_yaw += F.cross_entropy(logits[:,3,temp,:],(point_batch)[:,temp,3])
                    if model.outputs > 4:
                        loss_p += F.cross_entropy(logits[:,4,temp,:],(point_batch)[:,temp,4])
                        loss_r += F.cross_entropy(logits[:,5,temp,:],(point_batch)[:,temp,5])

                point_batch = point_batch.to('cpu')
                logits = logits.to('cpu')
                acc_list = acc_metric(args,logits,point_batch,yaw_only=False)
                del point_batch
                del logits
                batch_loss = loss_x + loss_y + loss_z + loss_yaw + loss_p + loss_r
                batch_loss.backward()
                optimizer.step()
                writer.add_scalar('train_loss',batch_loss,ctr+epoch*len(loader))              
                writer.add_scalar('train_loss_x',loss_x,ctr+epoch*len(loader))
                writer.add_scalar('train_loss_y',loss_y,ctr+epoch*len(loader))
                writer.add_scalar('train_loss_z',loss_z,ctr+epoch*len(loader))
                writer.add_scalar('train_loss_yaw',loss_yaw,ctr+epoch*len(loader))
                if model.outputs > 4:
                    writer.add_scalar('train_loss_p',loss_p,ctr+epoch*len(loader))
                    writer.add_scalar('train_loss_r',loss_r,ctr+epoch*len(loader))
                for ii, acc in enumerate(acc_list):
                    writer.add_scalar('train_acc_'+str(ii),acc[0],ctr+epoch*len(loader))

                #print('Training Accuracy: ',acc_list)
                for i in range(len(acc_total)):
                    for j in range(2):
                        acc_total[i][j] = ((elements * acc_total[i][j]) + (b_size * acc_list[i][j]))/(elements + b_size)
                elements += b_size
                #print("Done: " + str(elements))
        print('Training Accuracy: ',acc_list)
        scheduler.step()#TODO:change when stepping happens
    print("Saving Model")
    name = save_path+'/model' + str(ctr) + '.pth.tar'
    torch.save(model.state_dict(),name)
    return name

def run_DAgger(args,sys_f,env_name,model,data_list=[],j=4,device=None,
        plot_step_flag=False,max_steps=100,dt=0.1,save_path=None,mean_image=None,eps=0.5,physics_hz=50,get_name=False,batch=128):
    ctr = 0
    image_spacing = 1/dt #number of timesteps between images in multi-image networks
    data_batch=[]
    model_path = createStampedFolder(os.path.join("./model/logs",'variable_log'))
    ts_path = addTimestamp(os.path.join("./model/logs","tensorboard_"))
    writer = SummaryWriter(ts_path)
    trial_num = 0
    while eps > 5e-3:
        #Filenames
        foldername = "trial" + str(trial_num) + "/"
        os.makedirs(save_path + foldername)
        num_fails = 0
        print("Running Environment ",ctr)
        (env,x0,camName,envName,orange,tree) = shuffleEnv(env_name,trial_num=trial_num)
        if camName is "":
            print("camName not found")
            env.close()
            return 2
        temp_gcop = run_gcop(x0,tree,orange,t=0,tf=max_steps*dt,N=int((max_steps*dt)*physics_hz))
        expert_path = temp_gcop[0]
        u_path = temp_gcop[1]
        print('Just checking that ',len(expert_path),' is ',max_steps+1)
        for step in range(max_steps+1):
            step_time = step*dt
            gcop_x = expert_path[int(step_time*physics_hz)]
            #Get Image
            image_arr = None
            for ii in range(model.num_images):
                temp_idx = max(0,int(step_time*physics_hz - int(ii*image_spacing)))
                camAct = makeCamAct(expert_path[temp_idx])
                if image_arr is None:
                    image_arr = unity_image(env,camAct,camName)
                    #print(image_arr.shape)
                else:
                    image_arr = np.concatenate((image_arr,unity_image(env,camAct,camName)),axis=2)#TODO: check axis number
                    #print(image_arr.shape)
            #Optionally save image
            if save_path is not None:
                save_image_array(image_arr[:,:,0:3],save_path+foldername,"sim_image"+str(step)) #TODO: use concatenation axis from above
            #Calculate new goal
            goals = run_model(model,image_arr,mean_image, device)# TODO: fix-->'DataParallel' object has no attribute 'outputs': when adding device with multi gpu
            expert_goals = wp_from_traj(expert_path,step_time)
            cost = DAggerCompare(gcop_x,goals,expert_goals,tree[0:3])
            if cost > eps:
                #Compare Failed->Save image and continue on gcop traj
                data_batch.append((image_arr,transform_local(expert_goals,gcop_x)))
                num_fails =  1
            else:
                #Compare Passed->Take step and recompute expert
                x_new = sys_f(gcop_x,goals,dt,plot_flag=True)
                new_path = run_gcop(x_new,tree,orange,t=step*dt,tf=max_steps*dt,N=int((max_steps*dt)*physics_hz)) #max_steps-1-step)
                expert_path[-len(new_path[0]):]=new_path[0]
                u_path[-len(new_path[1]):]=new_path[1]
        if num_fails < 20:
            eps = eps/2
            print("Only ",num_fails," interventions, reducing eps to ",eps)
        if len(data_batch) > 100:
            print("Retraining")
            #Retrain model
            pt_trans = transforms.Compose([pointToBins(model.min,model.max,model.bins)])#,GaussLabels(1,1e-10,model.bins)])
            new_dataset = list_to_ds(data_batch, pt_trans)
            data_list.append(new_dataset)
            run18_data = load_run18(args,model) #Done here so that everytime different random trajs from run18 get picked
            data_list.append(run18_data) #read online it might work, but not 100% sure
            dataloader = DataLoader(ConcatDataset(data_list),batch_size=batch,shuffle=True,num_workers=j)
            model_name = retrain_model(args,model,dataloader,device,writer=writer,save_path=model_path,ctr=ctr)
            data_list = data_list[:-1]
        ctr += 1
        trial_num += 1
        env.close()
    if get_name:
        return 0, model_name
    else:
        return 0


class FakeArgs:
    def __init__(self,load,gpu,seed,num_images,num_pts,capacity,bins,outputs,mins,maxs,resnet18,steps,hz,physics,env,plot_step,mean_image,worker_id,j,batch):
        self.load = load
        self.gpu = gpu
        self.seed = seed
        self.num_images = num_images
        self.num_pts = num_pts
        self.capacity = capacity
        self.bins = bins
        self.outputs = outputs
        self.min = mins
        self.max = maxs
        self.resnet18 = resnet18
        self.steps = steps
        self.hz = hz
        self.physics = physics
        self.env = env
        self.plot_step = plot_step
        self.mean_image = mean_image
        self.worker_id = worker_id
        self.j = j
        self.batch = batch


def main_function(load_model=None,gpu=None,seed=0,num_images=1,num_pts=3,capacity=1,bins=1,outputs=6,resnet18=False,steps=100,hz=5,physics=5,env="unity/env_v7",plot_step=False,
                  mean_image='data/mean_imgv2_data_Run20.npy',worker_id=0,j=4,batch=128):
  #parser = argparse.ArgumentParser()
  #parser.add_argument('load', help='model to load')
  #parser.add_argument('--gpu', help='gpu to use')
  #parser.add_argument('--seed', type=int, default=0, help='random seed')
  #Model Options
  #parser.add_argument('--num_images', type=int, default=1, help='number of input images')
  #parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
  #parser.add_argument('--capacity', type=float, default=1, help='network capacity')
  #parser.add_argument('--bins', type=int, default=30, help='number of bins per coordinate')
  #parser.add_argument('--outputs', type=int, default=6, help='number of coordinates')
  #parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
  #parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
  #parser.add_argument('--resnet18', type=int, default=0, help='ResNet18')
  #Simulation Options
  #parser.add_argument('--iters', type=int, default=30, help='number of simulations')
  #parser.add_argument('--steps', type=int, default=100, help='Steps per simulation')
  #parser.add_argument('--hz', type=float, default=5, help='Recalculation rate')
  #parser.add_argument('--physics', type=float, default=5, help='Recalculation rate')
  #parser.add_argument('--env', type=str, default="unity/env_v7", help='unity filename')
  #parser.add_argument('--plot_step', type=bool, default=False, help='PLot each step')
  #parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run20.npy', help='Mean Image')
  #parser.add_argument('--worker_id', type=int, default=0, help='Worker ID for Unity')

  #Training Options
  #parser.add_argument('-j', type=int, default=4, help='number of loader workers')
  #parser.add_argument('--batch', type=int, default=256, help='batch size')
  #args = parser.parse_args()
  mins = [(0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0,-1,-0.15,-np.pi,-np.pi/2,-np.pi),(0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0,-2,-0.3,-np.pi,-np.pi/2,-np.pi),(0,-3,-0.5,-np.pi,-np.pi/2,-np.pi)]
  maxs = [(1,0.5,0.1,np.pi,np.pi/2,np.pi),(2,1,0.15,np.pi,np.pi/2,np.pi),(4,1.5,0.2,np.pi,np.pi/2,np.pi),(6,2,0.3,np.pi,np.pi/2,np.pi),(7,0.3,0.5,np.pi,np.pi/2,np.pi)]
  np.random.seed(seed)
  #Load model
  if not resnet18:
      model = OrangeNet8(capacity=capacity,num_img=num_images,num_pts=num_pts,bins=bins,mins=mins,maxs=maxs,n_outputs=outputs)
  else:
      model = OrangeNet18(capacity=capacity,num_img=num_images,num_pts=num_pts,bins=bins,mins=mins,maxs=maxs,n_outputs=outputs)

  model.min = mins
  model.max = maxs

  args = FakeArgs(load_model,gpu,seed,num_images,num_pts,capacity,bins,outputs,mins,maxs,resnet18,steps,hz,physics,env,plot_step,mean_image,worker_id,j,batch)

  #if args.worker_id == 100:
  #    args.worker_id = args.seed

  if os.path.isfile(load):
      checkpoint = torch.load(load)
      model.load_state_dict(checkpoint)
      model.eval()
      print("Loaded Model: ",load)
  else:
      print("No checkpoint found at: ", load)
      return
  #Load Mean Image
  if not (os.path.exists(mean_image)):
      print('mean image file not found', mean_image)
      return 0
  else:
      print('mean image file found')
      mean_image = np.load(mean_image)
  #Pick CUDA Device
  use_cuda = torch.cuda.is_available()
  print('Cuda Flag: ',use_cuda)
  if use_cuda:
      if gpu:
          device = torch.device('cuda:'+str(gpu))
          model = model.to(device)
      else:
          device = torch.device('cuda')
          model = model.to(device)
          #if (torch.cuda.device_count() > 1):
          #    model = nn.DataParallel(model)
  else:
      device = torch.device('cpu')
  #Make Run Folder
  run_num = 0
  globalfolder = 'data/dagger/Sim' + str(run_num) + '_' + str(steps) + '_' + str(hz) + '_' + str(physics) + '/'
  while os.path.exists(globalfolder):
      run_num += 1
      globalfolder = 'data/dagger/Sim' + str(run_num) +  '_' + str(steps) + '_' + str(hz) + '_' + str(physics) + '/'
  #Load in Initial Dataset
  datasets = []
  #TODO: Run18 HERE: did it inside runDagger, for random trajectories being used for each retrainining
  #Perform DAgger process

  ret_value, model_name = run_DAgger(args,sys_f_gcop,env,model,datasets,j=j,max_steps=steps,dt=1/hz,device = device,
                         save_path=globalfolder,mean_image=mean_image,physics_hz=physics,get_name=True,batch=batch)
  if ret_value is not 0:
      print("Dagger failed with code: ", ret_value)

  return model_name

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('load', help='model to load')
  parser.add_argument('--gpu', help='gpu to use')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  #Model Options
  parser.add_argument('--num_images', type=int, default=1, help='number of input images')
  parser.add_argument('--num_pts', type=int, default=3, help='number of output waypoints')
  parser.add_argument('--capacity', type=float, default=1, help='network capacity')
  parser.add_argument('--bins', type=int, default=30, help='number of bins per coordinate')
  parser.add_argument('--outputs', type=int, default=6, help='number of coordinates')
  parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
  parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
  parser.add_argument('--resnet18', type=int, default=0, help='ResNet18')
  #Simulation Options
  #parser.add_argument('--iters', type=int, default=30, help='number of simulations')
  parser.add_argument('--steps', type=int, default=100, help='Steps per simulation')
  parser.add_argument('--hz', type=float, default=5, help='Recalculation rate')
  parser.add_argument('--physics', type=float, default=5, help='Recalculation rate')
  parser.add_argument('--env', type=str, default="unity/env_v7", help='unity filename')
  parser.add_argument('--plot_step', type=bool, default=False, help='PLot each step')
  parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run20.npy', help='Mean Image')
  parser.add_argument('--worker_id', type=int, default=0, help='Worker ID for Unity')
  #Training Options
  parser.add_argument('-j', type=int, default=4, help='number of loader workers')
  parser.add_argument('--batch', type=int, default=256, help='batch size')
  args = parser.parse_args()
  args.min = [(0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0,-1,-0.15,-np.pi,-np.pi/2,-np.pi),(0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0,-2,-0.3,-np.pi,-np.pi/2,-np.pi),(0,-3,-0.5,-np.pi,-np.pi/2,-np.pi)]
  args.max = [(1,0.5,0.1,np.pi,np.pi/2,np.pi),(2,1,0.15,np.pi,np.pi/2,np.pi),(4,1.5,0.2,np.pi,np.pi/2,np.pi),(6,2,0.3,np.pi,np.pi/2,np.pi),(7,0.3,0.5,np.pi,np.pi/2,np.pi)]
  np.random.seed(args.seed)
  #Load model
  if not args.resnet18:
      model = OrangeNet8(capacity=args.capacity,num_img=args.num_images,num_pts=args.num_pts,bins=args.bins,mins=args.min,maxs=args.max,n_outputs=args.outputs)
  else:
      model = OrangeNet18(capacity=args.capacity,num_img=args.num_images,num_pts=args.num_pts,bins=args.bins,mins=args.min,maxs=args.max,n_outputs=args.outputs)

  model.min = args.min
  model.max = args.max

  #if args.worker_id == 100:
  #    args.worker_id = args.seed

  if os.path.isfile(args.load):
      checkpoint = torch.load(args.load)
      model.load_state_dict(checkpoint)
      model.eval()
      print("Loaded Model: ",args.load)
  else:
      print("No checkpoint found at: ", args.load)
      return
  #Load Mean Image
  if not (os.path.exists(args.mean_image)):
      print('mean image file not found', args.mean_image)
      return 0
  else:
      print('mean image file found')
      mean_image = np.load(args.mean_image)
  #Pick CUDA Device
  use_cuda = torch.cuda.is_available()
  print('Cuda Flag: ',use_cuda)
  if use_cuda:
      if args.gpu:
          device = torch.device('cuda:'+str(args.gpu))
          model = model.to(device)
      else:
          device = torch.device('cuda')
          model = model.to(device)
          #if (torch.cuda.device_count() > 1):
          #    model = nn.DataParallel(model)
  else:
      device = torch.device('cpu')
  #Make Run Folder
  run_num = 0
  globalfolder = 'data/dagger/Sim' + str(run_num) + '_' + str(args.steps) + '_' + str(args.hz) + '_' + str(args.physics) + '/'
  while os.path.exists(globalfolder):
      run_num += 1
      globalfolder = 'data/dagger/Sim' + str(run_num) +  '_' + str(args.steps) + '_' + str(args.hz) + '_' + str(args.physics) + '/'
  #Load in Initial Dataset
  datasets = []
  #TODO: Run18 HERE: did it inside runDagger, for random trajectories being used for each retrainining
  #Perform DAgger process
  ret_value = run_DAgger(args,sys_f_gcop,args.env,model,datasets,j=args.j,max_steps=args.steps,dt=1/args.hz,device = device,
                         save_path=globalfolder,mean_image=mean_image,physics_hz=args.physics)
  if ret_value is not 0:
      print("Dagger failed with code: ", ret_value)

if __name__ == '__main__':
  main()
