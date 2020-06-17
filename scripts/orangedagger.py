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


class DAggerSet(Dataset):
    def __init__(self,batch_list):
        self.data = batch_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        #TODO: doublecheck format here
        value = self.data[i]
        return {'image':value[0],'points':values[1]}

def DAggerCompare(x,goals,ref_goals,cyl_o,cyl_h=1.6,cyl_r=0.6):
    #TODO: adjust gamma
    gamma = 1
    r = R.from_euler('zyx',x[3:6])
    for g in goals:
        g_trans = r.apply(g[0:3]) + x[0:3]
        if (g_trans[2] < cyl_o[2] + cyl_h) and (g_trans[2] > cyl_o[2]):
            dx = (cyl_o[0] - gtrans[0])
            dy = (cyl_o[1] - gtrans[2])
            if cyl_r*cyl_r > (dx*dx + dy*dy):
                return float('inf')
    metric = 0
    for (g,ref) in zip(goals,ref_goals):
        metric += np.linalg.norm(g[0:3]-ref[0]) #distance
        g_dir = R.from_euler('zyx',g[3:6]).apply([1,0,0])
        ref_dir = ref[1][:,0]
        metric += gamma*(1 - np.dot(g_dir,ref_dir))
    return metric

def wp_from_traj(expert_path,t,tf=15,goal_times=[1,2,3]):
    N = len(expert_path)
    h = float(tf)/N
    goal_idx = [int((t+temp)/h) for temp in goal_times]
    waypoints = []
    for idx in goal_idx:
        if idx >= N:
            waypoints.append(expert_path[N-1])
        elif idx < 0:
            waypoints.append(expert_path[0])
        else:
            waypoints.append(expert_path[idx])
    return waypoints

def transform_local(points,x):
    #transform points from gcop to torch
    torch_points = []
    r_inv = R.from_euler('zyx',x[3:6]).inv()
    for p in points:
        pos = r_inv.apply(p[0] - x[0:3]))
        rot = (r_inv*R.from_dcm(p[1])).as_euler('zyx')
        torch_points.append(np.hstack((pos,rot)))
    return torch_points

def list_to_ds(batch,save_path = None,name = None): 
    if save_path is not None:
        with open(save_path + name,'wb') as f:
            pickle.dump(batch,f,pickle.HIGHEST_PROTOCOL)
    return DAggerSet(batch)

def load_run18(args):
    from customDatasetsOrientation import OrangeSimDataSet, SubSet
    custom = "Run18"
    traj = True
    data = "./data/Run18/"
    val_perc = 0.85

    pt_trans = transforms.Compose([pointToBins(args.min, args.max, args.bins)])
    img_trans = transforms.Compose([RandomHorizontalTrajFlip()])

    dataclass = OrangeSimDataSet(data, args.num_images, args.num_pts, pt_trans, img_trans, custom_dataset=custom)

    val_order = np.ceil(len(dataclass.num_samples_dir_size)*val_perc).astype(int)
    #val_indices = []
    train_indices = []
    val_data = {}
    val_data["order"] = np.array(random.choices(list(dataclass.num_samples_dir_size.keys()), k=val_order))
    #print(val_data["order"])

    #for x in val_data["order"]:
    #    val_indices.extend(list(range(dataclass.num_samples_dir[x]['start'], dataclass.num_samples_dir[x]['end'])))
    #    val_data[x] = dataclass.num_samples_dir[x]

    for i, x in enumerate(list(dataclass.num_samples_dir_size.keys())):
        if x not in val_data["order"]:
            train_indices.extend(list(range(dataclass.num_samples_dir[x]['start'], dataclass.num_samples_dir[x]['end'])))

    #val_idx = len(val_indices)
    train_idx =  len(train_indices) #dataclass.num_samples - val_idx

    random.shuffle(train_indices)

    #val_idx = np.array(val_indices)
    train_idx = np.array(train_indices)

    train_data = SubSet(dataclass,train_idx)
    #val_data = SubSet(dataclass,val_idx)

    #Create DataLoaders
    #train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.j, worker_init_fn=dl_init)

    return train_data


def retrain_model(model,loader,device,ctr="Latest",writer=None,save_path=None,epochs=2,learning_rate=5e-3):
    #Run epochs of training on model
    #Include tensorboard output
    #Create Optimizer
    learn_rate_decay = np.power(1e-3,1/float(args.epochs))#0.9991#10 / args.epochs
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=learn_rate_decay)
    #Learn Rate for Dagger is an open question
    for epoch in range(epochs):
        #acc_total = [[0., 0.], [0., 0.], [0., 0.]]
        #elements = 0.
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
                acc_list = acc_metric(args,logits,point_batch,args.yaw_only)
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
                #for i in range(len(acc_total)):
                #    for j in range(2):
                #        acc_total[i][j] = ((elements * acc_total[i][j]) + (b_size * acc_list[i][j]))/(elements + b_size)
                #elements += b_size
                #print(b_size)
        scheduler.step()#TODO:change when stepping happens
    print("Saving Model")
    name = save_path+'/model' + str(ctr) + '.pth.tar'
    torch.save(model.state_dict(),name)

def run_DAgger(args,sys_f,env,model,data_list=[],batch=512,j=4,device=None,
        plot_step_flag=False,max_steps=100,dt=0.1,save_path=None,mean_image=None):
    eps = 0.5
    ctr = 0
    data_batch=[]
    model_path = createStampedFolder(os.path.join("./model/logs",'variable_log'))
    ts_path = addTimestamp(os.path.join("./model/logs","tensorboard_"))
    writer = SummaryWriter(ts_path)
    while eps > 1e-2:
        #Filenames
        foldername = "trial" + str(trial_num) + "/"
        os.makedirs(save_path + foldername)
        num_fails = 0
        print("Running Environment ",ctr)
        (x0,camName,orange,tree) = shuffleEnv(env)
        if camName is "":
            print("camName not found")
            return 2
        expert_path = run_gcop(x0,tree,orange,t=0,tf=max_steps*dt,N=max_steps)
        print('Just checking that ',len(expert_path),' is ',max_steps+1)
        for step in range(max_steps+1):
            gcop_x = expert_path[step]
            #Get Image
            camAct = makeCamAct(gcop_x)
            image_arr = unity_image(env,camAct,camName)
            #Optionally save image
            if save_path is not None:
                save_image_array(image_arr,save_path+foldername,"sim_image"+str(step))
            #Calculate new goal
            goals = run_model(model,image_arr,mean_image,dev)
            expert_goals = wp_from_traj(expert_path,step*dt)
            cost = DAggerCompare(x,goal,expert_goals,tree[0:3])
            if cost > eps:
                #Compare Failed->Save image and continue on gcop traj
                data_batch.append((image_arr,transform_local(expert_goals,x))
                num_fails += 1
            else:
                #Compare Passed->Take step and recompute expert
                x_new = sys_f(gcop_x,goals,dt)
                new_path = run_gcop(x_new,tree,orange,t=step*dt,tf=max_steps*dt,N=max_steps-1-step)
                expert_path[-len(new_path):]=new_path
        if num_fails < 20:
            eps = eps/2
            print("Only ",num_fails," interventions, reducing eps to ",eps)
        if len(data_batch) > 100:
            print("Retraining")
            #Retrain model
            new_dataset = list_to_ds(data_batch)
            data_list.append(new_dataset)
            run18_data = load_run18(args) #Done here so that everytime different random trajs from run18 get picked
            data_list.append(run18_data) #read online it might work, but not 100% sure
            dataloader = DataLoader(ConcatDataset(data_list),batch_size=batch,shuffle=True,num_works=j)
            data_list = data_list[:-1]
            retrain_model(model,dataloader,dev,writer=writer,save_path=model_path,ctr=ctr)
        ctr += 1

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
  parser.add_argument('--outputs', type=int, default=3, help='number of coordinates')
  parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
  parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
  #Simulation Options
  parser.add_argument('--iters', type=int, default=30, help='number of simulations')
  parser.add_argument('--steps', type=int, default=100, help='Steps per simulation')
  parser.add_argument('--hz', type=float, default=10, help='Recalculation rate')
  parser.add_argument('--env', type=str, default="unity/env_v4", help='unity filename')
  parser.add_argument('--plot_step', type=bool, default=False, help='PLot each step')
  parser.add_argument('--mean_image', type=str, default='data/mean_imgv2_data_Run18.npy', help='Mean Image')
  #Training Options
  parser.add_argument('-j', type=int, default=4, help='number of loader workers')
  parser.add_argument('--batch', type=int, default=1024, help='batch size')
  args = parser.parse_args()
  args.min = [(0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0,-1,-0.15,-np.pi,-np.pi/2,-np.pi),(0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0,-2,-0.3,-np.pi,-np.pi/2,-np.pi),(0,-3,-0.5,-np.pi,-np.pi/2,-np.pi)]
  args.max = [(1,0.5,0.1,np.pi,np.pi/2,np.pi),(2,1,0.15,np.pi,np.pi/2,np.pi),(4,1.5,0.2,np.pi,np.pi/2,np.pi),(6,2,0.3,np.pi,np.pi/2,np.pi),(7,0.3,0.5,np.pi,np.pi/2,np.pi)]
  np.random.seed(args.seed)
  #Load model
  model = OrangeNet8(capacity=args.capacity,num_img=args.num_images,num_pts=args.num_pts,bins=args.bins,n_outputs=args.outputs)
  model.min = args.min
  model.max = args.max
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
  #Create environment
  env = UnityEnvironment(file_name=args.env,seed=0)
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
          if (torch.cuda.device_count() > 1):
              model = nn.DataParallel(model)
  else:
      device = torch.device('cpu')
  #Make Run Folder
  run_num = 0
  globalfolder = 'data/Sim' + str(run_num) + '/'
  while os.path.exists(globalfolder):
    run_num += 1
    globalfolder = 'data/Run' + str(run_num) + '/'
  #Load in Initial Dataset
  datasets = []
  datasets.append(#TODO: Run18 HERE
  #Perform DAgger process
  ret_value = run_DAgger(args,sys_f_linear,env,model,datasets,batch=args.batch,
                         j=args.j,max_steps=args.iters,dt=1/args.hz,device = device,
                         save_path=globalfolder,mean_image=args.mean_image)
  if err_code is not 0:
    print("Dagger failed with code: ",err_code)
  env.close()

if __name__ == '__main__':
  main()
