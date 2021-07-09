import tensorflow as tf
import numpy as np
from scipy import special as scispec
import argparse
import pickle
import os
from orangeclassnetarch import *
from datetime import datetime
import PIL.Image as img
import copy

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

def parseDirData(run_dir, seed, resample, val_perc, num_pts, dt = 1):
  trial_list = os.listdir(run_dir)
  num_list = []
  traj_data = []
  num_samples = 0
  for trial_dir in trial_list:
    trial_list = os.listdir(run_dir+'/'+trial_dir)
    num_samples_i = len(trial_list) - 1
    num_list.append(num_samples_i)
    with open(run_dir+'/'+trial_dir+'/data.pickle','rb') as f:
      data = pickle.load(f,encoding='latin1')
      traj_data.append(data)
    num_samples += num_samples_i
  np.random.seed(seed)
  if resample:
    rand_idx = np.random.choice(num_samples, size=num_samples, replace=True)
  else:
    rand_idx = np.random.permutation(num_samples)
  val_idx = np.ceil(num_samples*val_perc).astype(int)
  train_idx = num_samples - val_idx
  val_indices = rand_idx[-val_idx:]
  train_indices = rand_idx[:train_idx]
  # print(train_indices)
  # print(val_indices)
  return train_indices, val_indices, num_list, traj_data

def parseFiles(idx,num_list,run_dir,model,traj_data):
  idx = idx.astype(int)
  trial_idx = 0
  while True:
    #print(num_list[trial_idx])
    if num_list[trial_idx] <= idx:
      #print(num_list[trial_idx])
      idx = idx - num_list[trial_idx]
      trial_idx += 1
    else:
      break
  print((idx,trial_idx))
  trial_list = os.listdir(run_dir)
  trial = trial_list[trial_idx]
  image = None
  image_idx = idx
  for ii in range(model.num_images):
    temp_idx = max(0,image_idx - ii)
    temp_image = img.open(run_dir+'/'+trial + '/image' + str(temp_idx) + '.png').resize((model.h,model.w))
    temp_image = np.array(temp_image.getdata()).reshape(temp_image.size[0],temp_image.size[1],3)
    temp_image = temp_image[:,:,0:3]/255.0 #Cut out alpha
    # temp_image = temp_image/255.0
    if image is None:
        image = temp_image
    else:
        image = np.concatenate((image,temp_image),axis=2)
  points = traj_data[trial_idx][idx]
  local_pts = []
  ctr = 0
  for point in points:
    #Convert into bins
    min_i = model.min[ctr]
    max_i = model.max[ctr]
    bin_nums = (point - min_i)/(max_i-min_i)
    bin_nums_scaled = (bin_nums*model.bins).astype(int)
    #for temp in range(3):
        #if (bin_nums_scaled[temp] > model.bins-1) or (bin_nums_scaled[temp] < 0):
            #print(str(point) + ' is out of bounds: ' + str(min_i) + ' ' + str(max_i))
    bin_nums = np.clip(bin_nums_scaled,a_min=0,a_max=model.bins-1)
    ctr += 1
    labels = np.zeros((3,model.bins))    
    mean = 0.99
    stdev = 0.0001
    for j in range(len(bin_nums)):
      for i in range(labels.shape[1]):
        labels[j][i] = mean * (np.exp((-np.power(bin_nums[j]-i, 2))/(2 * np.power(stdev, 2))))
    local_pts.append(labels)
  local_pts = np.array(local_pts)
  local_pts.resize((model.num_points,3,model.bins))
  return image, local_pts

def loadData(idx,num_list,run_dir,model,mean_image,traj_data):
  images = []
  waypoints_x = []
  waypoints_y = []
  waypoints_z = []
  for ii in idx:
    image, waypoint = parseFiles(ii,num_list,run_dir,model,traj_data)
    images.append(np.array(image - mean_image))
    waypoints_x.append(waypoint[:,0,:])
    waypoints_y.append(waypoint[:,1,:])
    waypoints_z.append(waypoint[:,2,:])
  waypoints_x = np.array(waypoints_x)
  waypoints_y = np.array(waypoints_y)
  waypoints_z = np.array(waypoints_z)
  return np.array(images), np.array(waypoints_x).reshape(-1,model.num_points,model.bins), np.array(waypoints_y).reshape(-1,model.num_points,model.bins), np.array(waypoints_z).reshape(-1,model.num_points,model.bins)

def compute_mean_image(idx,run_dir,model,dt = 1):
  print('ERROR: WE SHOULDN"T BE HERE')
  exit(0)
  
def acc_metric(logits,x,y,z, model):
    soft_probs = logits#scispec.softmax(logits,axis=4)
    truth = [x,y,z]
    pt_list = []
    for pt in range(model.num_points):
        coord_list = []
        for ii in range(3):
            bin_length = float(model.max[pt][ii] - model.min[pt][ii])/model.bins
            logit_bin = np.argmax(logits[pt][ii],axis=1)
            #print(truth[ii].shape)
            true_bin = np.argmax(truth[ii][:,pt,:],axis=1)
            dist = (logit_bin - true_bin)*bin_length
            coord_list.append(dist)
        x_diff = coord_list[0]
        y_diff = coord_list[1]
        z_diff = coord_list[2]
        dist = np.vstack([x_diff, y_diff, z_diff])
        dist = np.linalg.norm(dist,axis=0)
        dist = np.mean(dist)
        pt_list.append(dist)
    return pt_list



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='data file')
  parser.add_argument('--load', help='model to load')
  parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train for')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--resample', action='store_true', help='resample data')
  parser.add_argument('--gpus', help='gpu to use')
  parser.add_argument('--num_images', type=int, default=2, help='number of input images')
  parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=5e-3, help='batch size')
  parser.add_argument('--num_pts', type=int, default=1, help='number of output waypoints')
  parser.add_argument('--capacity', type=float, default=1, help='network capacity')
  parser.add_argument('--cam_coord', type=float, default=-1, help='use focal length coordinates')
  parser.add_argument('--min', type=tuple, default=(0,-0.5,-0.5), help='minimum xyz ')
  parser.add_argument('--max', type=tuple, default=(1,0.5,0.5), help='maximum xyz')
  parser.add_argument('--bins', type=int, default=100, help='number of bins per coordinate')
  parser.add_argument('--dense', type=int, default=0, help='number of additional dense layers')
  args = parser.parse_args()
  args.min = [(0,-0.5,-0.1),(0,-1,-0.15),(0,-1.5,-0.2),(0,-2,-0.3),(0,-3,-0.5)]
  args.max = [(1,0.5,0.1),(2,1,0.15),(4,1.5,0.2),(6,2,0.3),(7,0.3,0.5)]

  if (args.gpus is not None):
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

  # Model and optimization params
  val_perc = 0.1
  #g_depths = [64, 64, 64]
  #f_depths = [64, 64, 64]
  batch_size = args.batch_size#512#1024#64
  num_epochs = args.epochs
  learning_rate = args.learning_rate#50#e-1
  learn_rate_decay = 100 / num_epochs
  save_variables_divider = 10
  log_path = './model/logs'
  save_path = createStampedFolder(os.path.join(log_path, 'variable_log'))

  ######################

  # Make model
  print ('Building model')
  model = OrangeClassNet(args.capacity, args.num_images, args.num_pts,
                         args.cam_coord, args.min, args.max, args.bins, args.dense)

  # Load in Data
  train_indices, val_indices, num_list, traj_data = parseDirData(args.data, args.seed, args.resample, val_perc, args.num_pts)
  num_train_samples = train_indices.shape[0]
  num_val_samples = val_indices.shape[0]

  # Train model
  print ('Training...')
  print ('Training Samples: ' + str(num_train_samples))
  print ('Validation Samples: ' + str(num_val_samples))
  data_loc = copy.deepcopy(args.data)
  data_loc_name = data_loc.strip("..").strip(".").strip("/").replace("/", "_")
  mean_img_loc = data_loc + "../mean_img_" + data_loc_name + '.npy' 
  if not (os.path.exists(mean_img_loc)):
    print('mean image file not found')
    mean_image = compute_mean_image(train_indices, data_loc, model)
    np.save(mean_img_loc, mean_image)
  else:
    print('mean image file found')
    mean_image = np.load(mean_img_loc)
  # mean_image = np.zeros((model.h, model.w, 3))

  val_inputs, val_outputs_x, val_outputs_y, val_outputs_z = loadData(val_indices,num_list,data_loc,model,mean_image,traj_data)

  print ('Validation Loaded')
  train_path = addTimestamp(os.path.join(log_path, 'train_'))
  val_path = addTimestamp(os.path.join(log_path, 'validation_'))
  plot_data_path = addTimestamp(os.path.join(log_path, 'plot_data_'))
  train_writer = tf.summary.FileWriter(train_path, graph=tf.get_default_graph())
  val_writer = tf.summary.FileWriter(val_path, graph=tf.get_default_graph())
  os.makedirs(plot_data_path)

  saver = tf.train.Saver()
  init = tf.global_variables_initializer()
  feed_dict = {}#model.keep_prob: 0.9}
  print ('Writers Set Up')

  with tf.Session() as sess:# Load model if specified 
    if args.load:
      saver.restore(sess, tf.train.latest_checkpoint(args.load))
      uninit_vars_op = tf.report_uninitialized_variables()
      uninit_vars = sess.run(uninit_vars_op)
      uninit_vars_op.mark_used()
      if uninit_vars.size != 0:
        print(uninit_vars)#, sep=',')
        sess.close()
        raise RuntimeError('Uninitialized variables present')
    else:
      sess.run(init)
    print ('Session')
    iters = 0
    plotting_data = dict()
    plotting_data['idx'] = range(5)
    plotting_data['truth'] = [val_outputs_x[plotting_data['idx']],
                              val_outputs_y[plotting_data['idx']],
                              val_outputs_z[plotting_data['idx']]]
    plotting_data['data'] = list()
    plotting_data['foc_l'] = args.cam_coord
    plotting_data['min'] = model.min
    plotting_data['max'] = model.max
    plotting_data['bins'] = model.bins
    for ii in plotting_data['idx']:
      plotting_data['data'].append([])
    #print(plotting_data)
    for epoch in range(num_epochs):
      print('Epoch: ', epoch)
      batch_idx = 0
      # Decay learning rate
      new_learn_rate = np.exp(-epoch*learn_rate_decay)*learning_rate
      print('Learning Rate Set to: ' + str(new_learn_rate))
      model.learning_fac.assign(new_learn_rate) 
      
      while batch_idx < num_train_samples:
        end_idx = min(batch_idx + batch_size, num_train_samples)
        train_inputs, train_outputs_x, train_outputs_y, train_outputs_z = loadData(train_indices[batch_idx:end_idx],num_list,data_loc, model, mean_image,traj_data)
        feed_dict[model.image_input] = train_inputs
        feed_dict[model.waypoint_output_x] = train_outputs_x
        feed_dict[model.waypoint_output_y] = train_outputs_y
        feed_dict[model.waypoint_output_z] = train_outputs_z
        #sess.run([model.train_summary_op, model.train_step], feed_dict=feed_dict)
        sess.run(model.train_step, feed_dict=feed_dict)
        batch_idx = batch_idx + batch_size
        iters = iters + 1
        if iters % 20 == 0:
          summary, logits = sess.run([model.train_summ,model.logits], feed_dict=feed_dict)
          accuracy = acc_metric(logits,train_outputs_x,train_outputs_y,train_outputs_z, model)
          print('Training Accuracy: ' + str(accuracy))
          train_writer.add_summary(summary, iters)
        #Clear references to data:
        train_inputs = train_outputs = feed_dict[model.image_input] = feed_dict[model.waypoint_output_x] = feed_dict[model.waypoint_output_y] = feed_dict[model.waypoint_output_z] = None

      val_batch_idx = 0
      num_validation = len(val_indices)
      #val_summary = 0
      val_cost = np.zeros((1,))
      resnet_output = np.zeros((args.num_pts, 3, 0, model.bins)) # 2nd arg for num_waypoints
      raw_losses = np.zeros((3,))
      accuracy = []

      while val_batch_idx < num_validation:
        val_batch_endx = min(val_batch_idx + batch_size, num_validation)
        val_dict = {model.image_input: val_inputs[val_batch_idx:val_batch_endx], 
            model.waypoint_output[0]: val_outputs_x[val_batch_idx:val_batch_endx],
            model.waypoint_output[1]: val_outputs_y[val_batch_idx:val_batch_endx],
            model.waypoint_output[2]: val_outputs_z[val_batch_idx:val_batch_endx]}
        
        val_summary_temp, val_cost_temp, resnet_output_temp, raw_losses_temp = sess.run([model.val_summ, model.objective, model.logits, model.losses], feed_dict=val_dict)
        val_writer.add_summary(val_summary_temp, iters)

        #val_summary_temp
        val_cost = np.multiply(val_cost, (float(val_batch_idx)/val_batch_endx)) + np.multiply(val_cost_temp, (float(val_batch_endx-val_batch_idx)/val_batch_endx))
        resnet_output_temp = np.array(resnet_output_temp)
        accuracy.append(acc_metric(resnet_output_temp,val_dict[model.waypoint_output[0]],val_dict[model.waypoint_output[1]],val_dict[model.waypoint_output[2]],model))
        resnet_output = np.concatenate((resnet_output, resnet_output_temp), axis=2)
        raw_losses = np.multiply(raw_losses_temp, (float(val_batch_idx)/val_batch_endx)) + np.multiply(np.array(raw_losses_temp), (float(val_batch_endx-val_batch_idx)/val_batch_endx))

        val_batch_idx = val_batch_endx
      
      accuracy = np.mean(accuracy,axis=0)
      print('Validation Summary = ', val_cost)
      print('Accuracy = ',accuracy)
      resnet_output = np.array(resnet_output)
      print(raw_losses)
      print(resnet_output.shape)
      for ii in plotting_data['idx']:
          plotting_data['data'][ii].append(resnet_output[:,:,ii,:])
      with open(plot_data_path+'/data.pickle','wb') as f:
          pickle.dump(plotting_data,f,pickle.HIGHEST_PROTOCOL)

      #val_writer.add_summary(val_summary, iters)

      train_writer.flush()
      val_writer.flush()
      # Save variables
      if ((epoch + 1) % save_variables_divider == 0 or (epoch == 0) or (epoch == num_epochs - 1)):
          print("Saving variables")
          if epoch == 0:
            print("For epoch 0")
            saver.save(sess, os.path.join(save_path, 'variables'), epoch)
          else:
            print("For epoch ", epoch)
            saver.save(sess, os.path.join(save_path, 'variables'), epoch, write_meta_graph=False)
      # Re-shuffle data after each epoch
      rand_idx = np.random.permutation(num_train_samples)
      train_indices = train_indices[rand_idx]
  train_writer.flush()
  val_writer.flush()
  print("Done")


if __name__ == '__main__':
    main()
