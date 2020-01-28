import tensorflow as tf
import numpy as np
import argparse
import pickle
import os
from orangenetarch import *
from datetime import datetime
import PIL.Image as img

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

def parseDirData(run_dir, seed, resample, val_perc, time_window = 5):
  trial_list = os.listdir(run_dir)
  num_samples = 0
  for trial_dir in trial_list:
    with open(run_dir+"/"+trial_dir+"/metadata.pickle",'rb') as data_f:
      data = pickle.load(data_f)#, encoding='latin1')
      N = data['N']
      tf = data['tf']
      h = float(N)/tf
      reduced_N = int(N - time_window*h)
      num_samples += reduced_N
  np.random.seed(seed)
  if resample:
    rand_idx = np.random.choice(num_samples, size=num_samples, replace=True)
  else:
    rand_idx = np.random.permutation(num_samples)
  val_idx = np.ceil(num_samples*val_perc).astype(int)
  train_idx = num_samples - val_idx
  val_indices = rand_idx[-val_idx:]
  train_indices = rand_idx[:train_idx]
  return train_indices, val_indices

def parseFiles(idx,traj_data,trial_dir, model):
  idx = idx.astype(int)
  image_idx = idx[0]
  image = None
  for ii in range(model.num_images):
    temp_idx = max(0,image_idx - ii)
    temp_image = img.open(trial_dir+'image'+str(temp_idx)+'.png').resize((model.w,model.h))
    temp_image = np.array(temp_image.getdata()).reshape(temp_image.size[0],temp_image.size[1],3)
    temp_image = temp_image[:,:,0:3]/255.0 #Cut out alpha
    temp_image = temp_image/255.0
    if image is None:
        image = temp_image
    else:
        image = np.concatenate((image,temp_image),axis=2)
  R0 = np.array(traj_data[image_idx][1])
  p0 = np.array(traj_data[image_idx][0])
  local_pts = []
  for i in idx:
    state = traj_data[i]
    point = np.matmul(R0.T,np.array(state[0]) - p0)
    local_pts.append(point)
  local_pts = np.array(local_pts[1:])
  local_pts.resize(model.output_dim)
  return image, local_pts

def loadData(idx,run_dir,model,time_window = 5):
  num_points = model.output_dim/3
  #Assume reduced N is constant for all trials
  trial_list = os.listdir(run_dir)
  trial_dir = trial_list[0]
  reduced_N = -1
  with open(run_dir+"/"+trial_dir+"/metadata.pickle",'rb') as data_f:
    data = pickle.load(data_f, encoding='latin1')
    N = data['N']
    tf = data['tf']
    h = float(N)/tf
    reduced_N = int(N - time_window*h)
  images = []
  waypoints = []
  trial_num = np.floor(idx/reduced_N).astype(int)
  image_num = np.mod(idx,reduced_N)
  for trial_idx, image_idx in zip(trial_num,image_num):
    trial_dir = run_dir+"/"+trial_list[trial_idx]+"/"
    traj_data = []
    with open(trial_dir+"trajdata.pickle",'rb') as data_f:
      traj_data = pickle.load(data_f, encoding='latin1')
    metadata = []
    with open(trial_dir+"metadata.pickle",'rb') as data_f:
      metadata = pickle.load(data_f, encoding='latin1')
    idx_final = image_idx + metadata['N']*time_window/metadata['tf']
    offset_idx = np.floor(np.linspace(image_idx,idx_final,num_points+1))
    image, waypoint = parseFiles(offset_idx,traj_data,trial_dir, model)
    waypoints.append(waypoint)
    images.append(image)
  return np.array(images), np.array(waypoints)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='data file')
  parser.add_argument('--load', help='model to load')
  parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train for')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--resample', action='store_true', help='resample data')
  parser.add_argument('--gpus', help='gpu to use')
  parser.add_argument('--num_images', type=int, default=2, help='number of input images')
  parser.add_argument('--batch_size', type=int, default=256, help='batch size')
  parser.add_argument('--num_pts', type=int, default=2, help='number of output waypoints')
  parser.add_argument('--capacity', type=float, default=1, help='network capacity')
  args = parser.parse_args()

  if (args.gpus is not None):
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

  # Model and optimization params
  val_perc = 0.01
  #g_depths = [64, 64, 64]
  #f_depths = [64, 64, 64]
  batch_size = args.batch_size#512#1024#64
  num_epochs = args.epochs
  learning_rate = 5e-2#50#e-1
  learn_rate_decay = 0#1000 / num_epochs
  save_variables_divider = 10
  log_path = './model/logs'
  save_path = createStampedFolder(os.path.join(log_path, 'variable_log'))

  ######################

  # Make model
  print ('Building model')
  model = OrangeResNet(args.capacity, args.num_images, args.num_pts)

  # Load in Data
  train_indices, val_indices = parseDirData(args.data, args.seed, args.resample, val_perc)
  num_train_samples = train_indices.shape[0]

  # Train model
  print ('Training...')
  val_inputs, val_outputs = loadData(val_indices,args.data, model)
  val_dict = {model.image_input: val_inputs, model.waypoint_output: val_outputs}
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
    plotting_data['truth'] = val_outputs[plotting_data['idx']]
    plotting_data['data'] = list()
    for ii in plotting_data['idx']:
      plotting_data['data'].append([])
    #print(plotting_data)
    for epoch in range(num_epochs):
      print('Epoch: ', epoch)
      batch_idx = 0
      # Decay learning rate
      model.learning_fac.assign(np.exp(-epoch*learn_rate_decay)*learning_rate)
      while batch_idx < num_train_samples:
        end_idx = min(batch_idx + batch_size, num_train_samples)
        train_inputs, train_outputs = loadData(train_indices[batch_idx:end_idx],args.data, model)
        feed_dict[model.image_input] = train_inputs
        feed_dict[model.waypoint_output] = train_outputs
        #sess.run([model.train_summary_op, model.train_step], feed_dict=feed_dict)
        sess.run(model.train_step, feed_dict=feed_dict)
        batch_idx = batch_idx + batch_size
        iters = iters + 1
        if iters % 20 == 0:
          summary = sess.run(model.train_summ, feed_dict=feed_dict)
          train_writer.add_summary(summary, iters)
        #Clear references to data:
        train_inputs = train_outputs = feed_dict[model.image_input] = feed_dict[model.waypoint_output] = None

      val_summary, val_cost, resnet_output = sess.run([model.val_summ, model.objective, model.resnet_output], feed_dict=val_dict)
      print('Validation Summary = ', val_cost)
      for ii in plotting_data['idx']:
          plotting_data['data'][ii].append(resnet_output[ii])
      with open(plot_data_path+'/data.pickle','wb') as f:
          pickle.dump(plotting_data,f,pickle.HIGHEST_PROTOCOL)

      val_writer.add_summary(val_summary, iters)

      train_writer.flush()
      val_writer.flush()
      # Save variables
      if ((epoch + 1) % save_variables_divider == 0 or (epoch == 0) or (epoch == num_epochs - 1)):
          if epoch == 0:
            saver.save(sess, os.path.join(save_path, 'variables'), epoch)
          else:
            saver.save(sess, os.path.join(save_path, 'variables'), epoch, write_meta_graph=False)
      # Re-shuffle data after each epoch
      rand_idx = np.random.permutation(num_train_samples)
      train_indices = train_indices[rand_idx]
  train_writer.flush()
  val_writer.flush()
  print("Done")


if __name__ == '__main__':
    main()
