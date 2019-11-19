import tensorflow as tf
import numpy as np
import argparse
import pickle
import os
from orangenetarch import *
from datetime import datetime
import matplotlib.image as mpimg

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
    with open(run_dir+"/"+trial_dir+"/metadata.pickl",'rb') as data_f:
      data = pickle.load(data_f, encoding='latin1')
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

def parseFiles(idx,traj_data,trial_dir):
  image_idx = idx[0]
  image = img.open(trial_dir+'image'+str(image_idx)+'.png')
  image = np.array(image.getdata()).reshape(image.size[0],image.size[1],4)
  image = image[:,:,0:3]/255.0 #Cut out alpha
  image = image/255.0
  R0 = np.array(traj_data[image_idx][1])
  p0 = np.array(traj_data[image_idx][0])
  local_pts = [np.matmul(R0.T,np.array(x[0])-p0) for x in traj_data[idx]]
  local_pts = np.array(local_pts[1:])
  return image, local_pts

def loadData(idx,run_dir,output_dim,time_window = 5):
  num_points = output_dim/3
  #Assume reduced N is constant for all trials
  trial_list = os.listdir(run_dir)
  trial_dir = trial_list[0]
  reduced_N = -1
  with open(run_dir+"/"+trial_dir+"/metadata.pickl",'rb') as data_f:
    data = pickle.load(data_f, encoding='latin1')
    N = data['N']
    tf = data['tf']
    h = float(N)/tf
    reduced_N = int(N - time_window*h)
  images = []
  waypoints = []
  trial_num = np.floor(idx/reduced_N)
  image_num = np.mod(idx,reduced_N)
  for trial_idx, image_idx in zip(trial_num,image_num):
    trial_dir = run_dir+"/"+trial_list[trial_idx]+"/"
    traj_data = []
    with open(trial_dir+"trajdata.pickle",'rb') as data_f:
      traj_data = pickle.load(data_f, encoding='latin1')
    metadata = []
    with open(trial_dir+"metadata.pickl",'rb') as data_f:
      metadata = pickle.load(data_f, encoding='latin1')
    idx_final = image_idx + metadata['N']*time_window/metadata['tf']
    offset_idx = np.floor(np.linspace(image_idx,idx_final,num_points+1))
    image, waypoint = parseFiles(offset_idx,traj_data,trial_dir)
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
  args = parser.parse_args()

  # Model and optimization params
  val_perc = 0.1
  #g_depths = [64, 64, 64]
  #f_depths = [64, 64, 64]
  batch_size = 32
  num_epochs = args.epochs
  #start_learn_rate = 1e-4
  learn_rate_decay = 2.5 / num_epochs
  save_variables_divider = 10
  log_path = './model/logs'
  save_path = createStampedFolder(os.path.join(log_path, 'variable_log'))

  ######################

  # Make model
  print ('Building model')
  model = OrangeResNet()

  # Load in Data
  train_indices, val_indices = parseDirData(args.data, args.seed, args.resample, val_perc)
  num_train_samples = train_indices.shape[0]

  # Train model
  print ('Training...')
  val_inputs, val_outputs = loadData(val_indices,args.data, model.output_dim)
  val_dict = {model.image_input: val_inputs, model.waypoint_output: val_outputs}

  train_path = addTimestamp(os.path.join(log_path, 'train_'))
  val_path = addTimestamp(os.path.join(log_path, 'validation_'))
  train_writer = tf.summary.FileWriter(train_path, graph=tf.get_default_graph())
  val_writer = tf.summary.FileWriter(val_path, graph=tf.get_default_graph())

  saver = tf.train.Saver()
  init = tf.global_variables_initializer()
  feed_dict = {model.keep_prob: 0.9}

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
    iters = 0
    for epoch in range(num_epochs):
      print('Epoch: ', epoch)
      batch_idx = 0
      # Decay learning rate
      model.learning_fac.assign(np.exp(-epoch*model.learn_rate_decay)*model.learning_fac_init)
      while batch_idx < num_train_samples:
        end_idx = min(batch_idx + batch_size, num_train_samples)
        train_inputs, train_outputs = loadData(train_indices[batch_idx:end_idx],args.data)
        feed_dict[model.image_input] = train_inputs
        feed_dict[model.waypoint_output] = train_outputs
        #sess.run([model.train_summary_op, model.train_step], feed_dict=feed_dict)
        sess.run(model.train_step, feed_dict=feed_dict)
        batch_idx = batch_idx + batch_size
        iters = iters + 1
        if iters % 20 == 0:
          summary = sess.run(model.train_summ, feed_dict=feed_dict)
          train_writer.add_summary(summary, iters)
        train_inputs = train_outputs = feed_dict[model.image_imput] = feed_dict[model.true_knots] = None

      val_summary, val_cost = sess.run([model.val_summ, model.objective], feed_dict=val_dict)
      print('Validation Summary = ', val_cost)

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
      train_inputs = train_inputs[rand_idx, :]
      train_outputs = train_outputs[rand_idx, :]
  train_writer.flush()
  val_writer.flush()
  print("Done")


if __name__ == '__main__':
    main()
