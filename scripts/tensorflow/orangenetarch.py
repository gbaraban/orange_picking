import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import io

#From https://github.com/uzh-rpg/sim2real_drone_racing/blob/master/learning/deep_drone_racing_learner/src/ddr_learner/models/nets.py
def resnet8(img_input, output_dim, scope='Prediction', reuse=False, f=0.25, reg=True):
    """
    Define model architecture. The parameter 'f' controls the network width.
    """
    img_input = Input(tensor=img_input)
    kr = None
    if reg:
        kr = regularizers.l2(1e-4)

    with tf.variable_scope(scope, reuse=reuse):
        x1 = Conv2D(int(32*f), (5, 5), strides=[2, 2], padding='same')(img_input)
        x1 = MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x1)

        # First residual block
        x2 = Activation('relu')(x1)
        x2 = Conv2D(int(32*f), (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=kr)(x2)

        x2 = Activation('relu')(x2)
        x2 = Conv2D(int(32*f), (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=kr)(x2)

        x1 = Conv2D(int(32*f), (1, 1), strides=[2, 2], padding='same')(x1)
        x3 = add([x1, x2])

        # Second residual block
        x4 = Activation('relu')(x3)
        x4 = Conv2D(int(64*f), (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=kr)(x4)
        x4 = Activation('relu')(x4)
        x4 = Conv2D(int(64*f), (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=kr)(x4)

        x3 = Conv2D(int(64*f), (1, 1), strides=[2, 2], padding='same')(x3)
        x5 = add([x3, x4])

        # Third residual block
        x6 = Activation('relu')(x5)
        x6 = Conv2D(int(128*f), (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=kr)(x6)

        x6 = Activation('relu')(x6)
        x6 = Conv2D(int(128*f), (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=kr)(x6)

        x5 = Conv2D(int(128*f), (1, 1), strides=[2, 2], padding='same')(x5)
        x7 = add([x5, x6])

        x = Flatten()(x7)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(int(256*f))(x)
        x = Activation('relu')(x)

        # Output channel
        logits = Dense(output_dim)(x)

    return logits

class OrangeResNet:
  #def gen_image(self,new_waypoint, true_waypoint):
  #  new_waypoint = np.array(new_waypoint)
  #  true_waypoint = np.array(true_waypoint)
  #  new_waypoint = np.reshape(new_waypoint[0,:],[self.num_points,3])
  #  true_waypoint = np.reshape(true_waypoint[0,:],[self.num_points,3])
  #  figure = plt.figure()
  #  plt.plot(true_waypoint[:,0],true_waypoint[:,1],color='g')
  #  self.waypoint_list.append(new_waypoint)
  #  list_len = len(self.waypoint_list)
  #  for i in range(list_len):
  #    plt.plot(self.waypoint_list[i][:,0],self.waypoint_list[i][:,1],color=str(float(i)/list_len))
  #  buf = io.BytesIO()
  #  plt.savefig(buf, format='png')
  #  plt.close(figure)
  #  buf.seek(0)
  #  image = tf.image.decode_png(buf.getvalue(), channels=4)
  #  image = tf.expand_dims(image, 0)
  #  return image

  def __init__(self, capacity = 1, num_img = 2, num_pts = 1, focal_l = -1):
    #Parameters
    self.w = 300
    self.h = 200
    self.num_points = num_pts
    self.num_images = num_img
    self.output_dim = self.num_points*3
    self.f = capacity#5.0#2.0#1.5#125#1#0.25
    self.learning_fac_init=0.000001
    self.reg = False
    self.foc_l = focal_l
    #Inputs
    self.image_input = tf.placeholder(tf.float32,shape=[None,self.w,self.h,3*self.num_images],name='image_input')
    self.waypoint_output = tf.placeholder(tf.float32,shape=[None,self.output_dim],name="waypoints")
    #Network Architecture
    self.resnet_output = resnet8(self.image_input,output_dim=self.output_dim, f=self.f, reg=self.reg)
    #Training
    self.objective = tf.reduce_mean(tf.square(self.resnet_output - self.waypoint_output))
    self.learning_fac = tf.Variable(self.learning_fac_init)
    opt_op = tf.train.AdamOptimizer(self.learning_fac).minimize(self.objective)
    self.train_step = opt_op
    #self.iterations = tf.Variable(0)
    self.waypoint_list = []
    #self.generate_image = self.gen_image(self.resnet_output,self.waypoint_output)

    # Summary
    self.train_summ = tf.summary.scalar('Training Objective Function', self.objective)
    self.val_summ = tf.summary.scalar('Validation Objective Function', self.objective)
    self.lf_summ = tf.summary.scalar('Learning Factor', self.learning_fac)
    #self.val_image_summ = tf.summary.image('Validation Image',self.generate_image)
    self.merge = tf.summary.merge_all()
