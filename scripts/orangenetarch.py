import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras import regularizers

#From https://github.com/uzh-rpg/sim2real_drone_racing/blob/master/learning/deep_drone_racing_learner/src/ddr_learner/models/nets.py
def resnet8(img_input, output_dim, scope='Prediction', reuse=False, f=0.25):
    """
    Define model architecture. The parameter 'f' controls the network width.
    """
    img_input = Input(tensor=img_input)

    with tf.variable_scope(scope, reuse=reuse):
        x1 = Conv2D(int(32*f), (5, 5), strides=[2, 2], padding='same')(img_input)
        x1 = MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x1)

        # First residual block
        x2 = Activation('relu')(x1)
        x2 = Conv2D(int(32*f), (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x2)

        x2 = Activation('relu')(x2)
        x2 = Conv2D(int(32*f), (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x2)

        x1 = Conv2D(int(32*f), (1, 1), strides=[2, 2], padding='same')(x1)
        x3 = add([x1, x2])

        # Second residual block
        x4 = Activation('relu')(x3)
        x4 = Conv2D(int(64*f), (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x4)
        x4 = Activation('relu')(x4)
        x4 = Conv2D(int(64*f), (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x4)

        x3 = Conv2D(int(64*f), (1, 1), strides=[2, 2], padding='same')(x3)
        x5 = add([x3, x4])

        # Third residual block
        x6 = Activation('relu')(x5)
        x6 = Conv2D(int(128*f), (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x6)

        x6 = Activation('relu')(x6)
        x6 = Conv2D(int(128*f), (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x6)

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
  def __init__(self):
    #Parameters
    self.num_points = 5
    self.output_dim = self.num_points*3
    self.f = 0.25
    #Inputs
    self.image_input = tf.placeholder(tf.float32,shape=[None,rows,columns,1],name='image_input')#TODO: ask if dict is better than generate batches/cond method
    self.waypoint_output = tf.placeholder(tf.float32,shape=[None,self.output_dim],name="waypoints")#
    #Network Architecture
    self.resnet_output = resnet8(image_input,output_dim=self.output_dim, f=self.f)
    #Training
    self.objective = tf.reduce_mean(tf.square(self.resnet_output - self.waypoint_output))

    self.learning_fac = tf.Variable(self.learning_fac_init)
    opt_op = tf.train.AdamOptimizer(self.learning_fac).minimize(self.objective)
    self.train_step = opt_op
    #self.iterations = tf.Variable(0)

    # Summary
    self.train_summ = tf.summary.scalar('Training Objective Function', self.objective)
    self.val_summ = tf.summary.scalar('Validation Objective Function', self.objective)
    self.lf_summ = tf.summary.scalar('Learning Factor', self.learning_fac)
    self.merge = tf.summary.merge_all()
