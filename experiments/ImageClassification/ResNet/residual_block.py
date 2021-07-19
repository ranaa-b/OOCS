import tensorflow as tf
import numpy as np
from Compute_Kernels import *

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output



class BasicBlockOOCS(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlockOOCS, self).__init__()
        self.conv11 = tf.keras.layers.Conv2D(filters=filter_num/2,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.bn11 = tf.keras.layers.BatchNormalization()
        self.conv12 = tf.keras.layers.Conv2D(filters=filter_num/2,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.bn12 = tf.keras.layers.BatchNormalization()

        self.conv21 = tf.keras.layers.Conv2D(filters=filter_num / 2,
                                             kernel_size=(3, 3),
                                             strides=1,
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.bn21 = tf.keras.layers.BatchNormalization()
        self.conv22 = tf.keras.layers.Conv2D(filters=filter_num / 2,
                                             kernel_size=(3, 3),
                                             strides=1,
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.bn22 = tf.keras.layers.BatchNormalization()

        self.conv_On_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=32, out_channels=32,
                                                     off=False)
        self.conv_Off_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=32, out_channels=32,
                                                      off=True)


        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=1))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x0 = self.conv11(inputs)
        x0 = self.bn11(x0)
        x0 = tf.nn.relu(x0)
        sm_on = self.sorround_modulation_DoG_on(inputs, kernel_size=5, in_channels=32, out_channels=32) + inputs
        sm_on = sm_on + x0
        sm_on = self.conv12(sm_on)
        sm_on = self.bn12(sm_on, training=training)

        x1 = self.conv21(inputs)
        x1 = self.bn21(x1)
        x1 = tf.nn.relu(x1)
        sm_off = self.sorround_modulation_DoG_off(inputs, kernel_size=5, in_channels=32, out_channels=32) + inputs
        sm_off = sm_off + x1
        sm_off = self.conv22(sm_off)
        sm_off = self.bn22(sm_off, training=training)

        x = tf.keras.layers.concatenate([sm_on, sm_off], axis=-1)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

    def sorround_modulation_DoG_on(self, input, kernel_size, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.conv_On_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=1, padding='SAME')

        return tf.nn.relu(output)

    def sorround_modulation_DoG_off(self, input, kernel_size, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.conv_Off_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=1, padding='SAME')

        return tf.nn.relu(output)



def make_basic_block_layer_OOCS(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlockOOCS(filter_num, stride=2))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


