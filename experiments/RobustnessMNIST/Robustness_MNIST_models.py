import tensorflow as tf
from tensorflow.keras import Model, layers
from Compute_Kernels import *

num_classes = 10

class Basenet(Model):

    def __init__(self):
        super(Basenet, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        self.fc1 = layers.Dense(100, activation=tf.nn.relu)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)

        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


class SM_CNN(Model):

    def __init__(self):
        super(SM_CNN, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        self.fc1 = layers.Dense(100, activation=tf.nn.relu)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])

        x = self.conv11(x)
        x = self.sorround_modulation_DoG_pic(x, in_channels=32, out_channels=32)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)

        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

    def sorround_modulation_DoG_pic(self, input, in_channels=64, out_channels=64):
        full1 = tf.fill([in_channels, out_channels], 0.17)
        full2 = tf.fill([in_channels, out_channels], 0.49)
        full3 = tf.fill([in_channels, out_channels], 1.0)
        full4 = tf.fill([in_channels, out_channels], -0.27)
        full5 = tf.fill([in_channels, out_channels], -0.23)
        full6 = tf.fill([in_channels, out_channels], -0.18)
        center = [[full4, full5, full6, full5, full4],
                  [full5, full1, full2, full1, full5],
                  [full6, full2, full3, full2, full6],
                  [full5, full1, full2, full1, full5],
                  [full4, full5, full6, full5, full4]]

        new_weights = tf.Variable(tf.reshape(center, (5, 5, in_channels, out_channels)), trainable=False)
        output = tf.nn.conv2d(input, filters=new_weights, strides=[1, 1, 1, 1], padding='SAME')

        return output


class OOCS(Model):

    def __init__(self):
        super(OOCS, self).__init__()
        self.conv_On_filters = On_Off_Center_filters(radius=1.0, gamma=1. / 2., in_channels=1, out_channels=1, off=False)
        self.conv_Off_filters = On_Off_Center_filters(radius=1.0, gamma=1. / 2., in_channels=1, out_channels=1, off=True)

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        self.fc1 = layers.Dense(100, activation=tf.nn.relu)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])

        sm_on = self.on_center_modulation_small(x, kernel_size=3, in_channels=1, out_channels=1)
        sm_off = self.off_center_modulation_small(x, kernel_size=3, in_channels=1, out_channels=1)
        x = sm_on + sm_off

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)

        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

    def on_center_modulation_small(self, kernel_size, input, in_channels, out_channels):
        filter_weights = tf.Variable(tf.cast(tf.reshape(self.conv_On_filters, (kernel_size, kernel_size, in_channels, out_channels)),dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=1, padding='SAME')

        return tf.nn.relu(output)

    def off_center_modulation_small(self, kernel_size, input, in_channels, out_channels):
        filter_weights = tf.Variable(tf.cast(tf.reshape(self.conv_Off_filters, (kernel_size, kernel_size, in_channels, out_channels)),dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=1, padding='SAME')

        return tf.nn.relu(output)