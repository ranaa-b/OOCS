import tensorflow as tf
from tensorflow.keras import Model, layers
from Compute_Kernels import *

num_classes = 5


class Basenet(Model):

    def __init__(self):
        super(Basenet, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.bn41 = layers.BatchNormalization()
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(512, activation=tf.nn.relu)
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass _ ResNet
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 96, 96, 1])

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


class Basenet_l2(Model):

    def __init__(self):
        super(Basenet_l2, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_regularizer='l2', bias_regularizer='l2')
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_regularizer='l2', bias_regularizer='l2')
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_regularizer='l2', bias_regularizer='l2')
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_regularizer='l2', bias_regularizer='l2')
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_regularizer='l2', bias_regularizer='l2')
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_regularizer='l2', bias_regularizer='l2')
        self.bn41 = layers.BatchNormalization()
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(512, activation=tf.nn.relu, kernel_regularizer='l2')
        self.fc2 = layers.Dense(128, activation=tf.nn.relu, kernel_regularizer='l2')

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass _ ResNet
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 96, 96, 1])

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


class Basenet_Dropout(Model):

    def __init__(self):
        super(Basenet_Dropout, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.bn41 = layers.BatchNormalization()
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(512, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass _ ResNet
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 96, 96, 1])

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=is_training)
        x = self.fc2(x)
        x = self.dropout2(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


class Basenet_bn(Model):

    def __init__(self):
        super(Basenet_bn, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.bn11 = layers.BatchNormalization()
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.bn12 = layers.BatchNormalization()
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.bn21 = layers.BatchNormalization()
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.bn22 = layers.BatchNormalization()
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.bn31 = layers.BatchNormalization()
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.bn41 = layers.BatchNormalization()
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        self.fc1 = layers.Dense(512, activation=tf.nn.relu, kernel_regularizer='l2')
        self.fc2 = layers.Dense(128, activation=tf.nn.relu, kernel_regularizer='l2')

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass _ ResNet
    def call(self, x, is_training=False, is_last_step=False):
        x = tf.reshape(x, [-1, 96, 96, 1])

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.bn21(x)
        x = self.conv22(x)
        x = self.bn22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.bn31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.bn41(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=is_training)
        x = self.fc2(x)
        x = self.dropout2(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


class SM_CNN(Model):

    def __init__(self):
        super(SM_CNN, self).__init__()

        self.SM_filters = SM_Kernel(in_channels=32, out_channels=32)

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        self.fc1 = layers.Dense(512, activation=tf.nn.relu)
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass _ ResNet
    def call(self, x, is_training=False, is_last_step=False):
        x = tf.reshape(x, [-1, 96, 96, 1])

        x = self.conv11(x)
        x = self.sorround_modulation(x, in_channels=32, out_channels=32)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

    def sorround_modulation(self, input, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.SM_filters, (5, 5, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=[1, 1, 1, 1], padding='SAME')

        return output


class OOCS(Model):

    def __init__(self):
        super(OOCS, self).__init__()

        self.conv_On_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=32, out_channels=32, off=False)
        self.conv_Off_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=32, out_channels=32, off=True)

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv210 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv211 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv220 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv221 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(128)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass _ ResNet
    def call(self, x, is_training=False, is_last_step=False):
        x = tf.reshape(x, [-1, 96, 96, 1])

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x0 = self.conv210(x)
        sm_on = self.sorround_modulation_DoG_on(x, kernel_size=5, in_channels=32, out_channels=32) + x
        sm_on = sm_on + x0
        sm_on = self.conv220(sm_on)
        x1 = self.conv211(x)
        sm_off = self.sorround_modulation_DoG_off(x, kernel_size=5, in_channels=32, out_channels=32) + x
        sm_off = sm_off + x1
        sm_off = self.conv221(sm_off)
        x = layers.concatenate([sm_on, sm_off], axis=-1)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

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