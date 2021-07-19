import tensorflow as tf
from tensorflow.keras import Model, layers
from Compute_Kernels import *
num_classes = 100


class Basenet0(Model):

    def __init__(self):
        super(Basenet0, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()

        # Fully connected layers.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

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
        x = self.conv51(x)
        x = self.maxpool5(x)
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


class Basenet1(Model):

    def __init__(self):
        super(Basenet1, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv23 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()

        # Fully connected layers.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv23(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
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


class Basenet2(Model):

    def __init__(self):
        super(Basenet2, self).__init__()

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv23 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.he_normal())
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()

        # Fully connected layers.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv23(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
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

class Basenet3(Model):

    def __init__(self):
        super(Basenet3, self).__init__()

        self.conv0 = layers.Conv2D(64, kernel_size=1, strides=2, padding='same')
        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()

        # Fully connected layers.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])
        residual = self.conv0(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = x + residual
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
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


class SM0(Model):

    def __init__(self):
        super(SM0, self).__init__()

        self.SM_filters = SM_Kernel(in_channels=32, out_channels=32)

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layers.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

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
        x = self.conv51(x)
        x = self.maxpool5(x)
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

    def sorround_modulation(self, input, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.SM_filters, (5, 5, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=[1, 1, 1, 1], padding='SAME')

        return output

class SM1(Model):

    def __init__(self):
        super(SM1, self).__init__()

        self.SM_filters = SM_kernel_DoG(kernel_size=5, sigma_e=1.2, sigma_i=1.4, in_channels=32, out_channels=32)

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layers.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

        x = self.conv11(x)
        x = self.sorround_modulation_DoG(x, kernel_size=5, in_channels=32, out_channels=32)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
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

    def sorround_modulation_DoG(self, kernel_size, input, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.SM_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=[1, 1, 1, 1], padding='SAME')

        return output

class SM2(Model):

    def __init__(self):
        super(SM2, self).__init__()
        self.OOCS_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=32, out_channels=32, off=False)
        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv22 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layers.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

        x = self.conv11(x)
        x = x + self.sorround_modulation_OOCS(x, kernel_size=5, in_channels=32, out_channels=32)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
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

    def sorround_modulation_OOCS(self, kernel_size, input, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.OOCS_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=[1, 1, 1, 1], padding='SAME')

        return output


class OOCS0(Model):

    def __init__(self):
        super(OOCS0, self).__init__()

        self.conv_On_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=3, out_channels=32,
                                                     off=False)
        self.conv_Off_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=3, out_channels=32,
                                                      off=True)

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv210 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv211 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv220 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.conv221 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

        sm_on = self.sorround_modulation_DoG_on(x, kernel_size=5, in_channels=3, out_channels=32)
        sm_off = self.sorround_modulation_DoG_off(x, kernel_size=5, in_channels=3, out_channels=32)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)

        x0 = self.conv210(x)
        sm_on = sm_on + x0
        sm_on = self.conv220(sm_on)

        x1 = self.conv211(x)
        sm_off = sm_off + x1
        sm_off = self.conv221(sm_off)
        x = layers.concatenate([sm_on, sm_off], axis=-1)

        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
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

    def sorround_modulation_DoG_on(self, input, kernel_size, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.conv_On_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=2, padding='SAME')

        return output

    def sorround_modulation_DoG_off(self, input, kernel_size, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.conv_Off_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=2, padding='SAME')

        return output



class OOCS1(Model):

    def __init__(self):
        super(OOCS1, self).__init__()

        self.conv_On_filters = On_Off_Center_filters(radius=1.0, gamma=1. / 2., in_channels=3, out_channels=32,
                                                     off=False)
        self.conv_Off_filters = On_Off_Center_filters(radius=1.0, gamma=1. / 2., in_channels=3, out_channels=32,
                                                      off=True)
        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv210 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv211 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv220 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv221 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

        sm_on = self.sorround_modulation_DoG_on(x, kernel_size=3, in_channels=3, out_channels=32)
        sm_off = self.sorround_modulation_DoG_off(x, kernel_size=3, in_channels=3, out_channels=32)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)

        x0 = self.conv210(x)
        sm_on = sm_on + x0
        sm_on = self.conv220(sm_on)

        x1 = self.conv211(x)
        sm_off = sm_off + x1
        sm_off = self.conv221(sm_off)
        x = layers.concatenate([sm_on, sm_off], axis=-1)

        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
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

    def sorround_modulation_DoG_on(self, input, kernel_size, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.conv_On_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=2, padding='SAME')

        return output

    def sorround_modulation_DoG_off(self, input, kernel_size, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.conv_Off_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=2, padding='SAME')

        return output

class OOCS2(Model):

    def __init__(self):
        super(OOCS2, self).__init__()

        self.conv_On_filters = Averaged_Kernel(radius=2.0, gamma=2. / 3., in_channels=3, out_channels=32,
                                                     off=False)
        self.conv_Off_filters = Averaged_Kernel(radius=2.0, gamma=2. / 3., in_channels=3, out_channels=32,
                                                      off=True)
        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv210 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv211 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv220 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv221 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

        sm_on = self.sorround_modulation_DoG_on(x, kernel_size=5, in_channels=3, out_channels=32)
        sm_off = self.sorround_modulation_DoG_off(x, kernel_size=5, in_channels=3, out_channels=32)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1(x)

        x0 = self.conv210(x)
        sm_on = sm_on + x0
        sm_on = self.conv220(sm_on)

        x1 = self.conv211(x)
        sm_off = sm_off + x1
        sm_off = self.conv221(sm_off)
        x = layers.concatenate([sm_on, sm_off], axis=-1)

        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
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

    def sorround_modulation_DoG_on(self, input, kernel_size, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.conv_On_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=2, padding='SAME')

        return output

    def sorround_modulation_DoG_off(self, input, kernel_size, in_channels, out_channels):
        filter_weights = tf.Variable(
            tf.cast(tf.reshape(self.conv_Off_filters, (kernel_size, kernel_size, in_channels, out_channels)),
                    dtype=tf.float32), trainable=False, dtype=tf.float32)
        output = tf.nn.conv2d(input, filters=filter_weights, strides=2, padding='SAME')

        return output


class OOCS3(Model):

    def __init__(self):
        super(OOCS3, self).__init__()

        self.conv_On_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=32, out_channels=32,
                                                     off=False)
        self.conv_Off_filters = On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=32, out_channels=32,
                                                      off=True)

        self.conv11 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.conv12 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv210 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv211 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv220 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.conv221 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                    kernel_initializer=tf.keras.initializers.he_normal())
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        self.flatten = layers.Flatten()
        # Fully connected layer.
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 192, 192, 3])

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
        x = self.conv51(x)
        x = self.maxpool5(x)
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