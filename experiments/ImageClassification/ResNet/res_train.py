from __future__ import absolute_import, division, print_function

import pickle
import keras
import h5py
import tensorflow as tf
import numpy as np
from sklearn import model_selection
from keras.preprocessing.image import ImageDataGenerator
from resnet import resnet_34, resnet_34OOCS
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=1)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=60, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--lr_drop", default=20, type=float)
parser.add_argument("--lr_decay", default=1e-4, type=float)
parser.add_argument("--gpu", default=0, type=float)
args = parser.parse_args()

# Training parameters.

image_height = 192
image_width = 192
channels = 3
num_classes = 100

BATCH_SIZE = args.batch_size
modelname = args.model
epochs = args.epochs
learning_rate = args.lr
lr_drop = args.lr_drop
lr_decay = args.lr_decay
def get_model():
    model = resnet_34OOCS()
    if modelname == "resnet34":
        model = resnet_34()
    if modelname == "resnet_34OOCS":
        model = resnet_34OOCS()

    model.build(input_shape=(None, image_height, image_width, channels))
    model.summary()
    return model


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[int(args.gpu)], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # create model
    model = get_model()

    train = h5py.File('../data/imagenetSubset/train.h5', 'r')
    test = h5py.File('../data/imagenetSubset/test.h5', 'r')
    val = h5py.File('../data/imagenetSubset/val.h5', 'r')

    x_train = train.get('images')
    y_train = train.get('labels_encoded')
    x_test = test.get('images')
    y_test = test.get('labels_encoded')
    x_val = val.get('images')
    y_val = val.get('labels_encoded')

    x_train, y_train, x_test, y_test, x_val, y_val = np.array(x_train, np.float32), np.array(y_train,
                                                                                             np.float32), np.array(
        x_test, np.float32), np.array(y_test, np.float32), np.array(x_val, np.float32), np.array(y_val, np.float32)
    # Normalize images value from [0, 255] to [0, 1].

    x_train, x_test, x_val = x_train / 255., x_test / 255., x_val / 255.

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # i
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)
    # Stochastic gradient descent optimizer.
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)


    def lr_scheduler(epoch):
        return learning_rate * (0.1 ** (epoch // lr_drop))


    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    hist = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=BATCH_SIZE),
                                  steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
                                  epochs=epochs,
                                  validation_data=(x_val, y_val), callbacks=[reduce_lr, model_checkpoint_callback],
                                  verbose=1)

    #model.load_weights("/tmp/checkpoint")
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    base_path = "../results/ImageClassification/ResNet"
    os.makedirs(base_path, exist_ok=True)
    with open("{}/{}_history.pkl".format(base_path, args.model), "wb") as f:
        pickle.dump(hist.history, f)
    #model.save_weights('./models/CR_SM1', save_format='tf')

