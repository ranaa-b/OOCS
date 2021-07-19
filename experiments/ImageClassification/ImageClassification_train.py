#CUDA_VISIBLE_DEVICES = 0
import pickle
import h5py
import tensorflow as tf
import numpy as np
from ImageClassification_models import Basenet0, Basenet1, Basenet2, Basenet3, SM0, SM1, SM2, OOCS0, OOCS1, OOCS2, OOCS3
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Basenet0")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--lr_drop", default=10, type=float)
args = parser.parse_args()

# Training parameters.
learning_rate = args.lr
lr_drop = args.lr_drop
epochs = args.epochs
batch_size = args.batch_size

train = h5py.File('../data/imagenetSubset/train.h5', 'r')
test = h5py.File('../data/imagenetSubset/test.h5', 'r')
val = h5py.File('../data/imagenetSubset/val.h5', 'r')

x_train = train.get('images')
y_train = train.get('labels_encoded')
x_test = test.get('images')
y_test = test.get('labels_encoded')
x_val = val.get('images')
y_val = val.get('labels_encoded')


x_train, x_test, x_val = np.array(x_train, np.float32), np.array(x_test, np.float32), np.array(x_val, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test, x_val = x_train / 255., x_test / 255., x_val / 255.

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(5000).batch(batch_size).prefetch(1)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.batch(batch_size).prefetch(1)

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.shuffle(5000).batch(batch_size).prefetch(1)


# Build neural network model.
if args.model == "Basenet0":
    conv_net = Basenet0()
elif args.model == "Basenet1":
    conv_net = Basenet1()
elif args.model == "Basenet2":
    conv_net = Basenet2()
elif args.model == "Basenet3":
    conv_net = Basenet3()
elif args.model == "SM0":
    conv_net = SM0()
elif args.model == "SM1":
    conv_net = SM1()
elif args.model == "SM2":
    conv_net = SM2()
elif args.model == "OOCS0":
    conv_net = OOCS0()
elif args.model == "OOCS1":
    conv_net = OOCS1()
elif args.model == "OOCS2":
    conv_net = OOCS2()
elif args.model == "OOCS3":
    conv_net = OOCS3()
else:
    raise ValueError("Unknown model type '{}'".format(args.model))


# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    y = tf.reshape(y, [-1])
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    y = tf.cast(y_true, tf.int64)
    y = tf.reshape(y, [-1])

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), y)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


optimizer = tf.optimizers.Adam(learning_rate)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = conv_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


losses = []
accs = []
losses = []
accs = []
val_losses = []
val_accs = []
validation = list(val_data.take(-1))
# Run training for the given number of epochs.
for epoch in range(epochs):
    sum_acc, sum_loss, sum_val_acc, sum_val_loss = 0., 0., 0., 0.
    if epoch == lr_drop:
        optimizer.learning_rate = learning_rate / 2.

    for step, (batch_x, batch_y) in enumerate(train_data.take(-1), 1):
        # Run the optimization to update W and b values.
        # print(batch_x[0], batch_y[0])
        run_optimization(batch_x, batch_y)

        pred = conv_net(batch_x)
        sum_loss += cross_entropy_loss(pred, batch_y)
        sum_acc += accuracy(pred, batch_y)

        val_batch_x, val_batch_y = list(val_data.take(1))[0]
        val_pred = conv_net(val_batch_x)
        sum_val_loss += cross_entropy_loss(val_pred, val_batch_y)
        sum_val_acc += accuracy(val_pred, val_batch_y)

    acc, loss, val_acc, val_loss = sum_acc / step, sum_loss / step, sum_val_acc / step, sum_val_loss / step
    print("epoch: %i, step:%i, loss: %f, accuracy: %f" % (step, epoch, loss, acc))

    print("validation epoch: %i, loss: %f, accuracy: %f" % (epoch, val_loss, val_acc))
    losses.append(loss)
    accs.append(acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

test_losses = []
test_accs = []
for step, (batch_x, batch_y) in enumerate(test_data.take(-1), 1):
    pred = conv_net(batch_x)
    loss = cross_entropy_loss(pred, batch_y)
    acc = accuracy(pred, batch_y)
    test_losses.append(loss)
    test_accs.append(acc)
print(np.array(test_accs).mean())

print(np.array(test_losses).mean())


# Log result in file
base_path = "../results/ImageClassification"
os.makedirs(base_path, exist_ok=True)
with open("{}/{}_Losses.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(losses, f)
with open("{}/{}_accs.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(accs, f)
with open("{}/{}_val_losses.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(val_losses, f)
with open("{}/{}_val_accs.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(val_accs, f)
with open("{}/{}_test_acc.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(np.array(test_accs).mean(), f)
with open("{}/{}_test_loss.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(np.array(test_losses).mean(), f)
