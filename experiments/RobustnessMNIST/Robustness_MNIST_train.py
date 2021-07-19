from random import Random, random
CUDA_VISIBLE_DEVICES = 0
import tensorflow as tf
from sklearn import model_selection
import numpy as np
import argparse
from Robustness_MNIST_models import Basenet, SM_CNN, OOCS
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Basenet")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--lr_drop", default=5, type=float)
args = parser.parse_args()

# Training parameters.
learning_rate = args.lr
epochs = args.epochs
batch_size = args.batch_size
lr_drop = args.lr_drop
lr_decay = 1e-6


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test_dark = []
for x in x_test:
    x_test_dark.append(abs(x - 255.0))

x_test_dark = np.array(x_test_dark)


# Normalize images value from [0, 255] to [0, 1].
x_train, x_test, x_test_dark = x_train / 255., x_test / 255., x_test_dark / 255.
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=42)
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.shuffle(5000).batch(batch_size).prefetch(1)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(5000).batch(batch_size).prefetch(1)

# Build neural network model.
if args.model == "Basenet":
    conv_net = Basenet()
elif args.model == "SM":
    conv_net = SM_CNN()
elif args.model == "OOCS":
    conv_net = OOCS()
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


# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

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

# Run training for the given number of steps.
for epoch in range(epochs):
    sum_acc, sum_loss, sum_val_acc, sum_val_loss = 0., 0., 0., 0.
    optimizer.learning_rate = learning_rate * (0.5 ** (epoch // lr_drop))

    for step, (batch_x, batch_y) in enumerate(train_data.take(-1), 1):
        # Run the optimization to update W and b values.
        run_optimization(batch_x, batch_y)

        pred = conv_net(batch_x)
        sum_loss += cross_entropy_loss(pred, batch_y)
        sum_acc += accuracy(pred, batch_y)

        val_batch_x, val_batch_y = list(val_data.take(1))[0]
        val_pred = conv_net(val_batch_x)
        sum_val_loss += cross_entropy_loss(val_pred, val_batch_y)
        sum_val_acc += accuracy(val_pred, val_batch_y)

    acc, loss, val_acc, val_loss = sum_acc / step, sum_loss / step, sum_val_acc / step, sum_val_loss / step
    print("epoch: %i, step:%i, loss: %f, accuracy: %f" % (epoch, step, loss, acc))

    print("validation epoch: %i, loss: %f, accuracy: %f" % (epoch, val_loss, val_acc))
    losses.append(loss)
    accs.append(acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

results = {}
pred = conv_net(x_test)
loss = cross_entropy_loss(pred, y_test)
acc = accuracy(pred, y_test)
results['test acc on original'] = acc
results['test loss on original'] = loss
print("test on original data, loss: %f, accuracy: %f" % (loss, acc))

pred = conv_net(x_test_dark)
loss = cross_entropy_loss(pred, y_test)
acc = accuracy(pred, y_test)
results['test acc on inverted'] = acc
results['test loss on inverted'] = loss
print("test on inverted data, loss: %f, accuracy: %f" % (loss, acc))

# Log result in file
base_path = "../results/Robustness_MNIST"
os.makedirs(base_path, exist_ok=True)
with open("{}/{}_test_results.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(results, f)




