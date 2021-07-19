#CUDA_VISIBLE_DEVICES = 0
import pickle
from Add_noise import *
import argparse
from Robustness_Norb_models import Basenet, Basenet_l2, Basenet_Dropout, Basenet_bn, SM_CNN, OOCS
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Basenet")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--training_steps", default=1000, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
args = parser.parse_args()

# Training parameters.
learning_rate = args.lr
training_steps = args.training_steps
batch_size = args.batch_size
display_step = 1

with open('../data/norb/train/x_train_light0.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('../data/norb/train/y_train_light0.pkl', 'rb') as f:
    y_train = pickle.load(f)

x_train = np.array(x_train, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train = x_train / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


# Build neural network model.
if args.model == "Basenet":
    conv_net = Basenet()
elif args.model == "Basenet_l2":
    conv_net = Basenet_l2()
elif args.model == "Basenet_Dropout":
    conv_net = Basenet_Dropout()
elif args.model == "Basenet_bn":
    conv_net = Basenet_bn()
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

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):

    if step % display_step == 0:
        # Run the optimization to update W and b values.
        run_optimization(batch_x, batch_y)

        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
        losses.append(loss)
        accs.append(acc)


results = {}

with open('../data/norb/test/x_test_light5.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('../data/norb/test/y_test_light5.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Convert to float32.
x_test = np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_test = x_test / 255.
pred = conv_net(x_test)
loss = cross_entropy_loss(pred, y_test)
acc = accuracy(pred, y_test)
results['light5 acc'], results['light5 loss'] = acc, loss
print("test on light5 loss: %f, accuracy: %f" % (loss, acc))

with open('../data/norb/test/x_test_light1.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('../data/norb/test/y_test_light1.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Convert to float32.
x_test = np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_test = x_test / 255.
pred = conv_net(x_test)
loss = cross_entropy_loss(pred, y_test)
acc = accuracy(pred, y_test)
results['light1 acc'], results['light1 loss'] = acc, loss

print("test on light1 loss: %f, accuracy: %f" % (loss, acc))

with open('../data/norb/test/x_test_light2.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('../data/norb/test/y_test_light2.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Convert to float32.
x_test = np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_test = x_test / 255.
pred = conv_net(x_test)
loss = cross_entropy_loss(pred, y_test)
acc = accuracy(pred, y_test)
results['light2 acc'], results['light2 loss'] = acc, loss

print("test on light2 loss: %f, accuracy: %f" % (loss, acc))

with open('../data/norb/test/x_test_light3.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('../data/norb/test/y_test_light3.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Convert to float32.
x_test = np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_test = x_test / 255.
pred = conv_net(x_test)
loss = cross_entropy_loss(pred, y_test)
acc = accuracy(pred, y_test)
results['light3 acc'], results['light3 loss'] = acc, loss

print("test on light3 loss: %f, accuracy: %f" % (loss, acc))

with open('../data/norb/test/x_test_light4.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('../data/norb/test/y_test_light4.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Convert to float32.
x_test = np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_test = x_test / 255.
pred = conv_net(x_test)
loss = cross_entropy_loss(pred, y_test)
acc = accuracy(pred, y_test)
results['light4 acc'], results['light4 loss'] = acc, loss

print("test on light4 loss: %f, accuracy: %f" % (loss, acc))

with open('../data/norb/test/x_test_light0.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('../data/norb/test/y_test_light0.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Convert to float32.
x_test = np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_test = x_test / 255.
pred = conv_net(x_test)
loss = cross_entropy_loss(pred, y_test)
acc = accuracy(pred, y_test)
results['light0 acc'], results['light0 loss'] = acc, loss

print("test on light0 loss: %f, accuracy: %f" % (loss, acc))


for amount in np.arange(0.05,0.21,0.05):
    x_test_new = salt_and_pepper_gray(x_test, amount=amount)

    pred = conv_net(x_test_new)
    loss = cross_entropy_loss(pred, y_test)
    acc = accuracy(pred, y_test)
    results['light0 salt&pepper %f, acc' %amount], results['light0 salt&pepper %f, loss' %amount] = acc, loss

    print("test on salt&pepper %f, loss: %f, accuracy: %f" % (amount, loss, acc))

for dev in np.arange(0.05,0.21,0.05):
    x_test_new = gaussian_noise(x_test, stddev=dev)

    pred = conv_net(x_test_new)
    loss = cross_entropy_loss(pred, y_test)
    acc = accuracy(pred, y_test)
    results['light0 gaussian %f, acc' %dev], results['light0 gaussian %f, loss' %dev] = acc, loss

    print("test on gaussian noise stdv %f, loss: %f, accuracy: %f" % (dev, loss, acc))

# Log result in file
base_path = "../results/Robustness_Norb"
os.makedirs(base_path, exist_ok=True)
with open("{}/{}_Losses.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(losses, f)
with open("{}/{}_accs.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(accs, f)
with open("{}/{}_test_results.pkl".format(base_path, args.model), "wb") as f:
    pickle.dump(results, f)
