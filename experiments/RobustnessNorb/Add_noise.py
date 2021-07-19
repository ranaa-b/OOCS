import numpy as np
import tensorflow as tf

seed = 2468642

def salt_and_pepper_gray(images, salt_vs_pepper = 0.5, amount = 0.01):
   # Need to produce a copy as to not modify the original image
   altered_images = images.copy()
   row, col = altered_images[0].shape
   num_salt = np.ceil(amount * altered_images[0].size * salt_vs_pepper)
   num_pepper = np.ceil(amount * altered_images[0].size * (1.0 - salt_vs_pepper))
   np.random.seed(seed)
   for X_img in altered_images:
      # Add Salt noise
      coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
      X_img[coords[0], coords[1]] = 1

      # Add Pepper noise
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
      X_img[coords[0], coords[1]] = 0
   return altered_images


def gaussian_noise(images, stddev=0.1):

    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=stddev,
                             dtype=tf.float32, seed=seed)
    altered_images = tf.add(images, noise)
    return altered_images