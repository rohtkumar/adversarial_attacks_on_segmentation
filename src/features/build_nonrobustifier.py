import tensorflow as tf
from attacks.robustifiers import pgd_l2_nonrobust
import numpy as np
import time
import os
from util import logger, tools
import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE


def robust_preprocess(img, label):
    """Defines preprocessing / data augmentation for robust & nonrobust features"""
    img = tf.image.resize_with_pad(img, 128 + 4, 128 + 4)
    img = tf.image.random_crop(img, size=[128, 128, 3])
    img = tf.image.stateless_random_flip_left_right(img, (15, 13))
    return img, label


def nonrobustify_dataset(args, std_training, train_ds, ):
    original_train = []
    non_robust_train = []
    t_train = []
    y_train = []
    iters = 100
#    saved_path = os.path.join(args.save, 'nonrobustified_' + args.model_name + '_robust_ds' + time.time())
    start_time = time.time()
    progbar_train = tf.keras.utils.Progbar(train_ds)
    # Loops through entire train dataset image-by-image
    for i, (img_batch, label_batch) in enumerate(train_ds):  # Unbatch splits batches into individual images
        inter_time = time.time()

        # Create a copy of the tensor
        img_batch_t = tf.identity(img_batch)
  #      print(img_batch_t.shape)
 #       print(label_batch.shape)
        # Generate a random label, get the delta and add in the perturbation (random uniform approach)
        #t_batch = np.random.randint(low=0, high=9, size=img_batch_t.shape[0])  
        t_batch = (label_batch + 1) % 10 #  <-- deterministic approach
#        print(t_batch.shape)
        # Update the image so that it is non-robust
        learned_delta = pgd_l2_nonrobust(std_training, img_batch_t, t_batch, epsilon=0.5, alpha=0.1, num_iter=iters)
        non_robust_update = img_batch_t + learned_delta

        # Append both the original and non-robust images and labels to the respective lists
        original_train.append(img_batch)
        y_train.append(label_batch)
        non_robust_train.append(non_robust_update)
        t_train.append(t_batch)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            elapsed_tracking = time.time() - inter_time
            print(f'Unrobustified {(i + 1) * args.batch_size} images in {elapsed:0.3f} seconds; Took {elapsed_tracking:0.3f} seconds for this particular iteration')
        progbar_train.update(i)

    # Print out the shapes
    print(tf.concat(non_robust_train, axis=0).shape)
    print(tf.concat(t_train, axis=0).shape)

    # Convert to to Tensorflow dataset
    non_robust_ds = tf.data.Dataset.from_tensor_slices(
        (tf.concat(non_robust_train, axis=0), tf.concat(t_train, axis=0))).prefetch(AUTOTUNE).map \
        (robust_preprocess, num_parallel_calls=AUTOTUNE).shuffle(len(non_robust_train)).batch(args.batch_size)

    original_non_robust_ds = tf.data.Dataset.from_tensor_slices(
        (tf.concat(original_train, axis=0), tf.concat(y_train, axis=0))).prefetch(AUTOTUNE).batch(args.batch_size)

    saved_path = os.path.join(args.save, 'nonrobustified_' + args.model_name + '_non_robust_ds')
    logging.info(f'Nonrobustifier on model completed..... Saving dataset at {saved_path}')
    tf.data.experimental.save(non_robust_ds, saved_path)
    saved_path1 = os.path.join(args.save, 'nonrobustified_' + args.model_name + '_orig_non_robust_ds')
    tf.data.experimental.save(original_non_robust_ds, saved_path1)
    # logging.info(f'Nonrobustifier dataset completed..... Saving dataset at {saved_path}')

