import os

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import get_source_inputs
from attacks.robustifiers import pgd_l2_robust
import numpy as np
import time
from util import logger, tools
import logging


def robust_preprocess(img, label):
    """Defines preprocessing / data augmentation for robust & nonrobust features"""
    img = tf.image.resize_with_pad(img, 128+4, 128+4)
    img = tf.image.random_crop(img, size=[128, 128, 3])
    img = tf.image.stateless_random_flip_left_right(img, (15, 13))
    return img, label


def robustify(args, robust_model, train_ds, iters=1000, alpha=0.1):
    #%%time
    with logger.LoggingBlock("Robustifier started", emph=True):
        robust_train = []
        orig_labels = []
        example = False

        saved_path = os.path.join(args.save, 'robustified_' + args.model_name + '_robust_ds')
        batch_train = len(train_ds) // args.batch_size
        train_temp = train_ds.take(1152)

        train_to_pull = list(iter(train_temp))
        start_rn = np.random.randint(0, len(train_temp))
        rand_batch = train_to_pull[start_rn][0]
        logging.info(f'Total trainset length are {len(train_temp)}')
        start_time = time.time()
        progbar_train = tf.keras.utils.Progbar(train_temp)
        for i, (img_batch, label_batch) in enumerate(train_temp):
            logging.info(f'Image shape {img_batch.shape} and Mask shape {label_batch.shape}')
            inter_time = time.time()

            # For the last batch, it is smaller than batch_size and thus we match the size for the batch of initial images
            if img_batch.shape[0] < args.batch_size:
                logging.info("Input train image batch smaller than batch size")
                rand_batch = rand_batch[:img_batch.shape[0]]

            # Get the goal representation
            goal_representation = robust_model(img_batch)
            # logging.info(f'Total batches are {batch_train}')
            # Upate the batch of images
            learned_delta = pgd_l2_robust(robust_model, rand_batch, goal_representation, alpha=alpha, num_iter=iters)
            robust_update = (rand_batch + learned_delta)

            # Add the updated images and labels to their respective lists
            robust_train.append(robust_update)
            orig_labels.append(label_batch)

            # Measure the time
            if (i+1) % 10 == 0:
                elapsed = time.time() - start_time
                elapsed_tracking = time.time() - inter_time
                logging.info(f'Robustified {(i+1)*args.batch_size} images in {elapsed:0.3f} seconds; Took {elapsed_tracking:0.3f} seconds for this particular iteration')

            robust_ds = tf.data.Dataset.from_tensor_slices(
                (tf.concat(robust_train, axis=0), tf.concat(orig_labels, axis=0))) \
                .prefetch(tf.data.experimental.AUTOTUNE).map(robust_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .shuffle(len(robust_train)).batch(args.batch_size)
            tools.save_dataset(robust_ds, saved_path)
          # Reset random image batch
            rn = np.random.randint(0, len(train_temp)-1) # -1 because last batch might be smaller
            rand_batch = train_to_pull[rn][0]

            progbar_train.update(i)

        # Convert to TensorFlow Dataset
        # robust_ds = tf.data.Dataset.from_tensor_slices((tf.concat(robust_train, axis=0), tf.concat(orig_labels, axis=0)))\
        #     .prefetch(tf.data.experimental.AUTOTUNE).map(robust_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        #     .shuffle(len(robust_train)).batch(args.batch_size)


        logging.info(f'Robustifier dataset completed..... Saving dataset at {saved_path}')
        # tools.save_dataset(robust_ds, saved_path)
        # tf.data.experimental.save(robust_ds, '../data/robustified/'+args.model_name+'_robust_ds'+time.time())

        return robust_ds



  

