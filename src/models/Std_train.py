import numpy as np # for using np arrays

# for bulding and running deep learning model
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import get_source_inputs

import segmentation_models as sm
SM_FRAMEWORK=tf.keras
sm.set_framework('tf.keras')

from data.make_dataset import get_dataset
# from attacks.robustifiers import 

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import time


def train(model, img_size, batch_size, buffer_size):

    train_dataset, test_dataset = get_dataset(dataset_path, img_size, batch_size, buffer_size)
    
    dice_loss = sm.losses.DiceLoss() 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    model.compile(
        K.optimizers.Adam(0.0001),
        loss=total_loss,
        metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()],
    )
    
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    callbacks = [
        # to show samples after each epoch
        # DisplayCallback(),
        # to collect some useful metrics and visualize them in tensorboard
        tensorboard_callback,
        # if no accuracy improvements we can stop the training directly
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        # to save checkpoints
        tf.keras.callbacks.ModelCheckpoint('best_efficientnetb3_unet.h5', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    results = model.fit(
       train_dataset,
       epochs=100,
       steps_per_epoch = TRAINSET_SIZE // BATCH_SIZE,
       validation_steps= VALSET_SIZE // BATCH_SIZE,
       validation_data=dataset['val'],
       callbacks=callbacks
    )