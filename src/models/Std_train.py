# for bulding and running deep learning model
import datetime
import os

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from util import logger

SM_FRAMEWORK=tf.keras

sm.set_framework('tf.keras')

# from data.make_dataset import get_dataset
# from attacks.robustifiers import


def train(args, train_dataset, val_dataset):
    with logger.LoggingBlock("Start Standard Training", emph=True):
        saved_model = os.path.join(args.save, 'best_' + args.model_name + '_unet.h5')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(args.save)
        callbacks = [
            # to show samples after each epoch
            # DisplayCallback(),
            # to collect some useful metrics and visualize them in tensorboard
            tensorboard_callback,
            # if no accuracy improvements we can stop the training directly
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            # to save checkpoints
            tf.keras.callbacks.ModelCheckpoint( saved_model , verbose=1, save_best_only=True,
                                               save_weights_only=False)
        ]
        results = args.std_model.fit(train_dataset, epochs=100, validation_steps=len(val_dataset) // args.batch_size,
                                     validation_data=val_dataset, callbacks=callbacks)
        logger.info("Average test loss: " + np.average(results.history['loss']))


