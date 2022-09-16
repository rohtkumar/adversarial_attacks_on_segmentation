import tensorflow as tf
from tensorflow import keras as K
K.backend.set_image_data_format('channels_last')
from tensorflow.keras.utils import get_source_inputs

import segmentation_models as sm
<<<<<<< HEAD
SM_FRAMEWORK=tf.keras
=======
from segmentation_models import get_preprocessing

K.backend.set_image_data_format('channels_last')
SM_FRAMEWORK = tf.keras
>>>>>>> 71f5dd9197c4d1e01a17496f08cd286e7dcf6275
sm.set_framework('tf.keras')
from segmentation_models import get_preprocessing



def initialize_std_model(backbone, classes, activation):
    # BACKBONE = 'efficientnetb3' 'resnet50'
    preprocess_input = get_preprocessing(backbone)
    model = sm.Unet(BACKBONE ,input_shape=(128, 128, 3),classes=classes,activation=activation, encoder_weights='imagenet')
    return model

<<<<<<< HEAD
=======

def init_adv_model(args, classes, activation):
    dice_loss = args.training_loss1
    focal_loss = args.training_loss2
    total_loss = dice_loss + (1 * focal_loss)
    model = get_model(args, classes, activation)
    model.compile(
        K.optimizers.Adam(0.0001),
        loss=total_loss,
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)],
    )
    # boundaries = [15000, 20000]
    # values = [0.1, 0.01, 1e-3]
    # learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries, values)
    # tf.keras.backend.clear_session()
    # model.compile(
    #     loss=total_loss,
    #     optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_fn,
    #                                       momentum=0.9),
    #     metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), "accuracy"]
    # )
    return model


def get_robust_model(args):
    representation = args.adv_model.layers[-6]
    robustifier = tf.keras.Model(inputs=args.adv_model.layers[0].input,
                                 outputs=representation.output)
    return robustifier


>>>>>>> 71f5dd9197c4d1e01a17496f08cd286e7dcf6275
def load_weights(path, model):
    model.load_weights(path)
    return model
