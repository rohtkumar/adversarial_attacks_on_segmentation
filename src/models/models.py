import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import get_source_inputs
import segmentation_models as sm
from segmentation_models import get_preprocessing
import logging
from util import logger, tools
K.backend.set_image_data_format('channels_last')
SM_FRAMEWORK = tf.keras
sm.set_framework('tf.keras')


def get_model(args, classes, activation):

    if(args.model == 'Unet'):
        model = sm.Unet('resnet50', input_shape=(args.img_size, args.img_size, 3), classes=classes, activation=activation,
                    encoder_weights='imagenet')
    else:
        model = sm.Linknet('resnet50', input_shape=(args.img_size, args.img_size, 3), classes=classes,
                        activation=activation,
                        encoder_weights='imagenet')
    return model


def initialize_std_model(args, classes, activation):

    model = get_model(args, classes, activation)

    dice_loss = args.training_loss1
    focal_loss = args.training_loss2
    total_loss = dice_loss + (1 * focal_loss)
    # model = args.stdmodel
    model.compile(
        K.optimizers.Adam(0.0001),
        # loss=total_loss,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[sm.metrics.IOUScore(), 'accuracy']
    )

    return model

def initialize_std_model_test(args, classes, activation):
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [15000, 20000], [0.1, 0.01, 1e-3])

    model = get_model(args, classes, activation)
    logging.info(f'Created model {model}')
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_fn,
                                          momentum=0.9),
        # optimizer = tf.keras.optimizers.Adam(),
        metrics=[sm.metrics.IOUScore(), 'accuracy']
    )
    logging.info(f'Compiled model {model}')
    return model

def init_adv_model(args, classes, activation):
    # dice_loss = args.training_loss1
    # focal_loss = args.training_loss2
    # total_loss = dice_loss + (1 * focal_loss)
    model = get_model(args, classes, activation)
    model.compile(
        K.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[sm.metrics.IOUScore(threshold=0.5), 'accuracy']
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


def load_weights(path, model):
    model.load_weights(path)
    return model
