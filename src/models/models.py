import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import get_source_inputs
import segmentation_models as sm
from segmentation_models.base import Metric, functional as F
from segmentation_models import get_preprocessing
import logging
from util import logger, tools
import numpy as np
K.backend.set_image_data_format('channels_last')
SM_FRAMEWORK = tf.keras
sm.set_framework('tf.keras')


class MyIOUScore(Metric):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:
    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}
    Args:
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
    Returns:
       A callable ``iou_score`` instance. Can be used in ``model.compile(...)`` function.
    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index
    Example:
    .. code:: python
        metric = IOUScore()
        model.compile('SGD', loss=loss, metrics=[metric])
    """

    def __init__(
            self,
            class_weights=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            class_indexes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            threshold=None,
            per_image=False,
            smooth=1e-5,
            name=None,
    ):
        name = name or 'iou_score'
        super().__init__(name=name)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        pr=tf.argmax(pr, axis=-1)
        pr = tf.expand_dims(pr, axis=-1)
        pr = tf.cast(pr, tf.float32)
#        gt = tf.cast(gt, tf.int64)
        #import pdb;pdb.set_trace()
        return F.iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
            **self.submodules
        )


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
        metrics=[MyIOUScore(), 'accuracy'],
        #run_eagerly=True
        #metrics=[tf.keras.metrics.IoU(num_classes=10,target_class_ids=[0,1,2,3,4,5,6,7,8,9]), 'accuracy']
    )

    return model

def initialize_std_model_test(args, classes, activation):
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [15000, 20000], [0.1, 0.01, 1e-3])

    model = get_model(args, classes, activation)    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer= K.optimizers.Adam(0.0001),
        # optimizer = tf.keras.optimizers.Adam(),
        metrics=[MyIOUScore(), 'accuracy']
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
        metrics=[MyIOUScore(), 'accuracy']
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
    representation = args.adv_model.layers[-4]
    robustifier = tf.keras.Model(inputs=args.adv_model.layers[0].input,
                                 outputs=representation.output)
    return robustifier


def load_weights(path, model):
    model.load_weights(path)
    return model
