import tensorflow as tf
from tensorflow import keras as K
K.backend.set_image_data_format('channels_last')
from tensorflow.keras.utils import get_source_inputs

import segmentation_models as sm
SM_FRAMEWORK=tf.keras
sm.set_framework('tf.keras')
from segmentation_models import get_preprocessing
seed=25

import numpy as np

tf.random.set_seed(seed)
np.random.seed(seed)



def get_model(backbone, classes, activation):
    # BACKBONE = 'efficientnetb3' 'resnet50'
    preprocess_input = get_preprocessing(backbone)
    model = sm.Unet(BACKBONE ,input_shape=(128, 128, 3),classes=classes,activation=activation, encoder_weights='imagenet')
    return model

def load_weights(path, model):
    model.load_weights(path)
    return model