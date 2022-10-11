from os import listdir

import tensorflow as tf
from tensorflow import keras as K
# from tensorflow.keras.experimental.preprocessing import Normalization
K.backend.set_image_data_format('channels_last')

import numpy as np
IMG_SIZE = 128

def define_img_size(img_size):
    global IMG_SIZE
    IMG_SIZE = img_size
    # print(IMG_SIZE)

def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.
        For one Image path:
        .../dataset/images/image1.jpg
        Its corresponding annotation path is:
        .../dataset/masks/masks1.png

    Returns
    -------
    dict
        Dictionary mapping an image and its mask.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    
    mask_path = tf.strings.regex_replace(img_path, "images", "masks")
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # Since 255 exist, changing it with 0
    mask = tf.where(mask == 255, np.dtype('uint8').type(8), mask)
    return {'image': image, 'segmentation_mask': mask}


def parse_cityscape_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.
        For one Image path:
        .../dataset/images/image1.jpg
        Its corresponding annotation path is:
        .../dataset/masks/masks1.png

    Returns
    -------
    dict
        Dictionary mapping an image and its mask.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "gtFine_color", "leftImg8bit")
    mask_path = tf.strings.regex_replace(mask_path, "gtFine", "leftImg8bit/leftImg8bit")
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # Since 255 exist, changing it with 0
    # mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    return {'image': image, 'segmentation_mask': mask}

def encode_segmap(mask):
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

    # these are 19
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    ignore_index = 250

    # dictionary of valid classes 7:0, 8:1, 11:2
    class_map = dict(zip(valid_classes, range(19)))

    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an mask of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its masks.
    """
    # transforms = Normalization(mean=[123.675, 116.28, 103.53],
    #                       variance=[np.square(58.395),
    #                                 np.square(57.12),
    #                                 np.square(57.375)])
    # mean = [123.675, 116.28, 103.53]
    # std = [58.395, 57.12, 57.375]
    # input_image = tf.cast(input_image, tf.float32) / 255.0
    # for channel in range(3):
    #     input_image[:, :, channel] = (input_image[:, :, channel] - mean[channel]) / std[channel]

    input_image = tf.cast(input_image, tf.float32) / 255.0
    # input_image = transforms(input_image)
    # arry = np.array(input_mask, dtype=np.uint8)
    # input_mask = encode_segmap(arry)
    # input_mask = tf.convert_to_tensor(input_mask, dtype=tf.int64)
#     input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask
   

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its mask.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the mask also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its mask.

    Returns
    -------
    tuple
        A modified image and its mask.
    """

    # IMG_SIZE=128
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    # print(tf.__version__)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    # print(input_image)
    return input_image, input_mask   

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its mask.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the mask also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its mask.

    Returns
    -------
    tuple
        A modified image and its mask.
    """
#     print(datapoint['image'])
#     IMG_SIZE = 128
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask   

def robust_preprocess(img, label):
    """Defines preprocessing / data augmentation for robust & nonrobust features"""
    img = tf.image.resize_with_pad(img, 32+4, 32+4)
    img = tf.image.random_crop(img, size=[32, 32, 3])
    img = tf.image.stateless_random_flip_left_right(img, (15, 13))
    return img, label

def get_image_paths(dir):
    return sorted([dir + path for path in listdir(dir)])