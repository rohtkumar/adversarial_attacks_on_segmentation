from os import listdir

import tensorflow as tf


def get_dataset():
    COLAB_DIR = '/home/rohkumar/code/citysacpe/'
    GT_DIR = COLAB_DIR + 'gtFine/'
    IMG_DIR = COLAB_DIR + 'leftImg8bit/leftImg8bit/'

    IMG_SIZE = 128
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    GT_DIR = COLAB_DIR + 'gtFine/'
    IMG_DIR = COLAB_DIR + 'leftImg8bit/leftImg8bit/'

    def load_and_preprocess_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img /= 255.0
        return img

    def get_image_paths(dir):
        return sorted([dir + path for path in listdir(dir)])

    # create tf.Dataset objects
    gt_train_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(GT_DIR + 'train/'))
    gt_val_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(GT_DIR + 'val/'))
    gt_test_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(GT_DIR + 'test/'))

    gt_train_ds = gt_train_ds.map(load_and_preprocess_image)
    gt_val_ds = gt_val_ds.map(load_and_preprocess_image)
    gt_test_ds = gt_test_ds.map(load_and_preprocess_image)

    im_train_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(IMG_DIR + 'train/'))
    im_val_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(IMG_DIR + 'val/'))
    im_test_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(IMG_DIR + 'test/'))

    im_train_ds = im_train_ds.map(load_and_preprocess_image)
    im_val_ds = im_val_ds.map(load_and_preprocess_image)
    im_test_ds = im_test_ds.map(load_and_preprocess_image)