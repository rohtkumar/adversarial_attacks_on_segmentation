import tensorflow as tf
from util import parsing as parse
import logging
import cityscapesscripts as cs
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
# tf.data.experimental.AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_dataset(dataset_path, img_size, batch_size, buffer_size):
    logger = logging.getLogger(__name__)
    logger.info('making data set from image directory')
    
    all_dataset = tf.data.Dataset.list_files(dataset_path + "*.png")
    test_dataset = all_dataset.take(288) 
    train_dataset = all_dataset.skip(290)
    parse.define_img_size(img_size)
    TRAINSET_SIZE = len(train_dataset)
    VALSET_SIZE= len(test_dataset)
    
    logger.info(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    logger.info(f"The Training Dataset contains {VALSET_SIZE} images.")
    
    train_dataset = train_dataset.map(parse.parse_image)
    test_dataset = test_dataset.map(parse.parse_image)

    dataset = {"train": train_dataset, "val": test_dataset}

    dataset['train'] = dataset['train'].map(parse.load_image_train)
    dataset['train'] = dataset['train'].shuffle(buffer_size=buffer_size, seed=42)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(batch_size)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)


    dataset['val'] = dataset['val'].map(parse.load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(batch_size)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)
    # result = dataset['val'].apply(tf.data.experimental.assert_cardinality(288))
    # print(len(result))
    return dataset['train'].apply(tf.data.experimental.assert_cardinality(1152)), dataset['val'].apply(tf.data.experimental.assert_cardinality(288))


def get_cityscape_dataset(dataset_path, img_size, batch_size, buffer_size):

    logger = logging.getLogger(__name__)
    logger.info('making data set from image directory')
    parse.define_img_size(img_size)
    train_dataset = tf.data.Dataset.from_tensor_slices(parse.get_image_paths(dataset_path+'train/'))
    test_dataset = tf.data.Dataset.from_tensor_slices(parse.get_image_paths(dataset_path+'test/'))
    val_dataset = tf.data.Dataset.from_tensor_slices(parse.get_image_paths(dataset_path+'val/'))

    train_dataset = train_dataset.map(parse.parse_cityscape_image)
    test_dataset = test_dataset.map(parse.parse_cityscape_image)
    val_dataset = val_dataset.map(parse.parse_cityscape_image)

    TRAINSET_SIZE = len(train_dataset)
    VALSET_SIZE = len(val_dataset)
    TESTSET_SIZE = len(test_dataset)

    logger.info(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    logger.info(f"The Val Dataset contains {VALSET_SIZE} images.")
    logger.info(f"The Test Dataset contains {TESTSET_SIZE} images.")

    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    dataset['train'] = dataset['train'].map(parse.load_image_train)
    dataset['train'] = dataset['train'].shuffle(buffer_size=buffer_size, seed=42)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(batch_size)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    dataset['val'] = dataset['val'].map(parse.load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(batch_size)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    dataset['test'] = dataset['test'].map(parse.load_image_test)
    dataset['test'] = dataset['test'].repeat()
    dataset['test'] = dataset['test'].batch(batch_size)
    dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)

    return dataset['train'].apply(tf.data.experimental.assert_cardinality(2975)), dataset['val'].apply(tf.data.experimental.assert_cardinality(500)), dataset['test'].apply(tf.data.experimental.assert_cardinality(1525))


def get_robust_dataset(robust_train, orig_labels, batch_size):
    img = tf.concat(robust_train, axis=0)
    mask = tf.concat(orig_labels, axis=0)
    robust_ds = tf.data.Dataset.from_tensor_slices((img, mask))
    robust_ds = robust_ds.prefetch(AUTOTUNE)
    robust_ds = robust_ds.map(parse.robust_preprocess, num_parallel_calls=AUTOTUNE)
    robust_ds = robust_ds.shuffle(len(robust_train))
    robust_ds = robust_ds.batch(batch_size)
    
    return robust_dsl
   