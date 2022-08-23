import tensorflow as tf
from util import parsing as parse
import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

AUTOTUNE = tf.data.AUTOTUNE
seed=25
tf.random.set_seed(seed)
np.random.seed(seed)

def get_dataset(dataset_path, img_size, batch_size, buffer_size):
    logger = logging.getLogger(__name__)
    logger.info('making data set from image directory')
    
    all_dataset = tf.data.Dataset.list_files(dataset_path + "*.png", seed=seed)
    test_dataset = all_dataset.take(288) 
    train_dataset = all_dataset.skip(290)

    TRAINSET_SIZE = len(train_dataset)
    VALSET_SIZE= len(test_dataset)
    
    logger.info(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    logger.info(f"The Training Dataset contains {VALSET_SIZE} images.")
    
    train_dataset = train_dataset.map(parse_image)
    test_dataset = test_dataset.map(parse_image)

    dataset = {"train": train_dataset, "val": test_dataset}

    dataset['train'] = dataset['train'].map(parse.load_image_train)
    dataset['train'] = dataset['train'].shuffle(buffer_size=buffer_size, seed=seed)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(batch_size)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)


    dataset['val'] = dataset['val'].map(parse.load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(batch_size)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)
    
    return dataset['train'], dataset['val']

def get_robust_dataset(robust_train, orig_labels, batch_size):
    img = tf.concat(robust_train, axis=0)
    mask = tf.concat(orig_labels, axis=0)
    robust_ds = tf.data.Dataset.from_tensor_slices((img, mask))
    robust_ds = robust_ds.prefetch(AUTOTUNE)
    robust_ds = robust_ds.map(parse.robust_preprocess, num_parallel_calls=AUTOTUNE)
    robust_ds = robust_ds.shuffle(len(robust_train))
    robust_ds = robust_ds.batch(batch_size)
    
    return robust_dsl
   