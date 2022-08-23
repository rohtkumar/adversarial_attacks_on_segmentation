import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import get_source_inputs

from attacks.robustifiers import pgd_l2_robust

import numpy as np
import time

seed=25
tf.random.set_seed(seed)
np.random.seed(seed)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

%%time
def robustify(robust_model, train_ds, iters=1000, alpha=0.1, batch_size=32):
    robust_train = []
    orig_labels = []
    example = False

    train_to_pull = list(iter(train_ds))
    start_rn = np.random.randint(0, len(train_ds))
    rand_batch = train_to_pull[start_rn][0]

    start_time = time.time()
    for i, (img_batch, label_batch) in enumerate(train_ds):
        inter_time = time.time()  

        # For the last batch, it is smaller than batch_size and thus we match the size for the batch of initial images
        if img_batch.shape[0] < batch_size:
            rand_batch = rand_batch[:img_batch.shape[0]]

        # Get the goal representation
        goal_representation = robust_model(img_batch)
      
        # Upate the batch of images
        learned_delta = pgd_l2_robust(robust_mod, rand_batch, goal_representation, alpha=alpha, num_iter=iters)
        robust_update = (rand_batch + learned_delta)

        # Add the updated images and labels to their respective lists
        robust_train.append(robust_update)
        orig_labels.append(label_batch)
      
        # Measure the time
        if (i+1) % 10 == 0:
            elapsed = time.time() - start_time
            elapsed_tracking = time.time() - inter_time
            logging.info(f'Robustified {(i+1)*batch_size} images in {elapsed:0.3f} seconds; Took {elapsed_tracking:0.3f} seconds for this particular iteration')    
      
      # Reset random image batch
        rn = np.random.randint(0, len(train_ds)-1) # -1 because last batch might be smaller
        rand_batch = train_to_pull[rn][0]

    return robust_train, orig_labels



# Convert to TensorFlow Dataset
robust_ds = tf.data.Dataset.from_tensor_slices((tf.concat(robust_train, axis=0), tf.concat(orig_labels, axis=0))).prefetch(AUTOTUNE).map(robust_preprocess, num_parallel_calls=AUTOTUNE).shuffle(len(robust_train)).batch(BATCH_SIZE)
     

