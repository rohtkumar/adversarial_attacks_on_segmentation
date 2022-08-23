import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import get_source_inputs

import numpy as np
import time

SM_FRAMEWORK=tf.keras
import segmentation_models as sm

sm.set_framework('tf.keras')
K.backend.set_image_data_format('channels_last')
from segmentation_models import get_preprocessing
seed=25
tf.random.set_seed(seed)
np.random.seed(seed)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

def adversarial_training(model, train_ds, test_ds, train_attack=pgd_l2_adv, test_attack=None, epochs=5, verbose=True, test_kwargs=None, **kwargs):
    """Runs the "adversarial" training loop described in Ilyas et al. (2019)
    
    Adversarial training allows for two separate attacks, one during training and a separate one during
    test. Note that if 'train_attack' is set to None, then this becomes standard training.
    
    Args:
        - model (TFModel): A compiled TF / Keras model
        - train_ds (TFData): a tensorflow data object for the training data
        - test_ds (TFData): a tensorflow data object for the test data 
        - attack (function): an attack function (i.e. PGD L2) to perturb
                test data if evaluating adversarial performance. Default : None
        - epochs (int): number of epochs to run training. Default: 5.
        - verbose (bool): Report results after each epoch. Otherwise
                will return the train / test accuracies at the end of training. 
                Default: True"""

    # Create train and test functions wrapped
    if train_attack is not None:
        train_attack_tf = tf.function(train_attack)
    if test_attack is not None:
        test_attack_tf = tf.function(test_attack)

    for n in range(epochs):
#         print(n)
        t = time.time()
        train_losses = []
        train_accs = []
        for b in train_ds:
            X, y = b
#             print("here")
            # Create adversarially perturbed training data
            if train_attack is not None:
                delta = train_attack_tf(model, X, y, **kwargs)
                Xd = X + delta
            else:
                Xd = X  
            # Train model on adversarially perturbed data
            l, l1, acc, other_returned_val = model.train_on_batch(Xd, y)
#             print(acc)
            train_losses.append(l)
            train_accs.append(acc)

        test_losses = []
        test_accs = []
        for vb in test_ds:
            Xtest, ytest = vb
            
            # When attack is specified (ie not None), apply
            # attack at test time; do not apply in training due to
            # 'standard_training' definition
            if test_attack is not None:
                if isinstance(test_kwargs, dict):
                    delta = test_attack_tf(model, Xtest, ytest, **test_kwargs)
                else:
                    delta = test_attack_tf(model, Xtest, ytest)
                Xdtest = Xtest + delta
            else:
                # when test_attack is not specified 
                Xdtest = Xtest
            
            l, l1, acc, other_returned_val = model.test_on_batch(Xdtest, ytest)
#             print({l}, {acc}, {other_returned_val})
            test_losses.append(l)
            test_accs.append(acc)
        
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)
        
        test_loss = sum(test_losses) / len(test_losses)
        test_acc = sum(test_accs) / len(test_accs)
        
        if verbose:
            logging.info(f"Epoch {n}/{epochs}, Time: {(time.time()-t):0.2f} -- Train Loss: {train_loss:0.2f}, \
                Train Acc: {train_acc:0.2f}, Test Loss: {test_loss:0.2f}, Test Acc: {test_acc:0.2f}")
    
    # Return final train and test losses
    if verbose == False:
        return {'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc}