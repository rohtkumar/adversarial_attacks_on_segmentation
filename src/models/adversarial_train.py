import os

import tensorflow as tf
from tensorflow import keras as K
from util import logger, tools
import logging
import time
SM_FRAMEWORK=tf.keras
import segmentation_models as sm
sm.set_framework('tf.keras')
K.backend.set_image_data_format('channels_last')

def adversarial_training(args, train_ds, test_ds, train_attack, test_attack=None, epochs=5, verbose=True, test_kwargs=None, **kwargs):
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

    with logger.LoggingBlock("Start Adversarial Training ", emph=True):
        model = args.adv_model
        saved_model = os.path.join(args.save, 'adv_' + args.model_name + '_unet.h5')
        train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(792))
        batch_train = len(train_ds) // args.batch_size
        batch_test = len(test_ds) // args.batch_size
        progbar_train = tf.keras.utils.Progbar(batch_train)
        progbar_test = tf.keras.utils.Progbar(batch_test)
        patience = args.early_stopping_patience
        wait = 0
        best = 0
        # Create train and test functions wrapped
        if train_attack is not None:
            train_attack_tf = tf.function(train_attack)
        if test_attack is not None:
            test_attack_tf = tf.function(test_attack)

        metrics_names = ['loss', 'IOU', 'acc']

        for n in range(epochs):
            t = time.time()
            train_losses = []
            train_accs = []

            for i, b in enumerate(train_ds.take(batch_train)):
                X, y = b

                # Create adversarially perturbed training data
                if train_attack is not None:
                    delta = train_attack_tf(model, X, y, **kwargs)
                    Xd = X + delta
                else:
                    Xd = X
                # Train model on adversarially perturbed data
                loss, iou, acc = model.train_on_batch(Xd, y)
                train_losses.append(loss)
                train_accs.append(acc)
                values = [('loss', loss),('IOU', iou), ('acc', acc) ]
                progbar_train.update(i, values=values)

            test_losses = []
            test_accs = []
            for i, vb in enumerate(test_ds.take(batch_test)):
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

                loss, iou, acc = model.test_on_batch(Xdtest, ytest)
    #             print({l}, {acc}, {other_returned_val})
                test_losses.append(loss)
                test_accs.append(acc)
                values = [('loss', loss), ('IOU', iou), ('acc', acc)]
                progbar_test.update(i, values=values)

            train_loss = sum(train_losses) / len(train_losses)
            train_acc = sum(train_accs) / len(train_accs)

            test_loss = sum(test_losses) / len(test_losses)
            test_acc = sum(test_accs) / len(test_accs)

            wait += 1
            if test_loss > best:
                best = test_loss
                wait = 0
            if wait >= patience:
                break

            if verbose:
                logging.info(f"Epoch {n}/{epochs}, Time: {(time.time()-t):0.2f} -- Train Loss: {train_loss:0.2f}, \
                    Train Acc: {train_acc:0.2f}, Test Loss: {test_loss:0.2f}, Test Acc: {test_acc:0.2f}")
        logging.info("Adversarial Training completed..... Saving model.")
        tools.save_model(model, saved_model)
