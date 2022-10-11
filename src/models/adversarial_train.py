import tensorflow as tf
from tensorflow import keras as K
from util import logger
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
        train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(1153))
        # iterator = train_ds.make_initializable_iterator()
        # tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
        progbar_train = tf.keras.utils.Progbar(len(train_ds))
        progbar_test = tf.keras.utils.Progbar(len(test_ds))
        # Create train and test functions wrapped
        if train_attack is not None:
            print(train_attack)
            train_attack_tf = tf.function(train_attack)
        if test_attack is not None:
            test_attack_tf = tf.function(test_attack)

        metrics_names = ['acc', 'IOU']

        for n in range(epochs):
    #         print(n)
            t = time.time()
            train_losses = []
            train_accs = []

            for i, b in enumerate(train_ds//args.batch_size):
                X, y = b
    #             print("here")
                # Create adversarially perturbed training data
                if train_attack is not None:
                    delta = train_attack_tf(model, X, y, **kwargs)
                    Xd = X + delta
                else:
                    Xd = X
                # Train model on adversarially perturbed data
                l, l1, other_returned_val = model.train_on_batch(Xd, y)
    #             print(acc)
    #             tf.print(l1)
                train_losses.append(l)
                train_accs.append(l1)
                values = [('acc', l1), ('IOU', l)]
                progbar_train.update(i, values=values)

            test_losses = []
            test_accs = []
            for i, vb in enumerate(test_ds//args.batch_size):
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

                l, l1, other_returned_val = model.test_on_batch(Xdtest, ytest)
    #             print({l}, {acc}, {other_returned_val})
                test_losses.append(l)
                test_accs.append(l1)
                progbar_test.update(i)

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