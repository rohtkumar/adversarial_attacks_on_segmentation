import numpy as np # for using np arrays

# for bulding and running deep learning model
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import get_source_inputs

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import time




def norm(Z):
    """Compute norms over all but the first dimension"""
    return tf.norm(tf.reshape(Z, (Z.shape[0], -1)), axis=1)

# PGD L2 for Adversarial Examples #
"""PGD.

    Parameters
    ----------
    Model : [type], optional
        [Input dataset, by default None
    X,y : int, optional
        Number of sample to show, by default 1
    alpha : 
    epsilon:
    delta : 
    """
def single_pgd_step_adv(model, X, y, alpha, epsilon, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE # Use no aggregation - will give gradient separtely for each ex.
            )(y, model(X + delta)) # comparing to label for original data point
    grad = tape.gradient(loss, delta)

    normgrad = tf.reshape(norm(grad), (-1, 1, 1, 1))
    z = delta + alpha * (grad / (normgrad + 1e-10))

    normz = tf.reshape(norm(z), (-1, 1, 1, 1))
    delta = epsilon * z / (tf.math.maximum(normz, epsilon) + 1e-10)
    return delta, loss

def pgd_l2_adv(model, X, y, alpha, num_iter, epsilon=0, example=False):
    fn = tf.function(single_pgd_step_adv)
    delta = tf.zeros_like(X)
    loss = 0
    for t in range(num_iter):
        delta, loss = fn(model, X, y, alpha, epsilon, delta)
        
    if example:
        print(f'{num_iter} iterations, final MSE {loss}')
    return delta

# Helper for pgd_linf
@tf.function
def onestep_pgd_linf(model, X, y, epsilon, alpha, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )(y, model(X + delta))

    delta = tf.clip_by_value(delta + alpha*tf.sign(tape.gradient(loss, delta)), X-epsilon, X+epsilon)
    
    return delta


def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = tf.zeros_like(X)
    for t in range(num_iter):
        delta = onestep_pgd_linf(model, X, y, epsilon, alpha, delta)
    return delta


@tf.function
def fgsm_attack(model, X, y, epsilon=0.1):
    delta = tf.zeros_like(X)
    
    # use GradientTape to perform autodiff
    with tf.GradientTape() as tape:
        # specifically 'watch' delta
        # see here: https://www.tensorflow.org/guide/autodiff
        tape.watch(delta)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )(y, model(X + delta))
    delta = tf.sign(tape.gradient(loss, delta)) * epsilon
    return delta


def run_adversarial_attack(model, valid_dataset, attack, attack_params=None, **kwargs):
    """Runs only the adversarial attack on a trained model.

    Differs from standard training by skipping the gradient updates which can be highly costly. Can be
    demonstrated from training model in standard way and then applying"""
    # Convert function to tf function
    attack_fn = tf.function(attack)

    t = time.time()
    test_losses = []
    test_accs = []
    #     print(valid_dataset.length())
    for Xtest, ytest in valid_dataset:
        # Run attack perturbation
        if attack_params is not None:
            delta = attack_fn(model, Xtest, ytest, **attack_params)
        else:
            delta = attack_fn(model, Xtest, ytest, **kwargs)

        #         print(delta)
        Xdtest = Xtest + delta
        l, acc, *is_anything_else_being_returned = model.test_on_batch(Xdtest, ytest)
        print(l)
        clear_output(wait=True)
        test_losses.append(l)
        test_accs.append(acc)

    test_loss = sum(test_losses) / len(test_losses)
    test_acc = sum(test_accs) / len(test_accs)
    print(f"Time: {(time.time() - t):0.2f} Test Loss: {test_loss:0.2f}, Test Acc: {test_acc:0.2f}")