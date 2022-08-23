import tensorflow as tf
from tensorflow import keras as K
K.backend.set_image_data_format('channels_last')

tf.random.set_seed(25)

def single_pgd_step_robust(model, X, y, alpha, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )(y, model(X + delta)) # comparing to robust model representation layer
        
    grad = tape.gradient(loss, delta)
    normgrad = tf.reshape(norm(grad), (-1, 1, 1, 1))
    delta -= alpha*grad / (normgrad + 1e-10) # normalized gradient step
    delta = tf.math.minimum(tf.math.maximum(delta, -X), 1-X) # clip X+delta to [0,1]
    return delta, loss

def pgd_l2_robust(model, X, y, alpha, num_iter, epsilon=0, example=False):
    delta = tf.zeros_like(X)
    loss = 0
    fn = tf.function(single_pgd_step_robust)
    for t in range(num_iter):
        delta, loss = fn(model, X, y, alpha, delta)
    # Prints out loss to evaluate if it's actually learning (currently broken)
    if example:
        print(f'{num_iter} iterations, final MSE {loss}')
    return delta


def single_pgd_step_nonrobust(model, X, y, alpha, epsilon, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE # Use no aggregation - will give gradient separtely for each ex.
            )(y, model(X + delta)) # comparing to label for original data point
    grad = tape.gradient(loss, delta) #tape.gradient(loss, delta)

    # equivalent to delta += alpha*grad / norm(grad), just for batching
    normgrad = tf.reshape(norm(grad), (-1, 1, 1, 1))
    # changed from plus to minus b/c trying to minimize with non-robust
    z = delta - alpha * (grad / (normgrad + 1e-10))
    normz = tf.reshape(norm(z), (-1, 1, 1, 1))
    delta = epsilon * z / (tf.math.maximum(normz, epsilon) + 1e-10)
    return delta, loss

def pgd_l2_nonrobust(model, X, y, alpha, num_iter, epsilon=0, example=False):
    fn = tf.function(single_pgd_step_nonrobust)
    delta = tf.zeros_like(X)
    loss = 0
    for t in range(num_iter):
        delta, loss = fn(model, X, y, alpha, epsilon, delta)
        
    if example:
        print(f'{num_iter} iterations, final MSE {loss}')
    return delta   