import tensorflow as tf

from helper import weight_variable, bias_variable, lRelu

NAME = '1Dense'

def inference(x):
    input_dim = x.get_shape()[1].value * x.get_shape()[2].value
    x_flat = tf.reshape(x, [-1, input_dim], name='flatten')

    with tf.variable_scope('Dense1') as scope:
        weights = weight_variable([input_dim, 392])
        matmul = tf.matmul(x_flat, weights)
        norm = tf.contrib.layers.batch_norm(matmul)
        dense1 = lRelu(norm, name='activation')

    with tf.variable_scope('Dense2') as scope:
        weights = weight_variable([392, 10])
        matmul = tf.matmul(dense1, weights)
        bias = bias_variable([10])
        logits = tf.add(matmul, bias, name='logits')

    return logits


def loss(logits, y):
    
    with tf.variable_scope('Loss'):
        cost = tf.reduce_mean(\
               tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

        tf.summary.scalar('cross_entropy', cost)
    return cost
