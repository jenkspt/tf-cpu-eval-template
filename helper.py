
import tensorflow as tf

def print_block(string):
    print("\n\n\n{:#^50s}".format(''))
    print("\n{}".format(string))
    print("\n{:#^50s}".format(''))

def weight_variable(shape, name='weights'):
	initial = tf.truncated_normal(shape, stddev=0.02)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name='biases'):
	initial = tf.constant(.0001, shape=shape)
	return tf.Variable(initial, name=name)
    
def lRelu(x, name='lRelu'):
    return tf.maximum(x, 0.2*x, name=name)
