
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True, reshape=False)

import tensorflow as tf
import os

from model import NAME, inference, loss

LOG_DIR = './log_dir'


# Parameters
batch_size = 128    # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784       # MNIST data input (img shape: 28*28)
n_classes = 10      # MNIST total classes (0-9 digits)
run = 1
train_name = '{}-run-{}'.format(NAME, run)

####### log paths ###################
CHECKPOINTS = os.path.join(LOG_DIR, train_name, 'checkpoints')
TEST_LOGS = os.path.join(LOG_DIR, train_name, 'test')

if not os.path.isdir(TEST_LOGS):
    os.mkdir(TEST_LOGS)

####### THE EVALUATION GRAPH ########
global_step = tf.Variable(0, name='global_step', trainable=False)

x = tf.placeholder(tf.float32, [None, 28, 28, 1], 'inputs')
y = tf.placeholder(tf.float32, [None, n_classes], 'targets')

logits = inference(x)

cost_node = loss(logits, y)


summary_node = tf.summary.merge_all()   # Get all tensorboard variable summaries
saver = tf.train.Saver()

# Run Evaluation on the CPU only
config = tf.ConfigProto(
        device_count = {'CPU': 0}
    )
with tf.Session() as sess:
    ## INITIALIZE THE SESSION 
    ckpt_prefix = tf.train.latest_checkpoint(CHECKPOINTS)
    # Load the meta graph, and clear GPU device
    saver = tf.train.import_meta_graph(ckpt_prefix + '.meta', clear_devices=True)
    saver.restore(sess, ckpt_prefix)

    # Tensorboard writer
    summary_writer = tf.summary.FileWriter(TEST_LOGS, graph=sess.graph)

    ## Evaluation 
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    feed_dict = {x: batch_x, y: batch_y}
    fetch = [global_step, summary_node, cost_node]
    step, summary, cost = sess.run(fetch, feed_dict=feed_dict)
    print('Global step: {}, Loss: {}'.format(step, cost))
