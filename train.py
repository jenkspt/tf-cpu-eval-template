
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True, reshape=False)

import tensorflow as tf
import os

from model import NAME, inference, loss
from helper import print_block

LOG_DIR = './log_dir'

######### PARAMETERS ################
learning_rate = 0.001
training_epochs = 20
batch_size = 128    # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784       # MNIST data input (img shape: 28*28)
n_classes = 10      # MNIST total classes (0-9 digits)
resume = True
run = 1
train_name = '{}-run-{}'.format(NAME, run)

######### LOG PATHS #################
CHECKPOINTS = os.path.join(LOG_DIR, train_name, 'checkpoints')
TRAIN_LOGS = os.path.join(LOG_DIR, train_name, 'train')


######### THE TRAINING GRAPH ######## 
global_step = tf.Variable(0, name='global_step', trainable=False)

x = tf.placeholder(tf.float32, [None, 28, 28, 1], 'inputs')
y = tf.placeholder(tf.float32, [None, n_classes], 'targets')

logits = inference(x)

#-------- OPTMIZATION --------------# 
cost_node = loss(logits, y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(cost_node, global_step=global_step)


summary_node = tf.summary.merge_all()   # Get all tensorboard variable summaries
saver = tf.train.Saver(max_to_keep=1)
#####################################

ckpt_prefix = os.path.join(CHECKPOINTS, 'model')
print_string = 'Global step: {}, Loss: {}'

with tf.Session() as sess:
    ## INITIALIZE THE SESSION 
    if resume:
        print_block('Restoring training session from checkpoint')
        last_ckpt = tf.train.latest_checkpoint(CHECKPOINTS)
        saver.restore(sess, last_ckpt)
    else:
        os.makedirs(CHECKPOINTS)
        os.mkdir(TRAIN_LOGS)
        print_block("Initializing new training session")
        sess.run(tf.global_variables_initializer())
    
    # Tensorboard writer
    summary_writer = tf.summary.FileWriter(TRAIN_LOGS, graph=sess.graph)
    
    step = sess.run(global_step)        # Get the global step
    ## TRAINING
    try:                                # Catch early stopping
        for epoch in range(training_epochs):
            total_batch = mnist.train.num_examples//batch_size
            # Loop over all batches
            for i in range(total_batch):

                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                feed_dict = {x: batch_x, y: batch_y}
                fetch = [train_op, summary_node, cost_node, global_step]
                _, summary, cost, step = sess.run(fetch, feed_dict=feed_dict)

                if i%100 == 0:
                    # Write summary to tensorboard
                    summary_writer.add_summary(summary, step)
                    print(print_string.format(step, cost))

            saver.save(sess, ckpt_prefix, global_step=step)

    # SAVE CHECKPOINT ON TERMINATION
    except KeyboardInterrupt:
        print_block('Training terminated ... saving checkpoint at step {}'.format(step))
        ckpt_name = os.path.join(CHECKPOINTS, 'model')
        saver.save(sess, ckpt_prefix, global_step=step)

