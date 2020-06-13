import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], './depthwise')
    sess.run(tf.global_variables_initializer())

    # transform_img = (np.ones((300,300,3))*127).astype(np.float32).reshape(1,300,300,3)
    transform_img = np.load('input.npy').reshape(1,300,300,3)


    input_x = sess.graph.get_tensor_by_name('data:0')

    conv17_2_mbox_conf = sess.graph.get_tensor_by_name('flatten_13/Reshape:0')
    conv17_2_mbox_loc = sess.graph.get_tensor_by_name('flatten_12/Reshape:0')

    result = sess.run([conv17_2_mbox_loc,conv17_2_mbox_conf],feed_dict={input_x:transform_img})

    print(result[0].shape)
    print(result[0].reshape(-1)[0:100])
    print(result[1].shape)
    print(result[1].reshape(-1)[0:100])
