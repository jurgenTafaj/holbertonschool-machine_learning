#!/usr/bin/env python3
"""
A function that evaluates the output of a neural network
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    A function that evaluates the outout of a neural network
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        feed_dict = {x: X, y: Y}
        eval_accuracy, eval_loss, eval_y_pred = sess.run(
            [accuracy, loss, y_pred], feed_dict)

    return eval_y_pred, eval_accuracy, eval_loss
