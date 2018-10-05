# -*- coding: utf-8 -*-
"""
@file:	 main.py
@author: liaolin (liaolin@baidu.com)
@date:	 Sat 29 Sep 2018 06:35:10 PM CST
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import math
import cPickle
import tensorflow as tf
import ConfigParser

from two_layer_bigru_model import BiGRUModel
sys.path.append('../Utils')
from data_utils import load_data, BatchGenerator, assign_pretrained_word_embedding, do_eval

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('vocab_size', 500000, 'maximum vocab size')
tf.app.flags.DEFINE_integer('embedding_size', 400, 'sum embedding size of word2vec and glove')
tf.app.flags.DEFINE_integer('sequence_length', 2000, 'max num words of text')
tf.app.flags.DEFINE_integer('num_epoches', 200, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('num_classes', 19, 'num classes')
tf.app.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding or not')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('summaries_dir', './log/', 'summary dir')
tf.app.flags.DEFINE_string('ckpt_dir', './ckpt/', 'checkpoint dir')

model_config = ConfigParser.ConfigParser()
model_config.read('../Config/params.conf')
model_config = dict(model_config.items('BiGRU'))

skf_pkl = '../Input/skf.10fold.pkl'
with open(skf_pkl, 'rb') as f:
    skf = cPickle.load(f)

train_sequence, train_labels, test_sequence = load_data(FLAGS.vocab_size)
train_idx, valid_idx = skf[0]
train_off_sequence, train_off_labels, valid_sequence, valid_labels = train_sequence[train_idx], \
        train_labels[train_idx], train_sequence[valid_idx], train_labels[valid_idx]
num_train, num_valid = train_off_sequence.shape[0], valid_sequence.shape[0]
print('num_train: {} num_valid: {}'.format(num_train, num_valid))
tr_batch_generator = BatchGenerator(FLAGS.sequence_length, FLAGS.batch_size, buffer_size=num_train)

with tf.Session() as sess:
    with tf.variable_scope('Model', reuse=None):
        model = BiGRUModel(FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.sequence_length, 
            FLAGS.num_classes, is_training=True, model_config=model_config, 
            learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver()
    sess.run(tr_batch_generator.iterator.initializer, feed_dict={tr_batch_generator.features: \
            train_off_sequence, tr_batch_generator.labels: train_off_labels})
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    print('Initializing Variables')
    sess.run(tf.global_variables_initializer())
    if FLAGS.use_embedding:
        assign_pretrained_word_embedding(sess, model)

    num_tr_batch = int(math.ceil(float(num_train) / FLAGS.batch_size))
    print('num_tr_batch: {}'.format(num_tr_batch))
    for epoch in xrange(FLAGS.num_epoches):
        loss, acc = 0.0, 0.0
        for counter in xrange(1, num_tr_batch + 1):
            x_train, y_train = sess.run(tr_batch_generator.batch)
            feed_dict = {model.input_x: x_train, model.input_y: y_train, model.dropout_keep_prob: 0.8}
            curr_loss, curr_acc, summary, _ = sess.run([model.loss_val, model.accuracy, \
                    model.merged, model.train_op], feed_dict)
            loss, acc = loss + curr_loss, acc + curr_acc

            if counter % 20 == 1:
                print('Epoch: {0}\tBatch: {1}\tTrain loss: {2}\tTrain accuracy: {3}'.format(\
                        epoch, counter, loss/float(counter), acc/float(counter)))

                save_path = FLAGS.ckpt_dir + 'model.ckpt'
                saver.save(sess, save_path, global_step=counter)

            train_writer.add_summary(summary, epoch * num_tr_batch + counter)
