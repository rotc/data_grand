# -*- coding: utf-8 -*-
"""
@file:	 eval.py
@author: liaolin (liaolin@baidu.com)
@date:	 Fri 05 Oct 2018 11:05:06 AM CST
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import math
import time
import cPickle
import tensorflow as tf
import ConfigParser
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import slim

from two_layer_bigru_model import BiGRUModel
sys.path.append('../Utils')
from data_utils import load_data, BatchGenerator, assign_pretrained_word_embedding, do_eval

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('vocab_size', 500000, 'maximum vocab size')
tf.app.flags.DEFINE_integer('embedding_size', 400, 'sum embedding size of word2vec and glove')
tf.app.flags.DEFINE_integer('sequence_length', 2000, 'max num words of text')
tf.app.flags.DEFINE_integer('num_epoches', 200, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'batch size')
tf.app.flags.DEFINE_integer('num_classes', 19, 'num classes')
tf.app.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding or not')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('summaries_dir', './log/', 'summary dir')
tf.app.flags.DEFINE_string('ckpt_dir', './ckpt/', 'checkpoint dir')

model_config = ConfigParser.ConfigParser()
model_config.read('../Config/params.conf')
model_config = dict(model_config.items('BiGRU'))

with tf.variable_scope('Model', reuse=None):
    model = BiGRUModel(FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.sequence_length, \
            FLAGS.num_classes, is_training=False, model_config=model_config, \
            learning_rate=FLAGS.learning_rate)

saver = tf.train.Saver()

skf_pkl = '../Input/skf.10fold.pkl'
with open(skf_pkl, 'rb') as f:
    skf = cPickle.load(f)
train_sequence, train_labels, test_sequence = load_data(FLAGS.vocab_size)
train_idx, valid_idx = skf[0]
train_off_sequence, train_off_labels, valid_sequence, valid_labels = train_sequence[train_idx], \
        train_labels[train_idx], train_sequence[valid_idx], train_labels[valid_idx]
nvalid = len(valid_idx)
valid_sequence, valid_labels = valid_sequence[:nvalid], valid_labels[:nvalid]
va_batch_generator = BatchGenerator(FLAGS.sequence_length, FLAGS.batch_size, 0)

curr_ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
num_valid = 1000
num_va_batch = int(math.ceil(float(num_valid) / FLAGS.batch_size))
interval = 600
with tf.Session() as sess:
    sess.run(va_batch_generator.iterator.initializer, feed_dict={va_batch_generator.features: \
            valid_sequence, va_batch_generator.labels: valid_labels})

    wait_ckpt_list = []
    while True:
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
        ckpt_list = ckpt.all_model_checkpoint_paths
        new_ckpt_list = [ckpt for ckpt in ckpt_list if ckpt not in wait_ckpt_list]
        wait_ckpt_list.extend(new_ckpt_list)
        for ckpt_path in wait_ckpt_list:
            step = int(ckpt_path.rsplit('-', 1)[1])
            saver.restore(sess, ckpt_path)
            time0 = time.time()
            eval_score = do_eval(sess, model, va_batch_generator, num_valid, num_va_batch)
            time1 = time.time()
            tf.logging.info('step: {} f1_score: {} elapse time: {}'.format(step, eval_score, \
                time1 - time0))
            #print('step: {} f1_score: {} elapse time: {}'.format(step, eval_score, time1 - time0)
        wait_ckpt_list = []
        time.sleep(interval)
