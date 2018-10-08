# -*- coding: utf-8 -*-
"""
@file:	 two_layer_bigru_model.py
@author: liaolin (liaolin@baidu.com)
@date:	 Sun 23 Sep 2018 12:17:57 PM CST
"""
import os

import tensorflow as tf
from tensorflow.contrib import rnn, cudnn_rnn

class BiGRUModel:
    def __init__(self, vocab_size, embedding_size, sequence_length, \
            num_classes, is_training, model_config, learning_rate=0.001, \
            initializer=tf.random_normal_initializer(stddev=0.1)):
        self.vocab_size = vocab_size + 1
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.is_training = is_training
        self.rnn_state_size = int(model_config['rnn_state_size'])
        self.fc_hidden_size1 = int(model_config['fc_hidden_size1'])
        self.fc_hidden_size2 = int(model_config['fc_hidden_size2'])
        self.learning_rate = learning_rate
        self.initializer = initializer

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None,], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.embedding = tf.get_variable('Embedding', shape=[self.vocab_size, \
                self.embedding_size], trainable=False, initializer=self.initializer)

        self.logits = self.inference()
        self.preds_proba = tf.nn.sigmoid(self.logits)
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')

        if not self.is_training:
            return

        self.loss_val = self.loss()
        self.train_op = self.optimize()

        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def inference(self):
        x = tf.nn.embedding_lookup(self.embedding, self.input_x)
        print('embedded words shape:', x.get_shape().as_list())

        if self.is_training:
            x = tf.nn.dropout(x, noise_shape=[tf.shape(x)[0], 1, self.embedding_size], \
                    keep_prob=self.dropout_keep_prob)
            print('dropout shape:', x.get_shape().as_list())

        fw_cell1 = cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.rnn_state_size)
        bw_cell1 = cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.rnn_state_size)
        x, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell1, bw_cell1, x, \
                dtype=tf.float32, scope='birnn1')
        x = tf.concat(x, axis=2)
        print('bigru1 output shape:', x.get_shape().as_list())

        fw_cell2 = cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.rnn_state_size)
        bw_cell2 = cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.rnn_state_size)
        x, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell2, bw_cell2, x, \
                dtype=tf.float32, scope='birnn2')
        x = tf.concat(x, axis=2)
        print('bigru2 output shape:', x.get_shape().as_list())

        x1 = tf.layers.average_pooling1d(x, pool_size=self.sequence_length, strides=[1], \
                padding='VALID', name='avg_pool')
        x2 = tf.layers.max_pooling1d(x, pool_size=self.sequence_length, strides=[1], \
                padding='VALID', name='max_pool')
        x = tf.squeeze(tf.concat([x1, x2], axis=2), axis=[1])
        print('pool concat output shape:', x.get_shape().as_list())

        x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(x, self.fc_hidden_size1)))
        if self.is_training:
            x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)
        x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(x, self.fc_hidden_size2)))

        logits = tf.layers.dense(x, self.num_classes)
        print('logits shape:', logits.get_shape().as_list())
        return logits

    def loss(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        loss_val = tf.reduce_mean(losses)
        tf.summary.scalar('loss', loss_val)
        return loss_val

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimize_op = optimizer.minimize(self.loss_val)
        return optimize_op
