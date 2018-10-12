# -*- coding: utf-8 -*-
"""
@file:	 two_layer_bigru_keras.py
@author: liaolin (liaolin@baidu.com)
@date:	 Sun 07 Oct 2018 09:39:30 AM CST
"""
import sys
import tensorflow as tf
from tensorflow import keras
sys.path.append('../CustomLayer')
from attention_layer import Attention

class BiGRUModel(keras.Model):
    def __init__(self, vocab_size, embedding_size, attention_size, pretrained_embedding, sequence_length, \
            num_classes, dropout_rate, model_config):
        super(BiGRUModel, self).__init__(name='bi_gru_model')
        self.vocab_size = vocab_size + 1
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.rnn_state_size = int(model_config['rnn_state_size'])
        self.fc_hidden_size1 = int(model_config['fc_hidden_size1'])
        self.fc_hidden_size2 = int(model_config['fc_hidden_size2'])

        self.embedding = keras.layers.Embedding(input_dim=self.vocab_size, \
                output_dim=self.embedding_size, input_length=self.sequence_length, \
                weights=[pretrained_embedding], trainable=False)
        self.spatial_dropout_1d = keras.layers.SpatialDropout1D(rate=dropout_rate)
        self.bidirectional_gru_1 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(units=\
                self.rnn_state_size, return_sequences=True))
        self.bidirectional_gru_2 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(units=\
                self.rnn_state_size, return_sequences=True))
        self.attention = Attention(self.attention_size)
        #self.global_average_pooling_1d = keras.layers.GlobalAveragePooling1D()
        #self.global_max_pooling_1d = keras.layers.GlobalMaxPooling1D()
        #self.concatenate = keras.layers.Concatenate()
        self.dropout = keras.layers.Dropout(rate=dropout_rate)
        self.activation = keras.layers.Activation('relu')
        self.batch_normalization_1 = keras.layers.BatchNormalization()
        self.batch_normalization_2 = keras.layers.BatchNormalization()
        self.dense_1 = keras.layers.Dense(self.fc_hidden_size1)
        self.dense_2 = keras.layers.Dense(self.fc_hidden_size2)
        self.dense_3 = keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        x = self.spatial_dropout_1d(self.embedding(inputs))
        x = self.bidirectional_gru_1(x)
        x = self.bidirectional_gru_2(x)
        #x1 = self.global_average_pooling_1d(x)
        #x2 = self.global_max_pooling_1d(x)
        #x = self.concatenate([x1, x2])
        x = self.attention(x)
        x = self.dropout(self.activation(self.batch_normalization_1(self.dense_1(x))))
        x = self.activation(self.batch_normalization_2(self.dense_2(x)))
        return self.dense_3(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
