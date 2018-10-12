# -*- coding: utf-8 -*-
"""
@file:	 capsule_layer.py
@author: liaolin (liaolin@baidu.com)
@date:	 Wed 10 Oct 2018 07:25:44 PM CST
"""
import tensorflow as tf
from tensorflow import keras

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vetors), axis, keepdims=True)
    scale = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm \
            + keras.backend.epsilon)
    return scale * vectors


def PrimaryCapsule(inputs, num_channels, dim_capsule, kernel_size, strides):
    primcaps = keras.layers.Conv1D(filters=num_channels*dim_capsule,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='valid',
                                   name='primcap_conv1d')
    primcap_num = primcaps.shape[1].value * num_channels
    primcaps = tf.reshape(primcaps, [-1, primcap_num, dim_capsule], name='primcap_reshape')
    return squash(primcaps)


class DigitCapsule(keras.layers.Layer):
    def __init__(self, digitcap_num, digitcap_dim, routings=3,
                 initializer=keras.initializers.RandomNormal(stddev=0.01), **kwargs):
        self.digitcap_num = digitcap_num
        self.digitcap_dim = digitcap_dim
        self.routings = routings
        self.initializer = initializer
        super(Capsule, self).__init__(kwargs)

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.primcap_num = input_shape[1]
        self.primcap_dim = input_shape[2]

        shape = tf.TensorShape([self.primcap_num, self.digitcap_num, 
                                self.digitcap_dim, self.primcap_dim])
        self.w = self.add_weight(name='w', 
                                 shape=shape, 
                                 initializer=self.initializer,
                                 trainable=True)
        super(Capsule, self).build(input_shape)

    def call(self, inputs):
        w_tiled = tf.tile(self.w, [self.batch_size, 1, 1, 1, 1], name='w_tiled')

        primcap_expanded = tf.expand_dims(inputs, -1, name='primcap_expanded_1')
        primcap_expanded = tf.expand_dims(primcap_expanded, 2, name='primcap_expanded_2')
        primcap_tiled = tf.tile(primcap_expanded, [1, 1, digitcap_num, 1, 1], 
                                name='primcap_tiled')

        digitcap_preds = tf.matmul(w_tiled, primcap_tiled, name='digitcap_preds')
