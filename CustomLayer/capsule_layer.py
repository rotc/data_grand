# -*- coding: utf-8 -*-
"""
@file:	 capsule_layer.py
@author: liaolin (liaolin@baidu.com)
@date:	 Wed 10 Oct 2018 07:25:44 PM CST
"""
import tensorflow as tf
from tensorflow import keras

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm \
            + keras.backend.epsilon())
    return scale * vectors


def PrimaryCapsule(inputs, num_channels, dim_capsule, kernel_size, strides):
    primcaps = keras.layers.Conv1D(filters=num_channels*dim_capsule,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='valid',
                                   name='primcap_conv1d')(inputs)
    primcap_num = primcaps.shape[1].value * num_channels
    primcaps = tf.reshape(primcaps, [-1, primcap_num, dim_capsule], name='primcap_reshape')
    return squash(primcaps)


class Mask(keras.layers.Layer):
    def __init__(self, digitcap_num, digitcap_dim):
        self.digitcap_num = digitcap_num
        self.digitcap_dim = digitcap_dim
        super(Mask, self).__init__(kwargs)

    def call(self, inputs):
        if type(inputs) == list:
            inputs, mask = inputs
        else:
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=2))
            y_pred = tf.argmax(x, axis=1)
            mask = tf.one_hot(y_pred, depth=self.digitcap_num)
            
        masked = tf.multiply(inputs, tf.expand_dims(mask, -1))
        return tf.reshape(masked, [-1, self.digitcap_num * self.digitcap_dim])


class DigitCapsule(keras.layers.Layer):
    #def __init__(self, batch_size, digitcap_num, digitcap_dim, routings=3,
    #             initializer=keras.initializers.RandomNormal(stddev=0.01), **kwargs):
    def __init__(self, digitcap_num, digitcap_dim, routings=3,
                 initializer=keras.initializers.RandomNormal(stddev=0.01), **kwargs):
        self.digitcap_num = digitcap_num
        self.digitcap_dim = digitcap_dim
        self.routings = routings
        self.initializer = initializer
        super(DigitCapsule, self).__init__(kwargs)

    def build(self, input_shape):
        self.primcap_num = input_shape[1]
        self.primcap_dim = input_shape[2]

        shape = tf.TensorShape([1, self.primcap_num, self.digitcap_num, 
                                self.digitcap_dim, self.primcap_dim])
        self.w = self.add_weight(name='w', 
                                 shape=shape, 
                                 initializer=self.initializer,
                                 trainable=True)
        super(DigitCapsule, self).build(input_shape)

    def call(self, inputs):
        self.batch_size = tf.shape(inputs)[0]
        w_tiled = tf.tile(self.w, [self.batch_size, 1, 1, 1, 1], name='w_tiled')

        primcap_expanded = tf.expand_dims(inputs, -1, name='primcap_expanded_1')
        primcap_expanded = tf.expand_dims(primcap_expanded, 2, name='primcap_expanded_2')
        primcap_tiled = tf.tile(primcap_expanded, [1, 1, self.digitcap_num, 1, 1], 
                                name='primcap_tiled')

        digitcap_preds = tf.matmul(w_tiled, primcap_tiled, name='digitcap_preds')
        b = tf.zeros([self.batch_size, self.primcap_num, self.digitcap_num, 1, 1], name='b')

        for i in xrange(self.routings):
            c = tf.nn.softmax(b, axis=2)
            weighted_preds = tf.multiply(c, digitcap_preds)
            s = tf.reduce_sum(weighted_preds, axis=1, keepdims=True)
            v = squash(s, axis=-2)

            if i < self.routings - 1:
                v_tiled = tf.tile(v, [1, self.primcap_num, 1, 1, 1])
                agreement = tf.matmul(digitcap_preds, v_tiled, transpose_a=True)
                b += agreement

        return tf.squeeze(v, [1, 4])

    def compute_output_shape(self, input_shape):
        shape = [self.batch_size, 1, self.digitcap_num, self.digitcap_dim, 1]
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(DigitCapsule, self).get_config()
        config = {
            'digitcap_num': self.digitcap_num,
            'digitcpa_dim': self.digitcap_dim,
            'routings': self.routings
            }
        base_config.update(config)
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
