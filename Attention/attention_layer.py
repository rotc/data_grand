# -*- coding: utf-8 -*-
"""
@file:	 attention_layer.py
@author: liaolin (liaolin@baidu.com)
@date:	 Mon 08 Oct 2018 09:48:16 AM CST
"""
import tensorflow as tf
from tensorflow import keras

class Attention(keras.layers.Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = output_dim
        self.initializer = keras.initializers.RandomNormal(stddev=0.1)
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        shape1 = tf.TensorShape((input_shape[-1], self.attention_size))
        self.w_omega = self.add_weight(name='w_omega', shape=shape1, \
                initializer=self.initializer, trainable=True)
        shape2 = tf.TensorShape((self.attention_size,))
        self.b_omega = self.add_weight(name='b_omega', shape=shape2, \
                initializer=self.initializer, trainable=True)
        self.u_omega = self.add_weight(name='u_omega', shape=shape2, \
                initializer=self.initializer, trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        v = tf.nn.tanh(tf.tensordot(inputs, self.w_omega, axis=1) + self.b_omega)
        vu = tf.tensordot(v, self.u_omega, axis=1, name='vu')
        alpha = tf.nn.softmax(vu, name='alpha')
        output = tf.reduce_sum(tf.multiply(inputs, tf.expand_dims(alpha, -1)), 1)
        return output

    def compute_output_shape(tf, input_shape):
        return tf.TensorShape(input_shape[0], input_shape[-1])

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['attention_size'] = self.attention_size
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
