# -*- coding: utf-8 -*-
"""
@file:	 main.py
@author: liaolin (liaolin@baidu.com)
@date:	 Sat 29 Sep 2018 06:35:10 PM CST
"""
import tensorflow as tf

from two_layer_bigru_model import BiGRUModel

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('vocab_size', 500000, 'maximum vocab size')
