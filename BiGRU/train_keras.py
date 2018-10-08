# -*- coding: utf-8 -*-
"""
@file:	 train_keras.py
@author: liaolin (liaolin@baidu.com)
@date:	 Sun 07 Oct 2018 05:41:54 PM CST
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import math
import numpy as np
import cPickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import ConfigParser

from two_layer_bigru_keras import BiGRUModel
sys.path.append('../Utils')
from data_utils import load_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('vocab_size', 500000, 'maximum vocab size')
tf.app.flags.DEFINE_integer('embedding_size', 400, 'sum embedding size of word2vec and glove')
tf.app.flags.DEFINE_integer('sequence_length', 2000, 'max num words of text')
tf.app.flags.DEFINE_integer('num_epoches', 200, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('num_classes', 19, 'num classes')
tf.app.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding or not')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('dropout_rate', 0.2, 'dropout rate')
tf.app.flags.DEFINE_string('summaries_dir', './log/', 'summary dir')
tf.app.flags.DEFINE_string('ckpt_dir', './ckpt/', 'checkpoint dir')

model_config = ConfigParser.ConfigParser()
model_config.read('../Config/params.conf')
model_config = dict(model_config.items('BiGRU'))

skf_pkl = '../Input/skf.10fold.pkl'
with open(skf_pkl, 'rb') as f:
    skf = cPickle.load(f)

train_sequence, train_labels, test_sequence = load_data(FLAGS.vocab_size)
train_labels = keras.utils.to_categorical(train_labels)
train_idx, valid_idx = skf[0]
train_off_sequence, train_off_labels, valid_sequence, valid_labels = train_sequence[train_idx], \
        train_labels[train_idx], train_sequence[valid_idx], train_labels[valid_idx]
num_train, num_valid = train_off_sequence.shape[0], valid_sequence.shape[0]
print('num_train: {} num_valid: {}'.format(num_train, num_valid))

tr_dataset = tf.data.Dataset.from_tensor_slices((train_off_sequence, train_off_labels))
tr_dataset = tr_dataset.shuffle(buffer_size=num_train).batch(FLAGS.batch_size).repeat()
va_dataset = tf.data.Dataset.from_tensor_slices((valid_sequence, valid_labels))
va_dataset = va_dataset.batch(1024).repeat()

pretrained_embedding_npy = '../WordEmbedding/embed_matrix.w2v200.glv200.npy'
pretrained_embedding = np.load(pretrained_embedding_npy)
model = BiGRUModel(FLAGS.vocab_size, FLAGS.embedding_size, pretrained_embedding, \
        FLAGS.sequence_length, FLAGS.num_classes, FLAGS.dropout_rate, model_config)

#AttributeError: 'TFOptimizer' object has no attribute 'lr'
#model.compile(optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate), \
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=6)
lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3)
num_tr_batch = int(math.ceil(float(num_train) / FLAGS.batch_size))
num_va_batch = int(math.ceil(float(num_valid) / 1024))
model.fit(tr_dataset, epochs=FLAGS.num_epoches, verbose=2, callbacks=[early_stopping, lr_decay], \
        validation_data=va_dataset, steps_per_epoch=num_tr_batch, validation_steps=num_va_batch)
