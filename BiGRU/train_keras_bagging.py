# -*- coding: utf-8 -*-
"""
@file:	 train_keras.py
@author: liaolin (liaolin@baidu.com)
@date:	 Sun 07 Oct 2018 05:41:54 PM CST
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import math
import numpy as np
import pandas as pd
import cPickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import ConfigParser

from two_layers_bigru_keras import BiGRUModel
sys.path.append('../Utils')
from data_utils import load_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('vocab_size', 500000, 'maximum vocab size')
tf.app.flags.DEFINE_integer('embedding_size', 400, 'sum embedding size of word2vec and glove')
tf.app.flags.DEFINE_integer('attention_size', 400, 'attention size of attention layer')
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

pretrained_embedding_npy = '../WordEmbedding/embed_matrix.w2v200.glv200.npy'
pretrained_embedding = np.load(pretrained_embedding_npy)

train_sequence, train_labels, test_sequence = load_data(FLAGS.vocab_size)
train_labels = keras.utils.to_categorical(train_labels)
num_train = train_sequence.shape[0]
num_test = test_sequence.shape[0]

oof_train = np.zeros((num_train, FLAGS.num_classes))
oof_test_skf = np.zeros((len(skf), num_test, FLAGS.num_classes))
for i, (train_idx, valid_idx) in enumerate(skf):
    print('Fold {0}/{1}'.format(i + 1, len(skf)))

    model = BiGRUModel(FLAGS.vocab_size, FLAGS.embedding_size, \
            pretrained_embedding, FLAGS.sequence_length, FLAGS.num_classes, FLAGS.dropout_rate, \
            model_config)

    train_idx, valid_idx = skf[0]
    train_off_sequence, train_off_labels, valid_sequence, valid_labels = train_sequence[train_idx], \
            train_labels[train_idx], train_sequence[valid_idx], train_labels[valid_idx]
    num_train_off, num_valid = train_off_sequence.shape[0], valid_sequence.shape[0]
    print('num_train_off: {} num_valid: {}'.format(num_train_off, num_valid))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=6, mode='max')
    lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, \
            mode='max')
    model_save_path = './model_cache/two_layers_bigru_fold{}.hdf5'.format(i + 1)
    model_save = keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_acc', verbose=2, \
            save_best_only=True, save_weights_only=True, mode='max')
    model.fit(train_off_sequence, train_off_labels, batch_size=FLAGS.batch_size, \
        epochs=FLAGS.num_epoches, verbose=2, callbacks=[early_stopping, lr_decay, model_save], \
        validation_data=(valid_sequence, valid_labels))
    
    # validation 
    model.load_weights(model_save_path)
    valid_preds = model.predict(valid_sequence)
    valid_y_pred = np.argmax(valid_preds, axis=1)
    valid_y_true = np.argmax(valid_labels, axis=1)
    eval_f1_score = f1_score(valid_y_true, valid_y_pred, average='macro')
    print('accuracy: {}'.format(accuracy_score(valid_y_true, valid_y_pred)))
    print('f1_score: {}'.format(eval_f1_score))
    oof_train[valid_idx] = valid_preds

    # test
    test_preds = model.predict(test_sequence)
    oof_test_skf[i] = test_preds

    del model
    tf.keras.backend.clear_session()

valid_y_pred = np.argmax(oof_train, axis=1)
valid_y_true = np.argmax(train_labels, axis=1)
eval_f1_score = f1_score(valid_y_true, valid_y_pred, average='macro')
## test prediction
test_ids_npy = '../Input/test_ids.npy'
test_ids = np.load(test_ids_npy)
test_preds = np.mean(oof_test_skf, axis=0)
test_y_pred = np.argmax(test_preds, axis=1) + 1
df_sub = pd.DataFrame({'id':test_ids, 'class':test_y_pred})
sub_names = ['id', 'class']
df_sub[sub_names].to_csv('../Submit/two_layers_bigru_f1{}.csv'.format(eval_f1_score), \
        index=False)
