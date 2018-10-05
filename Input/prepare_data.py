#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
@file:	 prepare_data.py
@author: liaolin (liaolin@baidu.com)
@date:	 Thu 20 Sep 2018 09:27:48 AM CST
@brief:	 prepare data 
"""
import os
import sys
import numpy as np
import pandas as pd
import word2vec
import cPickle
from tensorflow.keras.preprocessing import sequence, text
from sklearn.model_selection import StratifiedKFold

seed = 2018
np.random.seed(seed)

# prepare input pkl
train_csv = '../Input/train_set.csv'
train_pkl = '../Input/train_set.pkl'
test_csv = '../Input/test_set.csv'
test_pkl = '../Input/test_set.pkl'

if not os.path.isfile(train_pkl):
    df_train = pd.read_csv(train_csv, sep=',')
    df_train.to_pickle(train_pkl)

if not os.path.isfile(test_pkl):
    df_test = pd.read_csv(test_csv, sep=',')
    df_test.to_pickle(test_pkl)

nfold = 10
seed = 2018
skf_pkl = '../Input/skf.{}fold.pkl'.format(nfold)
if not os.path.isfile(skf_pkl):
    df_train = pd.read_pickle(train_pkl)
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    list_skf = list(skf.split(df_train.values, df_train['class'].values))
    with open(skf_pkl, 'wb') as f:
        cPickle.dump(list_skf, f)

# prepare corpus for training word embedding
corpus_word = '../WordEmbedding/corpus_word'
corpus_char = '../WordEmbedding/corpus_char'

if not os.path.isfile(corpus_word) or not os.path.isfile(corpus_char):
    df_train = pd.read_pickle(train_pkl)
    df_test = pd.read_pickle(test_pkl)

    df = pd.concat([df_train, df_test], ignore_index=True)

    if not os.path.isfile(corpus_word):
        df['word_seg'].to_csv(corpus_word, header=False, index=False)

    if not os.path.isfile(corpus_char):
        df['article'].to_csv(corpus_char, header=False, index=False)

# map word to index, generate pretrained embedding matrix
vocab_size = 500000
embed_size = 200
max_num_words = 2000

col = 'word_seg'
train_seq_npy = '../Input/train_seq.vocab{}.npy'.format(vocab_size)
test_seq_npy = '../Input/test_seq.vocab{}.npy'.format(vocab_size)
w2v_embed_matrix_npy = '../WordEmbedding/embed_matrix.w2v{}.npy'.format(embed_size)
gv_embed_matrix_npy = '../WordEmbedding/embed_matrix.gv{}.npy'.format(embed_size)
mix_embed_matrix_npy = '../WordEmbedding/embed_matrix.w2v{0}.glv{0}.npy'.format(embed_size)
w2v_file = '../WordEmbedding/word2vec/word2vec_vec{}.bin'.format(embed_size)
gv_file = '../WordEmbedding/glove/glove_vec{}.txt'.format(embed_size)

if not os.path.isfile(train_seq_npy) or not os.path.isfile(test_seq_npy) \
        or not os.path.isfile(mix_embed_matrix_npy):
    df_train = pd.read_pickle(train_pkl)
    df_test = pd.read_pickle(test_pkl)
    df = pd.concat([df_train, df_test], ignore_index=True)

    tokenizer = text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df[col].values)

    train_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(df_train[col].values), maxlen=max_num_words)
    test_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(df_test[col].values), maxlen=max_num_words)
    np.save(train_seq_npy.rsplit('.npy', 1)[0], train_seq)
    np.save(test_seq_npy.rsplit('.npy', 1)[0], test_seq)

    word_index = {w: i for w, i in tokenizer.word_index.iteritems() if i <= vocab_size}
    num_words = len(word_index)
    print 'num_words:', num_words

    print 'add word2vec begin'
    if not os.path.isfile(w2v_embed_matrix_npy):
        if not os.path.isfile(w2v_file):
            w2v_model = word2vec.word2vec('../WordEmbedding/corpus_word', w2v_file, size=embed_size, verbose=True)
        else: 
            w2v_model = word2vec.load(w2v_file)

        w2v_embed_matrix = np.zeros((num_words + 1, embed_size))
        for word, i in word_index.iteritems():
            if word in w2v_model:
                w2v_embed_matrix[i] = w2v_model[word]
            else:
                unk_vec = np.random.random(embed_size) * 0.5
                unk_vec = unk_vec - unk_vec.mean()
                w2v_embed_matrix[i] = unk_vec
        np.save(w2v_embed_matrix_npy.rsplit('.npy', 1)[0], w2v_embed_matrix)
    else:
        w2v_embed_matrix = np.load(w2v_embed_matrix_npy)
    print 'add word2vec finish'

    print 'add glove begin'
    if not os.path.isfile(gv_embed_matrix_npy):
        if not os.path.isfile(gv_file):
            cmd = 'cd ../WordEmbedding/glove; sh train.sh'
            print cmd
            os.system(cmd)
        gv_model = {}
        with open(gv_file) as f:
            for line in f:
                data = line.strip().split(' ')
                word = data[0]
                vector = np.array(data[1:], dtype='float32')
                gv_model[word] = vector

        gv_embed_matrix = np.zeros((num_words + 1, embed_size))
        for word, i in word_index.iteritems():
            if word in gv_model:
                gv_embed_matrix[i] = gv_model[word]
            else:
                unk_vec = np.random.random(embed_size) * 0.5
                unk_vec = unk_vec - unk_vec.mean()
                gv_embed_matrix[i] = unk_vec
        np.save(gv_embed_matrix_npy.rsplit('.npy', 1)[0], gv_embed_matrix)
    else:
        gv_embed_matrix = np.load(gv_embed_matrix_npy)
    print 'add glove finish'

    mix_embed_matrix = np.concatenate((w2v_embed_matrix, gv_embed_matrix), axis=1)
    np.save(mix_embed_matrix_npy, mix_embed_matrix)
