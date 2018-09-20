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
import pandas as pd

train_csv = '../input/train_set.csv'
train_pkl = '../input/train_set.pkl'
test_csv = '../input/test_set.csv'
test_pkl = '../input/test_set.pkl'

if not os.path.isfile(train_pkl):
    df_train = pd.read_csv(train_csv, sep=',')
    df_train.to_pickle(train_pkl)

if not os.path.isfile(test_pkl):
    df_test = pd.read_csv(test_csv, sep=',')
    df_test.to_pickle(test_pkl)

corpus_word = '../word_embedding/corpus_word'
corpus_char = '../word_embedding/corpus_char'

if not os.path.isfile(corpus_word) or not os.path.isfile(corpus_char):
    df_train = pd.read_pickle(train_pkl)
    df_test = pd.read_pickle(test_pkl)

    df = pd.concat([df_train, df_test], ignore_index=True)

    if not os.path.isfile(corpus_word):
        df['word_seg'].to_csv(corpus_word, header=False, index=False)

    if not os.path.isfile(corpus_char):
        df['article'].to_csv(corpus_char, header=False, index=False)
