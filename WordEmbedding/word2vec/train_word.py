#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
@file:	 train_word.py
@author: liaolin (liaolin@baidu.com)
@date:	 Fri 21 Sep 2018 08:25:02 PM CST
@brief:	 train word embedding  
"""
import word2vec

vector_size= 200

word2vec.word2vec('../corpus_word', 'word2vec_vec200.bin', size=vector_size, verbose=True)
