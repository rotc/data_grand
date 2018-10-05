# -*- coding: utf-8 -*-
"""
@file:	 gen_embed.py
@author: liaolin (liaolin@baidu.com)
@date:	 Mon 24 Sep 2018 09:07:07 PM CST
"""
import sys
import numpy as np
import word2vec

max_num_words = 2000
embed_size = 200

w2v_bin = './word2vec/word2vec_vec{}.bin'.format(embed_size)
w2v_model = word2vec.load(w2v_bin)
w2v_vocab, w2v_vectors = w2v_model.vocab, w2v_model.vectors

gl_embed = './glove/glove_vec{}.txt'.format(embed_size)
gl_model = {}
with open(gl_embed) as f:
    for line in f:
        word, vector = line.strip().split(' ', 1)
        gl_model[word] = vector
gl_vocab = np.array(gl_model.keys())

union_vocab = np.union1d(w2v_vocab, gl_vocab)
vocab_size = union_vocab.size()

embed_w2v_matrix = np.zeros(vocba_size, embed_size)
print 'build embed_w2v_matrix'
for i, word in union_vocab:
    if word in w2v_model.vocab:
        embed_vec = w2v_model(word)
    else:
        embed_vec = np.random.random(embed_size) * 0.5
        embed_vec = embed_vec - embed_vec.mean()
    embed_w2v_matrix[i] = embed_vec

embed_gl_matrix = np.zeros(vocab_size, embed_size)
print 'build embed_gl_matrix'
for i, word in union_vocab:
    if word in gl_vocab:
        embed_vec = gl_model(word)
    else:
        embed_vec = np.random.random(embed_size) * 0.5
        embed_vec = embed_vec - embed_vec.mean()
