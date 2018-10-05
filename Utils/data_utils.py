# -*- coding: utf-8 -*-
"""
@file:	 data_utils.py
@author: liaolin (liaolin@baidu.com)
@date:	 Sun 23 Sep 2018 05:36:28 PM CST
"""
import numpy as np
import pandas as pd
from torchtext import data
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score

def create_vocab(words, vocab_size):
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(vocab_size - 1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reversed_dictionary


def build_dateset(words_list, word2index):
    data_list = []
    for words in words_list:
        data = [word2index.get(word, 0) for word in words]


def load_data(vocab_size):
    train_sequence = np.load('../Input/train_seq.vocab{}.npy'.format(vocab_size))
    test_sequence = np.load('../Input/test_seq.vocab{}.npy'.format(vocab_size))

    df_train = pd.read_pickle('../Input/train_set.pkl')
    df_train['class'] = df_train['class'] - 1

    return train_sequence, df_train['class'], test_sequence


def assign_pretrained_word_embedding(sess, model):
    print('using pre-trained word embedding begin')
    pretrained_embedding_npy = '../WordEmbedding/embed_matrix.w2v200.glv200.npy'
    pretrained_embedding = np.load(pretrained_embedding_npy)
    assign_embedding = tf.assign(model.embedding, pretrained_embedding)
    sess.run(assign_embedding)
    x = sess.run(model.embedding)
    print('using pre-trained word embedding end')


def do_eval(sess, model, batch_generator, num_example, num_batch):
    all_preds = []
    all_labels = []
    for i in xrange(num_batch):
        x_valid, y_valid = sess.run(batch_generator.batch)
        feed_dict = {model.input_x: x_valid, model.input_y: y_valid, model.dropout_keep_prob: 1.0}
        preds = sess.run([model.predictions], feed_dict)
        all_preds.extend(list(np.squeeze(preds)))
        all_labels.extend(list(y_valid))
    eval_score = f1_score(all_labels, all_preds, average='macro')
    return eval_score


class BatchGenerator(object):
    def __init__(self, sequence_length, batch_size, buffer_size):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.features, self.labels = tf.placeholder(tf.int32, shape=[None, self.sequence_length]), \
                tf.placeholder(tf.int32, shape=[None])
        dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
        if buffer_size > 0:
            dataset = dataset.shuffle(buffer_size=self.buffer_size).batch(self.batch_size).repeat()
        else:
            dataset = dataset.batch(self.batch_size).repeat()
        self.iterator = dataset.make_initializable_iterator()
        self.batch = self.iterator.get_next()
