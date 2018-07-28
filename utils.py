# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import csv
import tensorflow as tf
import pandas as pd
import re
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from six.moves import urllib

urlretrieve = urllib.request.urlretrieve

def process_imdb():
    '''Load data into RAM'''
    with np.load('imdb.npz') as f:
        print('Preparing train set...')
        x_train, y_train = f['x_train'], f['y_train']
        print('Preparing test set...')
        x_test, y_test = f['x_test'], f['y_test']
    return x_train, x_test, y_train, y_test


def maybe_download_imdb(src="https://ikpublictutorial.blob.core.windows.net/deeplearningframeworks/imdb.npz"):
    '''Load the training and testing data
    Mirror of: https://s3.amazonaws.com/text-datasets/imdb.npz'''
    try:
        return process_imdb()
    except:
        # Catch exception that file doesn't exist
        # Download
        print('Data does not exist. Downloading ' + src)
        fname, h = urlretrieve(src, './imdb.npz')
        # No need to extract
        x_train, x_test, y_train, y_test = process_imdb()
        return x_train, x_test, y_train, y_test


def imdb_for_library(seq_len=100, max_features=20000, one_hot=False):
    ''' Replicates same pre-processing as:
    https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
    I'm not sure if we want to load another version of IMDB that has got
    words, but if it does have words we would still convert to index in this
    backend script that is not meant for others to see ...
    But I'm worried this obfuscates the data a bit?
    '''
    # 0 (padding), 1 (start), 2 (OOV)
    START_CHAR = 1
    OOV_CHAR = 2
    INDEX_FROM = 3
    # Raw data (has been encoded into words already)
    x_train, x_test, y_train, y_test = maybe_download_imdb()
    # Combine for processing
    idx = len(x_train)
    _xs = np.concatenate([x_train, x_test])
    # Words will start from INDEX_FROM (shift by 3)
    _xs = [[START_CHAR] + [w + INDEX_FROM for w in x] for x in _xs]
    # Max-features - replace words bigger than index with oov_char
    # E.g. if max_features = 5 then keep 0, 1, 2, 3, 4 i.e. words 3 and 4
    if max_features:
        print("Trimming to {} max-features".format(max_features))
        _xs = [[w if (w < max_features) else OOV_CHAR for w in x] for x in _xs]
        # Pad to same sequences
    print("Padding to length {}".format(seq_len))
    xs = np.zeros((len(_xs), seq_len), dtype=np.int)
    for o_idx, obs in enumerate(_xs):
        # Match keras pre-processing of taking last elements
        obs = obs[-seq_len:]
        for i_idx in range(len(obs)):
            if i_idx < seq_len:
                xs[o_idx][i_idx] = obs[i_idx]
    # One-hot
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = np.array(xs[:idx]).astype(np.int32)
    x_test = np.array(xs[idx:]).astype(np.int32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test


class DataProcessor(object):
    def __init__(self,data_file,vocab_size=20000,seperator=',',remove_special=True,lower=True, max_seq_len=50,header=None,
                    reverse=False, raw_data=False):
        self.data_file = data_file
        self.vocab_size = vocab_size
        self.seperator = seperator
        self.max_seq_len = max_seq_len
        self.raw_data = raw_data
        self.lower = lower
        self.reverse = reverse
        self.remove_special = remove_special
        self._raw_data , self._raw_labels = self._load_data(self.data_file,header=header)
        self.label_to_id = self._build_vocab_label()
        self.labels = np.asarray([self.label_to_id[i] for i in self._raw_labels])
        if not self.raw_data :
            self.word_to_id =  self._build_vocab()
            self.data = np.asarray(self._text_to_word_ids(self._raw_data))



    def _load_data(self,filename,contains_label=True,header=None):
        df = pd.read_csv(filename,sep=self.seperator,header=header)
        column_names = df.columns.values
        data = df[column_names[0]].values.tolist()
        if contains_label:
            label = [i.strip().lower() for i in df[column_names[1]].values.tolist()]
            return data, label
        return data

    def _split_to_words(self,text):
        if self.remove_special:
            text = re.sub(r'[^0-9a-zA-Z\?\.\s]', ' ', text.lower() if self.lower \
                                                                else text)
        return re.split('\s+',text)

    def _build_vocab(self):
        data = []
        for text in self._raw_data:
            data.extend(self._split_to_words(text))
        counter = collections.Counter(data)
        count_pairs = sorted(counter.most_common(self.vocab_size), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(1,len(words)+1)))
        return word_to_id

    def _build_vocab_label(self):
        counter = collections.Counter(self._raw_labels)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        labels, _ = list(zip(*count_pairs))
        label_to_id = dict(zip(labels, range(0,len(labels))))
        return label_to_id

    def _text_to_word_ids(self,text_list,reverse=False):

        text_ids = []
        for text_items in text_list:

            data = self._split_to_words(text_items)
            if self.reverse:
                data.reverse()
            # text_ids.append(([self.word_to_id[word] for word in data if word in self.word_to_id] + \
            #                            [0]*self.max_seq_len)[:self.max_seq_len])
            temp = ([self.word_to_id[word] for word in data if word in self.word_to_id] )
            if len(temp) < self.max_seq_len:
                temp =  ([0]*self.max_seq_len + temp)[-self.max_seq_len:]
            else:
                temp = temp[:self.max_seq_len]
            text_ids.append(temp)
        return text_ids


    def _convert_one_hot(self,data):
        pass

    def get_training_data(self,raw_text=False):
        if raw_text or self.raw_data:
            return self._raw_data, self.labels
        return self.data, self.labels

    def process_test_file(self,filename,contains_label=False,header=None):
        if contains_label:
            raw_test_data, raw_labels = self._load_data(filename,contains_label,header)
            test_data = raw_test_data
            if not self.raw_data:
                test_data = np.asarray(self._text_to_word_ids(raw_test_data))
            labels = np.asarray([self.label_to_id[i] for i in raw_labels])
            return test_data, labels
        else:
            raw_test_data, raw_labels = self._load_data(filename,contains_label,header)
            test_data = raw_test_data
            if not self.raw_data:
                test_data = np.asarray(self._text_to_word_ids(raw_test_data))
            labels = np.asarray([self.label_to_id[i] for i in raw_labels])
            return test_data

    def _load_glove(self,dim):
        """ Loads GloVe data.
        :param dim: word vector size (50, 100, 200)
        :return: GloVe word table
        """
        word2vec = {}
        print('Loading Glove Data.. Please Wait.. ')
        path = "data/glove/glove.6B." + str(dim) + 'd'
        if os.path.exists(path + '.cache'):
            with open(path + '.cache', 'rb') as cache_file:
                word2vec = pickle.load(cache_file)

        else:
            # Load n create cache
            with open(path + '.txt') as f:
                for line in f:
                    l = line.split()
                    word2vec[l[0]] = [float(x) for x in l[1:]]

            with open(path + '.cache', 'wb') as cache_file:
                pickle.dump(word2vec, cache_file)

        print("Loaded Glove data")
        return word2vec

    def get_embedding(self,dim):
        embedding = np.random.normal(loc=0.0, scale=0.1, size=[len(self.word_to_id)+1,dim])
        glove = self._load_glove(dim)
        for item in self.word_to_id:
            if item.lower() in glove:
                embedding[self.word_to_id[item]] = glove[item]
        return embedding



import math
import random
import numpy as np
from keras import utils


def _roundto(val, batch_size):
    return int(math.ceil(val / batch_size)) * batch_size


class BucketedSequence(utils.Sequence):
    """
    A Keras Sequence (dataset reader) of input sequences read in bucketed bins.
    Assumes all inputs are already padded using `pad_sequences` (where padding
    is prepended).
    """

    def __init__(self, num_buckets, batch_size, seq_lengths, x_seq, y):
        self.batch_size = batch_size
        # Count bucket sizes
        bucket_sizes, bucket_ranges = np.histogram(seq_lengths,
                                                   bins=num_buckets)

        # Obtain the (non-sequence) shapes of the inputs and outputs
        input_shape = (1,) if len(x_seq.shape) == 2 else x_seq.shape[2:]
        output_shape = (1,) if len(y.shape) == 1 else y.shape[1:]

        # Looking for non-empty buckets
        actual_buckets = [bucket_ranges[i+1]
                          for i,bs in enumerate(bucket_sizes) if bs > 0]
        actual_bucketsizes = [bs for bs in bucket_sizes if bs > 0]
        bucket_seqlen = [int(math.ceil(bs)) for bs in actual_buckets]
        num_actual = len(actual_buckets)
        print('Training with %d non-empty buckets' % num_actual)
        #print(bucket_seqlen)
        #print(actual_bucketsizes)
        self.bins = [(np.ndarray([bs, bsl] + list(input_shape), dtype=x_seq.dtype),
                      np.ndarray([bs] + list(output_shape), dtype=y.dtype))
                     for bsl,bs in zip(bucket_seqlen, actual_bucketsizes)]
        assert len(self.bins) == num_actual

        # Insert the sequences into the bins
        bctr = [0]*num_actual
        for i,sl in enumerate(seq_lengths):
            for j in range(num_actual):
                bsl = bucket_seqlen[j]
                if sl < bsl or j == num_actual - 1:
                    self.bins[j][0][bctr[j],:bsl] = x_seq[i,-bsl:]
                    self.bins[j][1][bctr[j],:] = y[i]
                    bctr[j] += 1
                    break

        self.num_samples = x_seq.shape[0]
        self.dataset_len = int(sum([math.ceil(bs / self.batch_size)
                                    for bs in actual_bucketsizes]))
        self._permute()

    def _permute(self):
        # Shuffle bins
        random.shuffle(self.bins)

        # Shuffle bin contents
        for i, (xbin, ybin) in enumerate(self.bins):
            index_array = np.random.permutation(xbin.shape[0])
            self.bins[i] = (xbin[index_array], ybin[index_array])

    def on_epoch_end(self):
        self._permute()

    def __len__(self):
        """ Returns the number of minibatches in this sequence. """
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin, idx_end = self.batch_size*idx, self.batch_size*(idx+1)

        # Obtain bin index
        for i,(xbin,ybin) in enumerate(self.bins):
            rounded_bin = _roundto(xbin.shape[0], self.batch_size)
            if idx_begin >= rounded_bin:
                idx_begin -= rounded_bin
                idx_end -= rounded_bin
                continue

            # Found bin
            idx_end = min(xbin.shape[0], idx_end) # Clamp to end of bin

            return xbin[idx_begin:idx_end], ybin[idx_begin:idx_end]


        raise ValueError('out of bounds')


if __name__ == '__main__':
    data_path = 'data/custom/LabelledData.txt'
    processor = DataProcessor(data_path,seperator=',,,',max_seq_len=30)
    X, y = processor.get_training_data()
    print(X.shape, y.shape)
