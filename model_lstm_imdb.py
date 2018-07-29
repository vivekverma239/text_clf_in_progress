'''
Code to classify sentiments

'''


# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import DataProcessor
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import DataProcessor, imdb_for_library, BucketedSequence
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
# from encoder import LSTMEncoderWithEmbedding, CNNEncoderWithEmbedding
from keras.layers import Dense, Embedding
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, Dropout

from tqdm import tqdm, trange
from time import sleep
import config
from config import *
import sys


class LSTMModel(object):

  def __init__(self, config, pretrained_embedding):
    self._input         = tf.placeholder(dtype=tf.int32,shape=[None,None],name='input')
    self._target        = tf.placeholder(dtype=tf.int32,shape=[None],name='target')
    self.batch_size     = config['batch_size']
    self.num_steps      = config['num_steps']
    self.embed_size     = config['embed_size']
    self.size           = config['hidden_size']
    self._lr            = config['lr']
    self.num_classes    = config['num_classes']
    self.keep_prob      = tf.Variable(config['keep_prob'],trainable=False)
    self.combine_mode   = config['combine_mode']
    self.weight_decay   = config['weight_decay']


    #
    # outputs = LSTMEncoderWithEmbedding(self._input,self.embed_size,self.size,\
    #                          config['vocab_size'],self.num_steps,\
    #                          self.keep_prob,embedding=pretrained_embedding,\
    #                          num_layers=config['num_layers'],\
    #                          variational_dropout=True,\
    #                          combine_mode='last').get_output()

    embed = Embedding(config['vocab_size']+1, self.embed_size)(self._input)
    outputs = tf.nn.dropout(embed,keep_prob=self.keep_prob)
    output = CuDNNLSTM(self.size)(embed)
    outputs = tf.nn.dropout(output,keep_prob=self.keep_prob)

    embed_avg = tf.reduce_mean(embed,axis=1)
    # embed_max = tf.reduce_max(embed,axis=1)
    # embed_min = tf.reduce_min(embed,axis=1)
    # outputs = tf.concat([outputs,embed_avg,embed_min,embed_max],axis=-1)
    outputs = tf.concat([outputs,embed_avg],axis=-1)
    # outputs = tf.contrib.layers.fully_connected(outputs,self.size)
    # outputs = tf.nn.dropout(outputs,keep_prob=self.keep_prob)
    # softmax_w = tf.get_variable("softmax_w", [self.size, self.num_classes], dtype=tf.float32)
    # softmax_b = tf.get_variable("softmax_b", [self.num_classes], dtype=tf.float32)
    # logits    = tf.matmul(outputs, softmax_w) + softmax_b
    logits = Dense(self.num_classes,activation=None)(outputs)


    # update the cost variables
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._target,logits=logits)
    self.l2_loss =  sum(tf.nn.l2_loss(tf_var)
        for tf_var in tf.trainable_variables()
        )
    self._cost = cost = tf.reduce_mean(loss) + self.weight_decay*self.l2_loss

    self._lr = tf.Variable(self._lr, trainable=False)
    tvars    = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config['max_grad_norm'])
    optimizer = tf.train.AdamOptimizer(self._lr)
    # optimizer = tf.train.GradientDescentOptimizer(self._lr)

    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    self.predicted_class = tf.cast(tf.argmax(tf.nn.softmax(logits),axis=-1),tf.int32)


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def target(self):
    return self._target

  @property
  def predict(self):
    return self.predicted_class

  @property
  def cost(self):
    return self._cost

  @property
  def lr(self):
    return self._lrtrain_op

  @property
  def train_op(self):
    return self._train_op


class Trainer(object):
    def __init__(self,config,X,y,embedding):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                X, y, test_size=0.5, random_state=12,shuffle=True)
        seq_len_train = np.sum((self.X_train > 0).astype(np.int32),-1)
        seq_len_test = np.sum((self.X_test > 0).astype(np.int32),-1)
        self.bucketed_sequence_train = BucketedSequence(num_buckets=config['num_buckets'],\
                                                batch_size=config['batch_size'],\
                                                seq_lengths=seq_len_train,\
                                                x_seq=self.X_train, y=self.y_train)
        self.bucketed_sequence_test = BucketedSequence(num_buckets=config['num_buckets'],\
                                                batch_size=config['batch_size'],\
                                                seq_lengths=seq_len_test,\
                                                x_seq=self.X_test, y=self.y_test)
        # self.bucketed_sequence_train = None
        # self.bucketed_sequence_test = None
        self.keep_prob = config['keep_prob']
        self.batch_size = config['batch_size']
        self.model = LSTMModel(config,embedding)
        self.train_op = self.model.train_op
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.num_epochs = 200
        self.epoch = 0
        self.lr = config['lr']
        self.save_path = config['save_path']
        self.lr_decay = config['lr_decay']
        self.max_epoch= config['max_epoch']
        self.max_max_epoch= config['max_max_epoch']

    def train(self):
        cost_history = []
        best_cost = 10000000
        early_stopping_epochs = 4
        for idx in range(self.max_max_epoch):
            if idx >= self.max_epoch :
                self.lr *=self.lr_decay
                self.model.assign_lr(self.sess,self.lr)
            train_cost, train_accuracy = self.run_epoch(self.X_train,self.y_train,\
                                                    training=True,\
                                                bucketed_sequence=self.bucketed_sequence_train)

            # print('Cost %0:.2f , Accuracy: 0:.2f'.format(cost,accuracy))
            val_cost, val_accuracy = self.run_epoch(self.X_test,self.y_test,\
                                                bucketed_sequence=self.bucketed_sequence_test)
            # self.bucketed_sequence_train.on_epoch_end()
            # self.bucketed_sequence_test.on_epoch_end()
            cost_history.append(val_cost)
            print('BATCH %d |Training: Cost %f , Accuracy: %f  | Validation: Cost %f , Accuracy: %f ' \
                            %(idx, round(train_cost,2),round(train_accuracy*100,2),\
                                round(val_cost,2),round(val_accuracy*100,2)))
            if val_cost < best_cost:
                best_cost = val_cost
                self.saver.save(self.sess, self.save_path +"model.ckpt")
            if len(cost_history) > early_stopping_epochs \
                            and min(cost_history[-early_stopping_epochs:]) != best_cost:
                print("Loss hasn't improved in {} epochs, stopping early".format(early_stopping_epochs) )
                break


    def run_epoch(self,X,y, training=False,bucketed_sequence=None):
        temp_cost  = []
        temp_acc   = []
        feed_dict  = {self.model.keep_prob: 1}
        ops        = [self.model.cost, self.model.predict]
        y_true = []
        if training:
            feed_dict[self.model.keep_prob] = self.keep_prob
            ops = [self.train_op,self.model.cost, self.model.predict]
        if not bucketed_sequence:
            for i in trange(int(len(X)/self.batch_size)+1):
                X_sub = X[i*self.batch_size: (i+1)*self.batch_size,:]
                y_sub = y[i*self.batch_size: (i+1)*self.batch_size]
                y_true.append(y_sub)
                feed_dict[self.model.input] = X_sub
                feed_dict[self.model.target] = y_sub
                ops_out = self.sess.run(ops,feed_dict)
                temp_cost.append(ops_out[-2])
                temp_acc.append( ops_out[-1] )
        else:
            for i in trange(len(bucketed_sequence)):
                X_sub, y_sub  = bucketed_sequence[i]
                y_true.append(y_sub)
                feed_dict[self.model.input] = X_sub
                feed_dict[self.model.target] = y_sub
                ops_out = self.sess.run(ops,feed_dict)
                temp_cost.append(ops_out[-2])
                temp_acc.append( ops_out[-1] )
        cost = np.mean(temp_cost)
        # print(classification_report(y_pred=self.predict(X),y_true=y))
        accuracy = accuracy_score(y_pred=np.concatenate(temp_acc),y_true=np.concatenate(y_true,axis=-1))
        return cost, accuracy

    def predict(self,X):
        temp_pred  = []
        feed_dict  = {self.model.keep_prob: 1}
        ops        = [self.model.predict]

        # for i in range(int(len(X)/self.batch_size)+1):
        #     X_sub = X[i*self.batch_size: (i+1)*self.batch_size,:]
        #     feed_dict[self.model.input] = X_sub
        #     ops_out = self.sess.run(ops,feed_dict)
        #     temp_pred.append(ops_out[0])

        for i in range(int(len(X))):
            X_sub = X[i*self.batch_size: (i+1)*self.batch_size,:]
            feed_dict[self.model.input] = X_sub
            ops_out = self.sess.run(ops,feed_dict)
            temp_pred.append(ops_out[0])

        return np.concatenate(temp_pred)

    def load_best_model(self):
        self.saver.restore(self.sess, self.save_path +"model.ckpt")
if __name__ == '__main__':

    # config = {
    #     'init_scale': 0.1,
    #     'lr': .001,
    #     'lr_decay': 0.8,
    #     'max_epoch':7,
    #     'max_max_epoch':40,
    #     'max_grad_norm': 5,
    #     'num_layers': 1,
    #     'num_steps': 300,
    #     'embed_size': 300,
    #     'hidden_size':300,
    #     'keep_prob': 0.5,
    #     'batch_size': 32,
    #     'num_classes': 2,
    #     'vocab_size': 40000,
    #     'combine_mode': 'last',
    #     'weight_decay': 1e-8,
    #     'save_path': 'checkpoint/imdb/'
    # }



    if len(sys.argv) ==2 and  sys.argv[1] == 'std':
        x_train, x_test, y_train, y_test = imdb_for_library(seq_len=config['num_steps'], max_features=config['vocab_size'])
    else:
        data_path = 'data/imdb/train.csv'
        data_processor = DataProcessor(data_path,vocab_size=config['vocab_size'],\
                        seperator=',',max_seq_len=config['num_steps'],header=0,reverse=True)
        x_train , y_train    = data_processor.get_training_data()
        x_test, y_test = data_processor.process_test_file( 'data/imdb/test.csv',contains_label=True,header=0)
        # embedding = data_processor.get_embedding(config['embed_size'])
    #     # print('Embedding Shape',embedding.shape)

    trainer = Trainer(config,x_train,y_train,embedding=None)
    trainer.train()
    trainer.load_best_model()
    pred = trainer.predict(x_test)
    print(classification_report(y_true=y_test,y_pred=pred))
