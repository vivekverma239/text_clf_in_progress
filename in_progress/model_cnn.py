'''
Code to classify sentiments

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import DataProcessor
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import DataProcessor
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from encoder import LSTMEncoderWithEmbedding, CNNEncoderWithEmbedding


class CNNModel(object):

  def __init__(self, config, pretrained_embedding):
    self._input         = tf.placeholder(dtype=tf.int32,shape=[None,config['num_steps']],name='input')
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

    outputs = CNNEncoderWithEmbedding(self._input,self.embed_size,self.size,\
                             config['vocab_size'],self.num_steps,\
                             self.keep_prob,embedding=pretrained_embedding,\
                             num_layers=config['num_layers'],\
                             variational_dropout=True,\
                             combine_mode='last').get_output()

    # outputs = tf.contrib.layers.fully_connected(outputs,self.size)
    # outputs = tf.nn.dropout(outputs,keep_prob=self.keep_prob)
    softmax_w = tf.get_variable("softmax_w", [self.size, self.num_classes], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [self.num_classes], dtype=tf.float32)
    logits    = tf.matmul(outputs, softmax_w) + softmax_b


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
                                data, labels, test_size=0.5, random_state=12,shuffle=True)

        self.keep_prob = config['keep_prob']
        self.batch_size = config['batch_size']
        self.model = CNNModel(config,embedding)
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
            train_cost, train_accuracy = self.run_epoch(self.X_train,self.y_train, training=True)
            # print('Cost %0:.2f , Accuracy: 0:.2f'.format(cost,accuracy))
            val_cost, val_accuracy = self.run_epoch(self.X_test,self.y_test)
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


    def run_epoch(self,X,y, training=False):
        temp_cost  = []
        temp_acc   = []
        feed_dict  = {self.model.keep_prob: 1}
        ops        = [self.model.cost, self.model.predict]
        if training:
            feed_dict[self.model.keep_prob] = self.keep_prob
            ops = [self.train_op,self.model.cost, self.model.predict]
        for i in range(int(len(X)/self.batch_size)+1):
            X_sub = X[i*self.batch_size: (i+1)*self.batch_size,:]
            y_sub = y[i*self.batch_size: (i+1)*self.batch_size]
            feed_dict[self.model.input] = X_sub
            feed_dict[self.model.target] = y_sub
            ops_out = self.sess.run(ops,feed_dict)
            temp_cost.append(ops_out[-2])
            temp_acc.append( ops_out[-1] )
        cost = np.mean(temp_cost)
        # print(classification_report(y_pred=self.predict(X),y_true=y))
        accuracy = accuracy_score(y_pred=np.concatenate(temp_acc),y_true=y)
        return cost, accuracy

    def predict(self,X):
        temp_pred  = []
        feed_dict  = {self.model.keep_prob: 1}
        ops        = [self.model.predict]

        for i in range(int(len(X)/self.batch_size)+1):
            X_sub = X[i*self.batch_size: (i+1)*self.batch_size,:]
            feed_dict[self.model.input] = X_sub
            ops_out = self.sess.run(ops,feed_dict)
            temp_pred.append(ops_out[0])

        return np.concatenate(temp_pred)

    def load_best_model(self):
        self.saver.restore(self.sess, self.save_path +"model.ckpt")
if __name__ == '__main__':

    config = {
        'init_scale': 0.1,
        'lr': .001,
        'lr_decay': 0.8,
        'max_epoch':7,
        'max_max_epoch':40,
        'max_grad_norm': 5,
        'num_layers': 1,
        'num_steps': 1000,
        'embed_size': 300,
        'hidden_size':128,
        'keep_prob': 0.75,
        'batch_size': 32,
        'num_classes': 2,
        'vocab_size': 60000,
        'combine_mode': 'last',
        'weight_decay': 5e-6,
        'save_path': 'checkpoint/imdb/'
    }

    data_path = 'data/imdb/train.csv'
    data_processor = DataProcessor(data_path,vocab_size=config['vocab_size'],\
                    seperator=',',max_seq_len=config['num_steps'],header=0,reverse=True)
    data , labels    = data_processor.get_training_data()
    print('Train Data Shape',data.shape,labels.shape)
    embedding = data_processor.get_embedding(config['embed_size'])
    print('Embedding Shape',embedding.shape)
    test_data, test_labels = data_processor.process_test_file( 'data/imdb/test.csv',contains_label=True,header=0)
    trainer = Trainer(config,data,labels,embedding)
    # trainer.X_test = test_data
    # trainer.y_test = test_labels
    trainer.train()
    trainer.load_best_model()
    pred = trainer.predict(test_data)
    print(classification_report(y_true=test_labels,y_pred=pred))
