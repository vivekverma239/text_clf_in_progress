'''
Code to classify sentences into question type.

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
from utils import DataProcessor
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import DataProcessor
from sklearn.metrics import classification_report
import numpy as np


class LSTMModel(object):

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
    with tf.device("/cpu:0"):
      embedding = tf.Variable(pretrained_embedding,dtype=tf.float32)
      inputs = tf.nn.embedding_lookup(embedding, self._input)

    inputs = tf.nn.dropout(inputs, self.keep_prob,noise_shape=[tf.shape(self._input)[0],1,self.embed_size])

    def lstm_cell(input_size):
        return tf.contrib.rnn.LSTMCell(
            input_size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
    def attn_cell(input_size):
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(input_size), output_keep_prob=config['keep_prob'],variational_recurrent=True,dtype=tf.float32)
    cell = tf.contrib.rnn.MultiRNNCell( [attn_cell(self.embed_size ) for i in range(config['num_layers'])])

    self._initial_state = cell.zero_state(tf.shape(self._input)[0], tf.float32)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    if self.combine_mode == 'mean':
        outputs = tf.stack(outputs,axis=1)
        outputs = tf.reduce_mean(outputs,axis=1)
        outputs = tf.nn.dropout(outputs,self.keep_prob)
    elif  self.combine_mode == 'last':
        outputs = outputs[-1]
        outputs = tf.dropout(outputs,self.keep_prob)
        outputs = tf.nn.dropout(outputs,self.keep_prob)
    else:
        outputs = tf.stack(outputs,axis=1)
        outputs = tf.reduce_mean(outputs,axis=1)
        outputs = tf.nn.dropout(outputs,self.keep_prob)

    softmax_w = tf.get_variable("softmax_w", [self.size, self.num_classes], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [self.num_classes], dtype=tf.float32)
    logits = tf.matmul(outputs, softmax_w) + softmax_b


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


class TrainingHelper(object):
    def __init__(self,config,X,y,embedding):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                data, labels, test_size=0.25, random_state=12)

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
        self.lr_decay = config['lr_decay']
        self.max_epoch= config['max_epoch']
        self.max_max_epoch= config['max_max_epoch']
        self.save_path = config['save_path']

    def train(self):
        cost_history = []
        best_cost = 10000000
        early_stopping_epochs = 20
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
                self.save_model(self.save_path+'model_best.ckpt')
            if len(cost_history) > early_stopping_epochs \
                            and min(cost_history[-early_stopping_epochs:]) != best_cost:
                print("Loss hasn't improved in {} epochs, stopping early".format(early_stopping_epochs) )
                break

    def evaluate(self,best_model=False):
        if best_model:
            self.load_model(self.save_path+'model_best.ckpt')
        y_pred = self.predict(self.X_test)
        print(classification_report(self.y_test,y_pred))

    def predict(self,X):
        temp = []
        feed_dict  = {self.model.keep_prob: 1}
        ops        = [self.model.predict]
        for i in range(int(len(X)/self.batch_size)+1):
            X_sub = X[i*self.batch_size: (i+1)*self.batch_size,:]
            feed_dict[self.model.input] = X_sub
            ops_out = self.sess.run(ops,feed_dict)
            temp.append(ops_out[0])
        predictions = np.concatenate(temp,axis=0)
        return predictions


    def run_epoch(self,X,y, training=True):
        temp_cost  = []
        temp_acc   = []
        feed_dict  = {self.model.keep_prob: 1}
        ops        = [self.model.cost, self.model.predict]
        if training:
            feed_dict[self.model.keep_prob] = self.keep_prob
            ops = [self.train_op,self.model.cost, self.model.predict]
        for i in range(int(len(X)/self.batch_size)-1):
            X_sub = X[i*self.batch_size: (i+1)*self.batch_size,:]
            y_sub = y[i*self.batch_size: (i+1)*self.batch_size]
            feed_dict[self.model.input] = X_sub
            feed_dict[self.model.target] = y_sub
            ops_out = self.sess.run(ops,feed_dict)
            temp_cost.append(ops_out[-2])
            temp_acc.append( np.mean(ops_out[-1] == y_sub))
        cost = np.mean(temp_cost)
        accuracy = np.mean(temp_acc)
        return cost, accuracy

    def save_model(self,model_name):
        self.saver.save(self.sess,model_name)

    def load_model(self,model_name):
        self.saver.restore(self.sess,model_name)


if __name__ == '__main__':

    config = {
        'init_scale': 0.1,
        'lr': .001,
        'lr_decay': 0.95,
        'max_epoch':5,
        'max_max_epoch':200,
        'max_grad_norm': 10,
        'num_layers': 1,
        'num_steps': 400,
        'embed_size': 50,
        'hidden_size': 50,
        'keep_prob': 0.75,
        'batch_size': 32,
        'num_classes': 5,
        'vocab_size': 10000,
        'combine_mode': 'mean',
        'weight_decay': 2e-6,
        'save_path': './saved_models/lstm/'
    }

    data_path = 'data/custom/LabelledData.txt'
    data_processor = DataProcessor(data_path,seperator=',,,',max_seq_len=config['num_steps'])
    data , labels    = data_processor.get_training_data()
    embedding = data_processor.get_embedding(config['embed_size'])
    trainer = TrainingHelper(config,data,labels,embedding)
    trainer.train()
    print('Training Complete.')
    print('Evaluating on best Model.')
    trainer.evaluate(best_model=True)
