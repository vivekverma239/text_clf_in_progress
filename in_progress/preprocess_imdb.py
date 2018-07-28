# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()

# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import DataProcessor
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import DataProcessor
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from encoder import CoVeEncoder

def embed_glove(pretrained_embedding):
    _input         = tf.placeholder(dtype=tf.int32,shape=[None,config['num_steps']],name='input')
    with tf.device("/cpu:0"):
        embedding = tf.Variable(pretrained_embedding,dtype=tf.float32,name='embedding')
    output = tf.nn.embedding_lookup(embedding, _input)
    return _input, output


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def write_tfrecord(inputs,outputs,writer):
    for idx in range(len(inputs)):

        # Put in the original images into array
        # Just for future check for correctness


        example = tf.train.Example(features=tf.train.Features(feature={
            'input': _float_feature(inputs[idx].reshape([-1])),
            'output':_int_feature([outputs[idx]])
            }))

        writer.write(example.SerializeToString())



class Trainer(object):
    def __init__(self,config,X,y,embedding,writer):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                data, labels, test_size=0.5, random_state=12,shuffle=True)
        self.writer = writer
        self.keep_prob = config['keep_prob']
        self.batch_size = config['batch_size']
        self.temp_input, self.temp_output = embed_glove(embedding)
        self.model = CoVeEncoder(self.temp_output)
        # config_temp = tf.ConfigProto(
        # device_count = {'GPU': 0}
        # )
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.model.load_weights(self.sess)
        self.saver = tf.train.Saver()
        self.num_epochs = 200
        self.epoch = 0
        self.lr = config['lr']
        self.save_path = config['save_path']
        self.lr_decay = config['lr_decay']
        self.max_epoch= config['max_epoch']
        self.max_max_epoch= config['max_max_epoch']
        self.run_epoch(X,y)



    def get_cove_vectors(self,X):
        return self.cove_model.predict(X)




    def run_epoch(self,X,y, training=False):

        for i in range(int(len(X)/self.batch_size)+1):
            X_sub = X[i*self.batch_size: (i+1)*self.batch_size,:]
            y_sub = y[i*self.batch_size: (i+1)*self.batch_size]
            X_temp = self.sess.run(self.model.output,{self.temp_input:X_sub})
            # X_sub = self.get_cove_vectors(X_temp)
            write_tfrecord(X_temp,y_sub,self.writer)



if __name__ == '__main__':

    config = {
        'init_scale': 0.1,
        'lr': .001,
        'lr_decay': 0.8,
        'max_epoch':7,
        'max_max_epoch':40,
        'max_grad_norm': 5,
        'num_layers': 1,
        'num_steps': 50,
        'embed_size': 600,
        'hidden_size':600,
        'keep_prob': 0.7,
        'batch_size': 16,
        'num_classes': 2,
        'vocab_size': 40000,
        'combine_mode': 'last',
        'weight_decay': 3e-6,
        'save_path': 'checkpoint/cove/'
    }
    tfrecords_filename = 'imdb_cove_train.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    tfrecords_filename_test = 'imdb_cove_test.tfrecords'
    writer_test = tf.python_io.TFRecordWriter(tfrecords_filename_test)

    data_path = 'data/imdb/train.csv'
    data_processor = DataProcessor(data_path,vocab_size=config['vocab_size'],\
                                   seperator=',',max_seq_len=config['num_steps'],\
                                   header=0,reverse=True)
    data , labels    = data_processor.get_training_data()
    print('Train Data Shape',data.shape,labels.shape)
    embedding = data_processor.get_embedding(300)
    print('Embedding Shape',embedding.shape)
    test_data, test_labels = data_processor.process_test_file( 'data/imdb/test.csv',contains_label=True,header=0)
    trainer = Trainer(config,data,labels,embedding,writer=writer)
    tf.reset_default_graph()
    trainer = Trainer(config,test_data,test_labels,embedding,writer=writer_test)
    # trainer.X_test = test_data
    # trainer.y_test = test_labels
    # trainer.train()
    # trainer.load_best_model()
    # pred = trainer.predict(test_data)
    # print(classification_report(y_true=test_labels,y_pred=pred))

    writer.close()
    writer_test.close()
