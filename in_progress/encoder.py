import tensorflow as tf
import tensorflow_hub as hub
import tempfile
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import TimeDistributed, Dense, LSTM, Dropout, Multiply,\
            Concatenate, Bidirectional

# class Encoder(object):
#     def __init__(self):
#         raise NotImplementedError
#
#     def encode(input,type):
#         '''
#         Input: list of input to encode
#         type: 'string' or 'token'
#         '''
#


class LSTMEncoderWithEmbedding(object):
    def __init__(self,input,embed_size,hidden_size,vocab_size,num_steps,\
                 keep_prob,embedding=None,\
                 num_layers=1,variational_dropout=True,combine_mode='last'):
        self._input         = input
        self.embed_size     = embed_size
        self.size           = hidden_size
        self.keep_prob      = keep_prob
        self.combine_mode   = combine_mode
        self.num_layers     = num_layers
        self.num_steps = num_steps
        self.vocab_size = vocab_size

        if embedding is not None:
            with tf.device("/cpu:0"):
                embedding = tf.Variable(embedding,dtype=tf.float32,name='embedding')
        else:
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [self.vocab_size+1, self.embed_size], dtype=tf.float32)

        inputs = tf.nn.embedding_lookup(embedding, self._input)
        inputs = tf.nn.dropout(inputs, self.keep_prob,noise_shape=[tf.shape(self._input)[0],1,self.embed_size])

        def lstm_cell(input_size):
          return tf.contrib.rnn.LSTMCell(
              input_size, forget_bias=0.0, state_is_tuple=True,
              reuse=tf.get_variable_scope().reuse)
        def attn_cell(input_size):
          return tf.contrib.rnn.DropoutWrapper(
              lstm_cell(input_size), output_keep_prob=keep_prob,variational_recurrent=True,dtype=tf.float32)

        cell = tf.contrib.rnn.MultiRNNCell( [attn_cell(self.size ) for i in range(self.num_layers)])

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
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        elif  self.combine_mode == 'last':
          outputs = outputs[-1]
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        elif  self.combine_mode == 'attention':
          attention_w = tf.get_variable('attention_w',shape=[self.size,1],dtype=tf.float32)
          outputs = tf.stack(outputs,axis=1)
          outputs = tf.reshape(outputs,[-1,self.size])
          output_weights = tf.nn.softmax(tf.matmul(outputs , attention_w))
          outputs = tf.multiply(outputs,output_weights)
          outputs = tf.reshape(outputs,[-1,self.num_steps,self.size])
          outputs = tf.reduce_sum(outputs,axis=1)
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        else:
          outputs = tf.stack(outputs,axis=1)
          outputs = tf.reduce_mean(outputs,axis=1)
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)

    def get_output(self):
        return self.outputs

class LSTMEncoder(object):
    def __init__(self,input,embed_size,hidden_size,vocab_size,num_steps,\
                 keep_prob,\
                 num_layers=1,variational_dropout=True,combine_mode='last'):
        self._input         = input
        self.embed_size     = embed_size
        self.size           = hidden_size
        self.keep_prob      = keep_prob
        self.combine_mode   = combine_mode
        self.num_layers     = num_layers
        self.num_steps = num_steps
        self.vocab_size = vocab_size

        inputs = self._input
        inputs = tf.nn.dropout(inputs, self.keep_prob,noise_shape=[tf.shape(self._input)[0],1,self.embed_size])

        def lstm_cell(input_size):
          return tf.contrib.rnn.LSTMCell(
              input_size, forget_bias=0.0, state_is_tuple=True,
              reuse=tf.get_variable_scope().reuse)
        def attn_cell(input_size):
          return tf.contrib.rnn.DropoutWrapper(
              lstm_cell(input_size), output_keep_prob=keep_prob,variational_recurrent=True,dtype=tf.float32)

        cell = tf.contrib.rnn.MultiRNNCell( [attn_cell(self.embed_size ) for i in range(self.num_layers)])

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
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        elif  self.combine_mode == 'last':
          outputs = outputs[-1]
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        elif  self.combine_mode == 'attention':
          attention_w = tf.get_variable('attention_w',shape=[self.size,1],dtype=tf.float32)
          outputs = tf.stack(outputs,axis=1)
          outputs = tf.reshape(outputs,[-1,self.size])
          output_weights = tf.nn.softmax(tf.matmul(outputs , attention_w))
          outputs = tf.multiply(outputs,output_weights)
          outputs = tf.reshape(outputs,[-1,self.num_steps,self.size])
          outputs = tf.reduce_sum(outputs,axis=1)
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        else:
          outputs = tf.stack(outputs,axis=1)
        #   outputs = tf.reduce_mean(outputs,axis=1)
          self.outputs = outputs

    def get_output(self):
        return self.outputs



class BiattentiveClassification(object):
    def __init__(self,input,embed_size,hidden_size,vocab_size,num_steps,\
                 keep_prob,\
                 num_layers=1,variational_dropout=True,combine_mode='last'):
        self._input         = input
        self.embed_size     = embed_size
        self.size           = hidden_size
        self.keep_prob      = keep_prob
        self.combine_mode   = combine_mode
        self.num_layers     = num_layers
        self.num_steps = num_steps
        self.vocab_size = vocab_size

        inputs = self._input
        inputs = tf.nn.dropout(inputs, self.keep_prob,noise_shape=[tf.shape(self._input)[0],1,self.embed_size])

        def lstm_cell(input_size):
          return tf.contrib.rnn.LSTMCell(
              input_size, forget_bias=0.0, state_is_tuple=True,
              reuse=tf.get_variable_scope().reuse)
        def attn_cell(input_size):
          return tf.contrib.rnn.DropoutWrapper(
              lstm_cell(input_size), output_keep_prob=keep_prob,variational_recurrent=True,dtype=tf.float32)

        cell = tf.contrib.rnn.MultiRNNCell( [attn_cell(self.embed_size ) for i in range(self.num_layers)])

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
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        elif  self.combine_mode == 'last':
          outputs = outputs[-1]
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        elif  self.combine_mode == 'attention':
          attention_w = tf.get_variable('attention_w',shape=[self.size,1],dtype=tf.float32)
          outputs = tf.stack(outputs,axis=1)
          outputs = tf.reshape(outputs,[-1,self.size])
          output_weights = tf.nn.softmax(tf.matmul(outputs , attention_w))
          outputs = tf.multiply(outputs,output_weights)
          outputs = tf.reshape(outputs,[-1,self.num_steps,self.size])
          outputs = tf.reduce_sum(outputs,axis=1)
          self.outputs = tf.nn.dropout(outputs,self.keep_prob)
        else:
          outputs = tf.stack(outputs,axis=1)
        #   outputs = tf.reduce_mean(outputs,axis=1)
          self.outputs = outputs

    def get_output(self):
        return self.outputs

class CNNEncoderWithEmbedding(object):
    def __init__(self,input,embed_size,hidden_size,vocab_size,num_steps,\
                 keep_prob,embedding=None,\
                 num_layers=1,variational_dropout=True,combine_mode='last'):
        self._input         = input
        self.embed_size     = embed_size
        self.size           = hidden_size
        self.keep_prob      = keep_prob
        self.combine_mode   = combine_mode
        self.num_layers     = num_layers
        self.num_steps = num_steps
        self.vocab_size = vocab_size

        if embedding is not None:
            with tf.device("/cpu:0"):
                embedding = tf.Variable(embedding,dtype=tf.float32,name='embedding')
        else:
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [self.vocab_size, self.embed_size], dtype=tf.float32,trainable=True)

        inputs = tf.nn.embedding_lookup(embedding, self._input)
        inputs = tf.nn.dropout(inputs, self.keep_prob,noise_shape=[tf.shape(self._input)[0],1,self.embed_size])


        outputs = tf.layers.Conv1D(filters=128,kernel_size=3,strides=1,\
                                padding='same',activation=tf.nn.relu)(inputs)
        outputs = tf.layers.MaxPooling1D(pool_size=5,strides=1)(outputs)
        outputs = tf.nn.dropout(outputs,keep_prob=self.keep_prob)
        # outputs = tf.layers.Conv1D(filters=32,kernel_size=3,strides=1,\
        #                         padding='same',activation=tf.nn.relu)(outputs)
        # outputs = tf.layers.MaxPooling1D(pool_size=5,strides=3)(outputs)
        # outputs = tf.nn.dropout(outputs,keep_prob=self.keep_prob)
        # outputs = tf.layers.Conv1D(filters=256,kernel_size=7,strides=1,\
        #                         padding='same',activation=tf.nn.relu)(outputs)
        # outputs = tf.layers.MaxPooling1D(pool_size=5,strides=2)(outputs)
        # outputs = tf.nn.dropout(outputs,keep_prob=self.keep_prob)
        outputs = tf.layers.MaxPooling1D(pool_size=outputs.get_shape().as_list()[1],strides=outputs.get_shape().as_list()[1])(outputs)
        self.outputs = tf.layers.flatten(outputs)


    def get_output(self):
        return self.outputs


TEXT_ENCODER_MODULES={'google/nnlm-en-dim50/1':\
									"https://tfhub.dev/google/nnlm-en-dim50/1",
					  'google/nnlm-en-dim128/1':\
									"https://tfhub.dev/google/nnlm-en-dim128/1",
					  'google/nnlm-en-dim50-with-normalization/1':\
									"https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1",
					  'google/nnlm-en-dim128-with-normalization/1':\
									"https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1",
					  'google/random-nnlm-en-dim128/1':\
					                "https://tfhub.dev/google/random-nnlm-en-dim128/1",
					  'google/universal-sentence-encoder/2':\
					                "https://tfhub.dev/google/universal-sentence-encoder/2",
					  'google/universal-sentence-encoder-large/2':\
					                "https://tfhub.dev/google/universal-sentence-encoder-large/2",
					  'google/elmo/2':\
					                "https://tfhub.dev/google/elmo/1"}


class TFHubEncoder(object):
    def __init__(self,input,name='google/elmo/2'):
        self._input = input
        with tf.device('/cpu:0'):
            self.embed = self._get_hub_embedding(name)
            self.embedded_input = self.embed(self._input)

    def _get_hub_embedding(self,name):
    	  embed = hub.Module(TEXT_ENCODER_MODULES[name])
    	  return embed

    def get_embedded_output(self):
        return self.embedded_input


class CoVeEncoder():

    def __init__(self,input):
        self.input = input
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope('CoVe'):
            with tf.Session() as sess:
                # with tf.device("/cpu:0"):
                with tf.variable_scope('model'):
                    K = tf.keras.backend
                    x = tf.keras.layers.Input(tensor= self.input, shape=(None,300))
                    y = tf.keras.layers.Masking(mask_value=0.,input_shape=(None,300))(x)
                    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True, recurrent_activation='sigmoid', name='lstm1'),name='bidir_1')(y)
                    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True, recurrent_activation='sigmoid', name='lstm2'),name='bidir_2')(y)

                    # These 2 layer are short-cut fix for the issue -
                    y_rev_mask_fix = tf.keras.layers.Lambda(lambda x: K.cast(K.any(K.not_equal(x, 0.), axis=-1, keepdims=True), K.floatx()))(x)
                    y = tf.keras.layers.Multiply()([y,y_rev_mask_fix])
                    self.output = y
                    self.cove = tf.keras.Model(inputs=x,outputs=y)
                    self.cove.load_weights('saved_models/CoVe/Keras_CoVe.h5')


                self.cove_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CoVe/model')

                self.tf_checkpoint_path = tf.train.Saver(self.cove_weights).save(sess, 'saved_models/CoVe/cove_model')
                print('Checkpoint Saved')

        self.model_weights_tensors = set(self.cove_weights)


    def load_weights(self,sess):
        # self.cove.load_weights('saved_models/CoVe/Keras_CoVe.h5')
        # sess = tf.get_default_session()
        tf.train.Saver(self.cove_weights).restore(sess, 'saved_models/CoVe/cove_model')

    # @property
    # def output(self):
    #     return self.output
    #
    # @property
    # def input(self):
    #     return self.input

    def set_input(self,input):
        self.cove.input = input
    def __getitem__(self, key):
        return self.outputs[key]



class seqCNN(object):
    """
    This CNN is an atempt to recreate the CNN described in this paper:
    Effective Use of Word Order for Text Categorization with Convolutional Neural Networks
    Rie Johnson, Tong Zhang
    """

    def __init__(self,num_classes, num_filters, num_pooled, vocabulary_length, region_size, max_sentence_length):
        #input layers and params
        filter_length = vocabulary_length*region_size
        max_sentence_length = max_sentence_length


        self.x_input = tf.placeholder(tf.float32, [None, sentence_length], name="x_input")
        processed_input = tf.one_hot(self.x_input,vocabulary_length)
        sentence_length = max_sentence_length*vocabulary_length
        self.processed_input = tf.reshape(processed_input,[None, sentence_length, 1, 1])
        self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")


        self.dropout_param = tf.placeholder(tf.float32, name="dropout_param")

        cnn_filter_shape = [filter_length, 1, 1, num_filters]
        W_CN = tf.Variable(tf.truncated_normal(cnn_filter_shape, stddev=0.1), name="W_CN")
        b_CN = tf.Variable(tf.truncated_normal([num_filters],stddev=0.1),name="b_CN")

        #conv-relu-pool layer
        conv = tf.nn.conv2d(
                        self.processed_input,
                        W_CN,
                        strides=[1, vocabulary_length, 1, 1],
                        padding="VALID",
                        name="conv"
                        )
        relu = tf.nn.relu(
                        tf.nn.bias_add(conv, b_CN),
                        name="relu"
                        )

        print('relu',relu.get_shape())
        pool_stride = [1,int((max_sentence_length-region_size+1)/num_pooled),1,1]
        pool = tf.nn.avg_pool(
                        relu,
                        ksize = pool_stride,
                        strides = pool_stride,
                        padding="VALID",
                        name="pool"
                        )
        print('pool',pool.get_shape())

        #dropout
        drop = tf.nn.dropout(
                        pool,
                        self.dropout_param
                        )

        #response normalization
        normalized = tf.nn.local_response_normalization(drop)

        #feature extraction and flatting for future
        self.pool_flat = tf.reshape(normalized, [-1, num_pooled*num_filters])

        #affine layer
        affine_filter_shape = [num_pooled*num_filters, num_classes]
        W_AF = tf.Variable(
                        tf.truncated_normal(affine_filter_shape, stddev=0.1),
                        name="W_AF"
                        )
        b_AF = tf.Variable(
                        tf.truncated_normal([num_classes], stddev=0.1),
                        name="b_AF"
                        )
        self.logits = tf.matmul(self.pool_flat,W_AF)+b_AF



class BCNEncoder():
    def __init__(self,input,size):
        self.size = size
        self.input = input
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope('BCN'):
            K = tf.keras.backend
            hidden_size = self.size
            encoder = Sequential()
            encoder.add(Dropout(0.1,input_shape=self.input.get_shape().as_list()[1:]))
            encoder.add(TimeDistributed(Dense(hidden_size,activation='relu',input_shape=self.input.get_shape().as_list()[1:])))

            # Afterwards, we do automatic shape inference:
            encoder.add(Bidirectional(LSTM(hidden_size,return_sequences=True)))

            left = encoder(self.input)
            right = encoder(self.input)
            biattention =tf.matmul(left,tf.transpose(right,perm=[0,2,1]))
            Ax = tf.nn.softmax(biattention,axis=-1)
            Ay = tf.nn.softmax(biattention,axis=1)
            Cx = tf.transpose(tf.matmul(tf.transpose(left,[0,2,1]),Ax),[0,2,1])
            Cy = tf.transpose( tf.matmul(tf.transpose(right,[0,2,1]),Ay),[0,2,1])
            XminusC = left - Cx
            YminusC = right - Cy
            XdotC  = tf.multiply(left,Cx)
            YdotC  = tf.multiply(right,Cy)
            Xc = tf.concat([left,XminusC,XdotC],axis=-1)
            Yc = tf.concat([right,YminusC,YdotC],axis=-1)

            integrate =  Sequential()
            # encoder.add(Dropout(0.2,input_shape=Xc.get_shape().as_list()[1:]))
            encoder.add(Bidirectional(LSTM(hidden_size,return_sequences=True)))

            def self_attention(hidden,scope):
                with tf.variable_scope(scope) as scope:
                    size = hidden.get_shape().as_list()
                    attention_w = tf.get_variable('attention_w',shape=[size[-1],1],dtype=tf.float32)
                    outputs = tf.reshape(hidden,[-1,size[-1]])
                    output_weights = tf.nn.softmax(tf.matmul(outputs , attention_w))
                    outputs = tf.multiply(outputs,output_weights)
                    outputs = tf.reshape(outputs,[-1,size[1],size[2]])
                    outputs = tf.reduce_sum(outputs,axis=1)
                    # outputs = tf.nn.dropout(outputs,self.keep_prob)
                return outputs

            def pool_layer(hidden,scope):
                #Min Pool Across Time
                min_  = tf.reduce_min(hidden,axis=1)
                max_  = tf.reduce_max(hidden,axis=1)
                mean_  = tf.reduce_mean(hidden,axis=1)
                # attn_ = self_attention(hidden,scope)
                return tf.concat([min_,max_,mean_],axis=-1)

            X_c = integrate(Xc)
            Y_c = integrate(Yc)

            pooled_x = pool_layer(X_c,'pool_x')
            pooled_y = pool_layer(Y_c,'pool_y')

            pooled = tf.concat([pooled_x,pooled_y],axis=-1)
            pooled = tf.contrib.layers.maxout(pooled,num_units=hidden_size)
            pooled = tf.contrib.layers.maxout(pooled,num_units=hidden_size)
            # pooled = tf.contrib.layers.maxout(pooled,num_units=hidden_size)
            self.outputs = pooled


    def get_output(self):
        return self.outputs
