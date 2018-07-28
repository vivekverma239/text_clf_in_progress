import tensorflow as tf



def compute_softmax(pre_logits,labels,weight,biases,num_samples,training):
    tf.nn.nce_loss( weights,
                    biases,
                    labels,
                    inputs,
                    num_sampled,
                    num_classes,
                    name='nce_loss'
                )
