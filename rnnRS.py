# -*- coding: utf-8 -*-

import tensorflow as tf
from baseRS import BaseRS


class RnnRs(BaseRS):
    
    def __init__(self,config):
        
        super(RnnRs, self).__init__(config)
        self.cell_type = self.config.cell_type
        self.layer_size = self.config.layer_size
        self.layer_activations = self.config.layer_activations
        self.layer_func = [self._get_activation_func(name) for  name in self.layer_activations ]
    
    def _create_placeholders(self):
        
        self.X_seq = tf.placeholder(tf.float32, shape=(None, None, self.dim)) # batch_size, time_steps, output_size
        self.Item = tf.placeholder(tf.float32, shape=(None, self.dim)) # batch_size, output_size
        self.Label = tf.placeholder(tf.float32, shape=(None,1))
        self.Len_seq = tf.placeholder(tf.int64, shape=(None))
    
    def _get_a_cell(self,size,func):
        if self.cell_type == 'rnn':
            return tf.nn.rnn_cell.BasicRNNCell(num_units = size, activation = func)
        elif self.cell_type == 'lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(num_units = size, activation = func) 
        elif self.cell_type == 'gru':
            return tf.nn.rnn_cell.GRUCell(num_units = size, activation = func) 
        else:
            raise ValueError('unknown rnn type. {0}'.format(self.cell_type)) 


    def _create_variables(self):
        with tf.name_scope('variables'):
            self.global_bias = tf.Variable(tf.truncated_normal([1], stddev=self.init_value*0.1, mean=0), dtype=tf.float32, name='glo_b')
    
    def _gather_last_output(self, data, seq_lens):
        '''用来获取rnn输出序列的最后输出结果'''
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype = tf.int64)
        indices = tf.stack([this_range, seq_lens-1], axis=1)
        return tf.gather_nd(data, indices)
    
    def _create_inference(self):
        with tf.name_scope('inference'):
            #创建了一个多层RNN
            rnn_cell = tf.contrib.rnn.MultiRNNcell([self._get_a_cell(size, func) for (size, func) in zip(self.layer_sizes, self.layer_func)])
            output, _ = tf.nn.tf.nn.dynamic_rnn(rnn_cell, self.X_seq, sequence_length = self.Len_seq, dtype=tf.float32 )
            u_t = self._gather_last_output(output, self.Len_seq)
            u_t = tf.reshape(u_t, (-1, self.layer_sizes[-1]), name = 'user_embedding')
            self.output = tf.sigmoid(tf.reduce_sum(tf.multiply(u_t, self.Item), 1, keepdims = True) + global_bias , name= 'prediction')

    def _create_loss(self):
        with tf.name_scope('loss'):
            #需要添加正则项！！！
            self.loss = self._get_loss(self.output, self.Label)
            tf.summary.scalar('loss', self.loss)
    def _create_optimizer(self):
        with tf.name_scope('optimize'):
            self.optimizer = self._optimize(self.loss, None)
    


