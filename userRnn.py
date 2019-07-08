# -*- coding: utf-8 -*-

import tensorflow as tf
from baseRS import BaseRS
import time
import numpy as np
import math
from  sklearn.metrics import roc_auc_score 



class UserRNN(BaseRS):
    
    def __init__(self,config, dl):
        
        super().__init__(config, dl)
        self.cell_type = self.config.cell_type
        self.layer_sizes = self.config.layer_sizes
        self.layer_activations = self.config.layer_activations
        self.init_value = self.config.init_value
        self.layer_func = [self._get_activation_func(name) for  name in self.layer_activations ]
        self._init_graph()
    def _create_placeholders(self):

        self.User = tf.placeholder(tf.int64, shape=(None,))
        self.X_seq = tf.placeholder(tf.int64, shape=(None, None)) # batch_size, time_steps, output_size
        self.Item = tf.placeholder(tf.int64, shape=(None,)) # batch_size, output_size
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
            self.biases = tf.Variable(tf.truncated_normal([self.dl.num_items], stddev=self.init_value*0.1, mean=0), dtype=tf.float32, name='biases')
            self.embedding_U = tf.Variable(tf.truncated_normal(shape=[self.dl.num_users, self.config.embedding_size], mean=0.0, stddev=0.01),\
                            name='embedding_P', dtype = tf.float32)
            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.dl.num_items, self.config.embedding_size], mean=0.0, stddev=0.01),\
                            name='c1', dtype = tf.float32)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.config.embedding_size], name='c2')
            self.embedding_P = tf.concat([self.c1, self.c2], 0 , name='emebedding_P') 
            
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.dl.num_items, self.config.embedding_size], mean=0.0, stddev=0.01),\
                            name='embedding_Q', dtype = tf.float32)
    
    def _gather_last_output(self, data, seq_lens):
        '''用来获取rnn输出序列的最后输出结果'''
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype = tf.int64)
        indices = tf.stack([this_range, seq_lens-1], axis=1)
        return tf.gather_nd(data, indices)
    
    def _create_inference(self):
        with tf.name_scope('inference'):
            #创建了一个多层RNN
            self.X_seq_embedding = tf.nn.embedding_lookup(self.embedding_P, self.X_seq)
            self.bias = tf.nn.embedding_lookup(self.biases, self.Item)
            self.Item_embedding = tf.nn.embedding_lookup(self.embedding_Q, self.Item)
            self.user_embedding = tf.nn.embedding_lookup(self.embedding_U, self.User)
            self.rnn_cell = tf.contrib.rnn.MultiRNNCell([self._get_a_cell(size, func) for (size, func) in zip(self.layer_sizes, self.layer_func)])
            output, _ = tf.nn.dynamic_rnn(self.rnn_cell, self.X_seq_embedding, sequence_length = self.Len_seq, dtype=tf.float32 )#, initial_state= (self.user_embedding,))
            u_t = self._gather_last_output(output, self.Len_seq)
            u_t = tf.reshape(u_t, (-1, self.layer_sizes[-1]), name = 'user_embedding')#+ self.user_embedding
            self.bias = tf.expand_dims(self.bias, 1)
            self.output = tf.sigmoid(tf.reduce_sum(tf.multiply(u_t, self.Item_embedding), 1, keepdims = True) + self.bias , name= 'prediction')
            #self.output = tf.sigmoid(tf.layers.dense(tf.multiply(u_t, self.Item_embedding), 1) + self.bias , name= 'prediction')
            tf.summary.histogram('predictions', self.output)
            tf.summary.histogram('user_embedding', self.user_embedding)
    def _create_loss(self):
        with tf.name_scope('loss'):
            #需要添加正则项！！！
            self.error = self._get_loss(self.output, self.Label)
            self.loss = self.error + self.config.regI1*(tf.reduce_sum(tf.square(self.embedding_Q)))+self.config.regI2*(tf.reduce_sum(tf.square(self.embedding_P)))
            self.loss += self.config.reg_bias*(tf.reduce_sum(tf.square(self.biases)))+self.config.regU*(tf.reduce_sum(tf.square(self.embedding_U)))

            tf.summary.scalar('error', self.error)
            tf.summary.scalar('loss', self.loss)

            
    def _create_optimizer(self):
        with tf.name_scope('optimize'):
            starter_learning_rate = self.config.lr
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
            self._glo_ite_counter, 100000, 0.96, staircase=True)

            self.optimizer= tf.train.AdamOptimizer(learning_rate)
            gradients = self.optimizer.compute_gradients(self.loss, var_list = tf.trainable_variables())
            gradients, v = zip(*gradients)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            self.train_step = self.optimizer.apply_gradients(zip(gradients,v))

            for g,v in zip(gradients,v):
                tf.summary.histogram('grad/{}'.format(v.name), g)
                tf.summary.histogram('grad/sparse/{}'.format(v.name), tf.nn.zero_fraction(g))

            self.summary = tf.summary.merge_all()
    def fit(self, user_history, target_items, labels, user_history_lens, user_data):
        error, loss, summary, _ = self.sess.run(
                  [self.error, self.loss, self.summary, self.train_step], 
                  {self.X_seq: user_history, 
                   self.Item: target_items,
                   self.Label: labels,
                   self.Len_seq: user_history_lens,
                   self.User : user_data
                   }
        )
        self.log_writer.add_summary(summary, self._glo_ite_counter)
        self._glo_ite_counter+=1
        return (error, loss)
    def evaluate(self, user_history, target_items, labels, user_history_lens, user_data):
        error, loss, preds  = self.sess.run(
                  [self.error, self.loss, self.output], 
                  {self.X_seq: user_history, 
                   self.Item: target_items,
                   self.Label: labels,
                   self.Len_seq: user_history_lens,
                   self.User : user_data
                   }

        )
        predictions = preds.flatten()
        labels = labels.flatten()
        auc = roc_auc_score( labels,predictions)

        predictions = predictions > 0.5
        precision = sum(predictions*labels)/sum(predictions)
        return (error, loss, precision, auc)

    def train_and_evaluate(self):
        self.config.print_info()
        for epoch_count in range(self.config.epoches):
                #train
                train_begin = time.time()
                train_loss = 0.0
                train_error = 0.0
                batch_i = 0
                for data in self.dl.getTrainBatches():

                    input_data = np.array(data[1])

                    item_data = np.array(data[2])
                    len_data = np.array(data[4])
                    label_data = np.array(data[3])[:, np.newaxis]
                    user_data = np.array(data[0])

                    error, loss = self.fit(input_data, item_data, label_data, len_data, user_data)
                    train_loss+=loss
                    train_error += error
                    batch_i+=1
                train_loss /= batch_i
                train_error /= batch_i
                train_time = time.time() - train_begin
                if epoch_count % self.config.verbose == 0:

     
                    eval_begin = time.time() 
                    hits,  aucs = [], []
                    test_loss = 0.0
                    test_error = 0.0
                    batch_i = 0
                    for data in self.dl.getTestBatches():
                        input_data = np.array(data[1])

                        item_data = np.array(data[2])
                        len_data = np.array(data[4])
                        label_data = np.array(data[3])[:, np.newaxis]
                        user_data = np.array(data[0])
                        error, loss, hr,  auc = self.evaluate(input_data, item_data, label_data, len_data, user_data)
                        test_loss+=loss
                        test_error += error
                        batch_i+=1
                        hits.append(hr)
                        aucs.append(auc)
                    test_loss /= batch_i
                    test_error /= batch_i
                    
                    hr, auc = np.array(hits).mean(), np.array(aucs).mean()
                    eval_time = time.time() - eval_begin
        #             print("Epoch %d [%.1fs ]:  train_loss = %.4f" % (
        #                     epoch_count,train_time, train_loss))    
                    print("Epoch %d [ %.1fs]: precision = %.4f,  AUC=%.4f, loss = %.4f ,error = %.4f [%.1fs] train_loss = %.4f ,train_error = %.4f [%.1fs]" % (
                                epoch_count, train_time, hr,  auc, test_loss, test_error, eval_time, train_loss, train_error, train_time))
                    # print("Epoch %d [ %.1fs]: hr = %.4f, ndcg = %.4f, AUC=%.4f, loss = %.4f ,error = %.4f [%.1fs] train_loss = %.4f ,train_error = %.4f [%.1fs]" % (
                    #             epoch_count, train_time, hr, ndcg, auc, test_loss, test_error, eval_time, train_loss, train_error, train_time))

    