# -*- coding: utf-8 -*-

import tensorflow as tf
from baseRS import BaseRS


class RnnRs(BaseRS):
    
    def __init__(self,config, dl):
        
        super(RnnRs, self).__init__(config)
        self.dl = dl
        self.cell_type = self.config.cell_type
        self.layer_size = self.config.layer_size
        self.layer_activations = self.config.layer_activations
        self.layer_func = [self._get_activation_func(name) for  name in self.layer_activations ]
    
    def _create_placeholders(self):
        
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
            self.biases = tf.Variable(tf.truncated_normal([1], stddev=self.init_value*0.1, mean=0), dtype=tf.float32, name='biases')
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.dl.num_items, self.config.embedding_size], mean=0.0, stddev=0.01),\
                            name='embedding_P', dtype = tf.float32)
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
            rnn_cell = tf.contrib.rnn.MultiRNNcell([self._get_a_cell(size, func) for (size, func) in zip(self.layer_sizes, self.layer_func)])
            output, _ = tf.nn.tf.nn.dynamic_rnn(rnn_cell, self.X_seq_embedding, sequence_length = self.Len_seq, dtype=tf.float32 )
            u_t = self._gather_last_output(output, self.Len_seq)
            u_t = tf.reshape(u_t, (-1, self.layer_sizes[-1]), name = 'user_embedding')
            self.output = tf.sigmoid(tf.reduce_sum(tf.multiply(u_t, self.Item_embedding), 1, keepdims = True) + self.bias , name= 'prediction')

    def _create_loss(self):
        with tf.name_scope('loss'):
            #需要添加正则项！！！
            self.error = self._get_loss(self.output, self.Label)
            self.loss = self.error + self.config.regI1*(tf.reduce_sum(tf.square(self.embedding_Q)))+self.config.regI2*(tf.reduce_sum(tf.square(self.embedding_P)))
            self.loss += self.reg_bias*(tf.reduce_sum(tf.square(self.biases)))

            tf.summary.scalar('error', self.error)
            tf.summary.scalar('loss', self.loss)

            self.summary = tf.summary.merge_all()
    def _create_optimizer(self):
        with tf.name_scope('optimize'):
            self.train_step = self._optimize(self.loss, None)

    def fit(self, user_history, target_items, labels, user_history_lens):
        error, loss, summary, _ = self.sess.run(
                  [self.error, self.loss, self.summary, self.train_step], 
                  {self.X_seq: user_history, 
                   self.Item: target_items,
                   self.Label: labels,
                   self.Len_seq: user_history_lens
                   }
        )
        self.log_writer.add_summary(summary, self._glo_ite_counter)
        self._glo_ite_counter+=1
        return (error, loss)
    def evaluate(self, user_history, target_items, labels, user_history_lens):
        error, loss, preds  = self.sess.run(
                  [self.error, self.loss, self.predictions], 
                  {self.X_seq: user_history, 
                   self.Item: target_items,
                   self.Label: labels,
                   self.Len_seq: user_history_lens
                   }

        )
        predictions = preds.flatten()
        neg_predict, pos_predict = predictions[:-1], predictions[-1]
        position = (neg_predict >= pos_predict).sum()
        #print(position)
        hr = position < 10
        ndcg = math.log(2) / math.log(position+2) if hr else 0
        return (error, loss, hr, ndcg)

    def train_and_evaluate(self):

        for epoch_count in range(self.config.epoches):
                #train
                train_begin = time.time()
                train_loss = 0.0
                train_error = 0.0
                batch_i = 0
                for data in self.dl.getTrainBatches():
                    input_data = np.array(data[0])
                    item_data = np.array(data[1])
                    len_data = np.array(data[2])
                    label_data = np.array(data[3])
            
                    error, loss = self.fit(input_data, item_data, label_data, len_data)
                    train_loss+=loss
                    train_error += error
                    batch_i+=1
                train_loss /= batch_i
                train_error /= batch_i
                train_time = time.time() - train_begin
                if epoch_count % self.config.verbose_count == 0:
     
                    eval_begin = time.time() 
                    hits, ndcgs, losses = [],[],[]
                    test_loss = 0.0
                    test_error = 0.0
                    batch_i = 0
                    for data in self.dl.getTestBatches():
                        input_data = np.array(data[0])
                        item_data = np.array(data[1])
                        len_data = np.array(data[2])
                        label_data = np.array(data[3])
                
                        error, loss, hr, ndcg = self.evaluate(input_data, item_data, label_data, len_data)
                        test_loss+=loss
                        test_error += error
                        batch_i+=1
                    train_loss /= batch_i
                    train_error /= batch_i
                    hits.append(hr)
                    ndcgs.append(ndcg)  
                    losses.append(test_loss)
                    hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                    eval_time = time.time() - eval_begin
        #             print("Epoch %d [%.1fs ]:  train_loss = %.4f" % (
        #                     epoch_count,train_time, train_loss))    
                    print("Epoch %d [ %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f ,error = %.4f [%.1fs] train_loss = %.4f ,train_error = %.4f [%.1fs]" % (
                                epoch_count, train_time, hr, ndcg, test_loss, test_error, eval_time, train_loss, train_error, train_time))

    


