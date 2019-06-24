# -*- coding: utf-8 -*-

import tensorflow as tf 



class BaseRS(object):
    def __init__(self, config):
        self.conig = config
    
    def _create_placeholders(self):
        pass
    
    def _create_variables(self):
        pass
    
    def _create_inference(self):
        pass
    
    def _get_loss(self, preds, Y):
        if self.type_of_loss == 'cross_entropy_loss':
            error = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(preds, [-1]), labels=tf.reshape(Y, [-1])),
                           name='cross_entropy_loss'
                           )
        elif self.type_of_loss == 'square_loss' or self.type_of_loss == 'rmse':
            error = tf.reduce_mean(tf.squared_difference(preds, Y, name='squared_diff'), name='mean_squared_diff')
        elif self.type_of_loss == 'log_loss':
            error = tf.reduce_mean(tf.losses.log_loss(predictions=preds, labels=Y), name='mean_log_loss')
        
        return error
    def _create_loss(self):
        pass
    
    def _optimize(self, loss , model_params ):
        if self.opt == 'adadelta':
            train_step = tf.train.AdadeltaOptimizer(self.lr, self.rho, self.epsilon).minimize(loss, var_list= model_params)
        elif self.opt == 'sgd':
            train_step = tf.train.GradientDescentOptimizer(self.lr, self.beta1, self.beta2, self.epsilon).minimize(loss,var_list=model_params)
        elif self.opt =='adam':
            train_step = tf.train.AdamOptimizer(self.lr).minimize(loss, var_list=model_params)
        elif self.opt =='ftrl':
            train_step = tf.train.FtrlOptimizer(self.lr).minimize(loss, var_list=model_params)
        return train_step

    def _create_optimizer(self):
        pass
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
    
    def _get_activation_func(self, name):
        if name == "tanh":
            return tf.tanh 
        elif name == 'sigmoid':
            return tf.sigmoid
        elif name == 'identity':
            return tf.identity
        elif name == 'relu':
            return tf.nn.relu
        elif name == 'relu6':
            return tf.nn.relu6
        return tf.tanh 
    def _init_graph(self):
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        
    