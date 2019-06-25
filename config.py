# -*- coding: utf-8 -*-

class Config(object):
    
    def __init__(self):
        
        self.data_path = 'data/ml-1m/rating.csv'
        self.train_path = self.data_path + 'rating.csv'
        self.test_path = None

        self.summary_path = 'summary/train/'
        self.epoches = 100
        self.verbose = 1
        self.num_users = 6040
        self.neg_count = 4
        #是否读取用户和movie的辅助数据
        self.user = 0
        self.movie = 0


        #超参数
        #正则化参数
        self.type_of_loss = 'log_loss'
        self.regI1 = 1e-4
        self.regI2 = 1e-4
        self.reg_bias = 1e-4

       
        #优化器参数

        self.opt = 'adadelta'
        self.lr = 0.1


        self.rho = 0.95
        self.epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999


        
        self.init_value = 0.1
        self.cell_type = 'rnn'
        self.layer_sizes = [16]
        self.layer_activations = ['tanh']
        self.embedding_size = 16
        self.min_window = 10
        self.max_window = 50

        print('config: lr:%.6f, loss:%s, opt:%s,activation:%S, embedding_size:%d'%(self.lr, self.type_of_loss,self.opt, \
            self.layer_activations[-1],self.embedding_size))

