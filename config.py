# -*- coding: utf-8 -*-

class Config(object):
    
    def __init__(self):
        
        self.data_path = '../data/ml-1m/rating.csv'
        self.train_path = '../data/ml-1m/train'
        self.test_path = '../data/ml-1m/test'

        self.summary_path = 'summary/train/'
        self.epoches = 100
        self.verbose = 1
        self.num_users = 6040
        self.neg_count = 4
        self.batch_size = 800
        #是否读取用户和movie的辅助数据
        self.user = 0
        self.movie = 0


        #超参数
        #正则化参数
        self.type_of_loss = 'log_loss'
        self.reg = 1e-3
        reg = self.reg
        self.regI1 = reg 
        self.regI2 = reg
        self.reg_bias = reg
        self.regU = reg
        self.reg_W = reg

       
        #优化器参数

        self.opt = 'adam'
        self.lr = 1e-3


        self.rho = 0.95
        self.epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999


        
        self.init_value = 0.1
        self.cell_type = 'gru'
        self.layer_sizes = [16]
        self.layer_activations = ['relu']
        self.embedding_size = 16
        self.window = 20
        self.min_window = 10
        self.max_window = 50
        self.batch_len = 128
        self.N = 10
        # attention
        self.weight_size = 16
        self.beta = 0.5
    def print_info(self):

        print('config: lr:%.6f, loss:%s, opt:%s, activation:%s, embedding_size:%d, cell:%s, reg:%.6f'%(self.lr, self.type_of_loss,self.opt, \
            self.layer_activations[-1],self.embedding_size, self.cell_type, self.reg))


