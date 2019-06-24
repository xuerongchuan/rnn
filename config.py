# -*- coding: utf-8 -*-

class Config(object):
    
    def __init__(self):
        
        self.data_path = 'data/'
        self.train_path = self.data_path + 'train.txt'
        self.test_path = None
<<<<<<< HEAD
        self.summary_path = 'summary/train/'
        self.epoches = 100
        self.verbose = 1
        self.num_users = 6040
        self.neg_count = 4
        #是否读取用户和movie的辅助数据
        self.user = 0
        self.movie = 0
=======
>>>>>>> parent of ee99985... 6-23

        #超参数
        #正则化参数
        self.loss = 'log_loss'
        self.lmbda = 0.001

       
        #优化器参数
<<<<<<< HEAD
        self.opt = 'adadelta'
        self.lr = 1e-4/6.0
=======
        self.opt = 'adam'
        self.lr = 0.001
>>>>>>> parent of ee99985... 6-23
        self.rho = 0.95
        self.epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999


        
<<<<<<< HEAD
        self.init_value = 0.1
        self.cell_type = 'rnn'
        self.layer_sizes = [32]
        self.layer_activations = ['tanh']
        self.embedding_size = 32
        self.max_window = 50
=======
        
        self.cell_type = 'rnn'
        self.layer_size = 64
        self.layer_activations = []
>>>>>>> parent of ee99985... 6-23
