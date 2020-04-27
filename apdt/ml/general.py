# pylint: disable=E1101
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import apdt
import pickle

class DataSet():
    '''DataSet can make DataPack object being Machine Learning ready.
    Description
    -----------
        Dataset object consist of three parts: pre_process, get_data, and post_process.

        - pre_process is to make DataPack obejct into a ML-ready ndarray self.tr and self.te, which have size N*D_1*D_2*...; The first dimension is considered as independent sample index; Beside this, two variable self.tr_batch_num and self.te_batch_num which claims the coressponding sample number should also be defined.
        - get_data has two key methods tr_get_batch and te_get_batch, will return a B*D_1*D_2*... as a batch of training/testing data, or list of ndarray when you need multi-input.
        - post_process defined the action that after every get_data call.

        In most case users have to re-define pre_process function to make their desired dataset structure. We also provide some pre-defined useful pre_process function for you. get_data and post_process typically don't need much work, but they can still be re-defined if you like.

    Parameters
    ----------
        datapack: DataPack object or str
            The datapack to be processed into a dataset. If a str is given, it will be considered as a pikle file path, and will construct dataset directly from that file.
        method: str
            If this argument is defined, we will use sepecific pre-defined pre_process fuction.
            Avaliable:
            - 'window': use a sliding window across time-axis to generate samples. locations in a time slice are supposed to be same (i.e. fixed-station). a samlple will be a N*T*D array. where N is the number of locations, T is window length and D is feature dimensions.
                argument for 'window':
                - split_ratio=0.7: the ratio of training set;
                - seq_len=100: the window length;
                - strides=seq_len-1: the sliding step;
                - normalize=True: if we do 0-1 normalize, Warning: dataset should always be properly normalized, and this operation is a in-place operation.
                - feature=['data0', 'data1', ...]: the feature list to be used. default to all features avaliable.

    '''
    def __init__(self, datapack, **kwarg):
        if type(datapack) is apdt.DataPack:
            self.pre_process(datapack, **kwarg)
        elif type(datapack) is str:
            self.load(datapack, **kwarg)

        self._init_counter(**kwarg)

    def dump(self, path):
        dumped_data = {'data': self.data, 'tr': self.tr, 'te': self.te}
        with open(path, 'wb') as f:
            pickle.dump(dumped_data, f, 1)

    def load(self, path, **kwarg):
        with open(path, 'rb') as f:
            dumped_data = pickle.load(f)
        self.data = dumped_data['data']
        self.tr = dumped_data['tr']
        self.te = dumped_data['te']
        self.tr_batch_num = self.tr.shape[0]
        self.te_batch_num = self.te.shape[0]

    def pre_process(self, datapack, **kwarg):
        if 'method' not in kwarg.keys():
            raise Exception("You have to re-define pre_process function or use a pre-defined one using 'method' argument.")

        if kwarg['method'] == 'window':
            if 'time-aligned' not in datapack.tag:
                raise Exception("datapack should be aligned along time axis first.")
            if 'fixed-location' not in datapack.tag:
                raise Exception("window method only support fixed station data.")
            
            if 'split_ratio' not in kwarg.keys():
                kwarg['split_ratio'] = 0.7
            if 'shuffle' not in kwarg.keys():
                kwarg['shuffle'] = True
            if 'seq_len' not in kwarg.keys():
                kwarg['seq_len'] = 100
            if 'strides' not in kwarg.keys():
                kwarg['strides'] = kwarg['seq_len'] - 1
            if 'normalize' not in kwarg.keys():
                kwarg['normalize'] = True
            if 'normalize_method' not in kwarg.keys():
                kwarg['normalize_method'] = '99pt'
            if 'feature' not in kwarg.keys():
                kwarg['feature'] = [x for x in datapack.data.columns if x.startswith('data')]

            self.method = 'window'
            T = datapack.time_length
            N = datapack.site_num
            D = len(kwarg['feature'])
            if kwarg['normalize']:
                datapack = apdt.proc.linear_normalize(datapack, kwarg['feature'], kwarg['normalize_method'])
            self.data = datapack.data.reset_index().sort_values(['datetime', 'site_id'])[kwarg['feature']].values.reshape((T, N, D))

            split_point = int(T * kwarg['split_ratio'])
            self.tr = self.data[:split_point]
            self.te = self.data[split_point:]
            self.tr_batch_num = (self.tr.shape[0]-kwarg['seq_len']+1+kwarg['strides'])//(kwarg['strides']+1)
            self.te_batch_num = (self.te.shape[0]-kwarg['seq_len']+1+kwarg['strides'])//(kwarg['strides']+1)
            if self.tr_batch_num == 0 or self.te_batch_num == 0:
                raise Exception("time_length is not enough to construct a window.")
            
            self.tr_list = []
            self.te_list = []
            for i in range(self.tr_batch_num):
                self.tr_list.append(self.tr[i*(kwarg['strides']+1):i*(kwarg['strides']+1)+kwarg['seq_len']])
            for i in range(self.te_batch_num):
                self.te_list.append(self.te[i*(kwarg['strides']+1):i*(kwarg['strides']+1)+kwarg['seq_len']])
            self.tr = np.array(self.tr_list)
            self.te = np.array(self.te_list)
            self.tr = np.transpose(self.tr, (0,2,1,3))
            self.te = np.transpose(self.te, (0,2,1,3))

    def _init_counter(self, **kwarg):
        if 'seed' in kwarg.keys():
            np.random.seed(kwarg['seed'])
        self.tr_batch_counter = 0
        self.te_batch_counter = 0
        self.tr_batch_perm = np.linspace(0, self.tr_batch_num-1, self.tr_batch_num, dtype=np.int32)
        np.random.shuffle(self.tr_batch_perm)
        self.te_batch_perm = np.linspace(0, self.te_batch_num-1, self.te_batch_num, dtype=np.int32)
        np.random.shuffle(self.te_batch_perm)

    def tr_get_batch(self, batch_size=1):
        if batch_size > self.tr_batch_num:
            batch_size = self.tr_batch_num
        if self.tr_batch_counter + batch_size > self.tr_batch_num:
            np.random.shuffle(self.tr_batch_perm)
            self.tr_batch_counter = 0
        target_index = self.tr_batch_perm[self.tr_batch_counter: self.tr_batch_counter + batch_size]
        self.tr_batch_counter = self.tr_batch_counter + batch_size
        batch = self.tr[target_index]
        batch = self.post_process(batch)
        return batch
    
    def te_get_batch(self, batch_size=1):
        if batch_size > self.te_batch_num:
            batch_size = self.te_batch_num
        if self.te_batch_counter + batch_size > self.te_batch_num:
            np.random.shuffle(self.te_batch_perm)
            self.te_batch_counter = 0
        target_index = self.te_batch_perm[self.te_batch_counter: self.te_batch_counter + batch_size]
        self.te_batch_counter = self.te_batch_counter + batch_size
        batch = self.te[target_index]
        batch = self.post_process(batch)
        return batch

    def post_process(self, batch):
        return batch

class TFModel():
    '''Provide model constructure interface based on Tensorflow (1.x).
    Description
    -----------
        There are some method of this class:

        - def_model(): you HAVE TO re-define this function to build your deep network. this fuction should define self.input, self.learning_rate, self.loss, and self.pred. self.input should be a placeholder tensor or a list of such, where training batch passed to TFModel will be sent into; self.learning_rate should be a placeholder tensor of single float value, which will control the gradient scale; self.loss should be a tensor of single float value, which will be minimized bu optimizer; self.pred should be a placeholder tensor or a list of such, which will be the output to be evaluated given some input.

        - fit(dataset): fit the model using given DataSet object. Note that please make sure the dataset structure is compatible with model's input interface.

        - eval(data): evaluate the model using given data (ndarray or a list of such, depends on self.input).

        - load(path): load parameters at given path.

    Parameters
    ----------
        - fit:
            - model_name='NewModel', str.
            - learning_rate=1e-3, float.
            - baseline=0, float. if the model loss is smaller than this, it will be automatically saved.
            - epoch=100, int. How many times should we go through the dataset.
            - print_every_n_epochs=1. int.
            - test_every_n_epochs=1. int.
            - learning_rate_decay_every_n_epochs=epoch+1. int. Disabled (i.e. fixed learning rate) as default.
            - learning_rate_decay=2.0. float. after every learning_rate_decay_every_n_epochs, we will apply lr=lr/learning_rate_decay to slow down the training process.
    '''
    def __init__(self, **kwarg):
        np.random.seed(0)
        tf.set_random_seed(0)
        self.def_model(**kwarg)
        self.setup_train_op(**kwarg)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def def_model(self, **kwarg):
        pass

    def setup_train_op(self, **kwarg):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
        gvs = [clipped_gvs[i] for i in range(0,len(clipped_gvs))]
        self.train_op = optimizer.apply_gradients(gvs)

    def load(self, path):
        self.saver.restore(self.sess, save_path=path)

    def update(self, data):
        pass

    def eval(self, data):
        if type(self.input) is list:
            feed_dict = {self.input[i]: data[i] for i in range(len(self.input))}
            feed_dict.update({self.learning_rate: 0.0})
        else:
            feed_dict = {self.learning_rate: 0.0, self.input: data}
        result = self.sess.run(self.pred, feed_dict)
        return result

    def fit(self, dataset, **kwarg):
        # parameter checking
        if 'model_name' not in kwarg.keys():
            kwarg['model_name'] = 'NewModel'
        if 'learning_rate' not in kwarg.keys():
            kwarg['learning_rate'] = 1e-3
        if 'baseline' not in kwarg.keys():
            kwarg['baseline'] = 0
        if 'epoch' not in kwarg.keys():
            kwarg['epoch'] = 100
        if 'print_every_n_epochs' not in kwarg.keys():
            kwarg['print_every_n_epochs'] = 1
        if 'test_every_n_epochs' not in kwarg.keys():
            kwarg['test_every_n_epochs'] = 1
        if 'learning_rate_decay_every_n_epochs' not in kwarg.keys():
            kwarg['learning_rate_decay_every_n_epochs'] = kwarg['epoch'] + 1
        if 'learning_rate_decay' not in kwarg.keys():
            kwarg['learning_rate_decay'] = 2.0
        if 'epoch' not in kwarg.keys():
            kwarg['epoch'] = 100

        version = time.strftime('%Y%m%d_%H%M%S')
        if not os.path.exists('model/'):
            os.mkdir('model/')
        os.mkdir('model/' + kwarg['model_name'] + version)
        lr = float(kwarg['learning_rate'])
        performance_recorder = kwarg['baseline']
        for epoch in range(kwarg['epoch']):
            # update an epoch
            train_ls = []
            for _ in range(dataset.tr_batch_num):
                batch = dataset.tr_get_batch()
                if type(self.input) is list:
                    feed_dict = {self.input[i]: batch[i] for i in range(len(self.input))}
                    feed_dict.update({self.learning_rate: lr})
                else:
                    feed_dict = {self.learning_rate: lr, self.input: batch}
                _, ls = self.sess.run([self.train_op, self.loss], feed_dict)
                train_ls.append(ls)
            
            # print log
            if (epoch+1)%kwarg['print_every_n_epochs'] == 0:
                train_ls = np.mean(train_ls)
                print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Done, Train loss ',round(train_ls,4))
            
            # learning_rate_decay
            if (epoch+1)%kwarg['learning_rate_decay_every_n_epochs'] == 0:
                    lr = lr/kwarg['learning_rate_decay']
                    print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],', Learning rate decay to ',lr)

            # test
            if (epoch+1)%kwarg['test_every_n_epochs'] == 0:
                test_ls = []
                for _ in range(dataset.te_batch_num):
                    batch = dataset.te_get_batch()
                    if type(self.input) is list:
                        feed_dict = {self.input[i]: batch[i] for i in range(len(self.input))}
                        feed_dict.update({self.learning_rate: lr})
                    else:
                        feed_dict = {self.learning_rate: lr, self.input: batch}
                    ls = self.sess.run(self.loss, feed_dict)
                    test_ls.append(ls)
                test_ls = np.mean(test_ls)

                print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Done, Test loss ',round(test_ls,4))
                target = test_ls
                if target < performance_recorder:
                    self.saver.save(self.sess,'model/'+kwarg['model_name']+version+'/model')
                    print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Model Save Success. New record ',target)
                    performance_recorder = target
        pass
