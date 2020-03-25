import numpy as np
import pandas as pd
import tensorflow as tf

class DataSet():
    '''DataSet can make DataPack object being Machine Learning ready.
    '''
    def __init__(self, datapack, **kwarg):
        if 'time-aligned' not in datapack.tag:
            raise Exception("datapack should be aligned along time axis first.")
        
        if hasattr(kwarg['method'], '__call__'):
            kwarg['method'](datapack, **kwarg)
        elif kwarg['method'] in ['window']:
            if kwarg['method'] == 'window':
                self._construct_window(datapack, **kwarg)
        else:
            raise Exception("method should be predefined or a callable.")

    def _construct_window(self, datapack, **kwarg):
        self.method = 'window'
        
        if 'split_ratio' not in kwarg.keys():
            kwarg['split_ratio'] = 0.7
        if 'shuffle' not in kwarg.keys():
            kwarg['shuffle'] = True
        if 'seq_len' not in kwarg.keys():
            kwarg['seq_len'] = 100

        T = datapack.time_length
        N = datapack.site_num
        self.data = datapack.data.reset_index().sort_values(['datetime', 'site_id'])['data0'].values.reshape((T, N, 1))

        if kwarg['shuffle'] is True:
            batch_perm = np.linspace(0, N-1, N)
            np.random.shuffle(batch_perm)
            self.data = self.data[:, batch_perm.astype(int), :]

        split_point = int(T * kwarg['split_ratio'])
        self.tr = self.data[:split_point]
        self.te = self.data[split_point:]
        self.tr = np.transpose(self.tr, (1,0,2))
        self.te = np.transpose(self.te, (1,0,2))

        self.seq_len = kwarg['seq_len']
        self.tr_batch_counter = 0
        self.tr_batch_num = self.tr.shape[1]//kwarg['seq_len']
        self.tr_batch_perm = np.linspace(0,self.tr_batch_num-1,self.tr_batch_num)
        np.random.shuffle(self.tr_batch_perm)
        self.te_batch_counter = 0
        self.te_batch_num = self.te.shape[1]//kwarg['seq_len']
        self.te_batch_perm = np.linspace(0,self.te_batch_num-1,self.te_batch_num)
        np.random.shuffle(self.te_batch_perm)

    def tr_get_batch(self):
        if self.method=='window':
            id = int(self.tr_batch_perm[self.tr_batch_counter])
            idx_start = self.seq_len * id
            idx_end = self.seq_len * (id + 1)
            self.tr_batch_counter = (self.tr_batch_counter + 1) % self.tr_batch_num
            if self.tr_batch_counter==0:
                np.random.shuffle(self.tr_batch_perm)
            batch = self.tr[:, idx_start:idx_end]
            return batch

    def te_get_batch(self,id=None):
        if self.method=='window':
            id = int(self.te_batch_perm[self.te_batch_counter])
            idx_start = self.seq_len * id
            idx_end = self.seq_len * (id + 1)
            self.te_batch_counter = (self.te_batch_counter + 1) % self.te_batch_num
            if self.te_batch_counter==0:
                np.random.shuffle(self.te_batch_perm)
            batch = self.te[:,idx_start:idx_end]
            return batch

class TFModel():
    def __init__(self, **kwarg):
        self.def_model(**kwarg)
        self._setup_train_op()

    def def_model(self, **kwarg):
        self.input = None
        self.learning_rate = 0
        self.loss = 0

    def _setup_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
        gvs = [clipped_gvs[i] for i in range(0,len(clipped_gvs))]
        self.train_op = optimizer.apply_gradients(gvs)

    def update(self, dataset):
        pass

    def eval(self, data):
        pass

    def fit(self, dataset, **kwarg):
        pass
