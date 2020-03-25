"""Some general definations of apdt.
"""
import numpy as np
import pandas as pd
import threading

class DataPack():
    """DataPack is the standard data structure used by apdt.
    """
    def __init__(self):
        self.raw_data = None
        self.data = None
        self.site_info = None
        self.data_type = None
        self.sample_unit = None
        self.tag = []

class SubThread(threading.Thread):
    """SubThread is an interface of python threading.
    """
    def __init__(self, func, args = ()):
        super(SubThread, self).__init__()
        self.func = func
        self.args = args
    
    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

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
        T = datapack.time_length
        N = datapack.site_num
        self.data = datapack.data.reset_index().sort_values(['datetime','site_id'])['data0'].values.reshape((T,N,1))

        batch_perm = np.linspace(0, N-1, N)
        np.random.shuffle(batch_perm)
        self.data = self.data[:,batch_perm.astype(int),:]

        self.tr = self.data[:81*108+1]
        self.te = self.data[81*108+1:81*216+2]
        self.tr = np.transpose(self.tr,(1,0,2))
        self.te = np.transpose(self.te,(1,0,2))
        self.tr = np.concatenate((self.tr,np.concatenate((self.tr,self.tr[:,-1:]),1)[:,1:,1:]),-1)
        self.te = np.concatenate((self.te,np.concatenate((self.te,self.te[:,-1:]),1)[:,1:,1:]),-1)

        self.batch_size = args['seq_len']-1
        self.tr_batch_counter = 0
        self.tr_batch_num = (self.tr.shape[1]-1)//(args['seq_len']-1)
        self.tr_batch_perm = np.linspace(0,self.tr_batch_num-1,self.tr_batch_num)
        np.random.shuffle(self.tr_batch_perm)
        self.te_batch_counter = 0
        self.te_batch_num = (self.te.shape[1]-1)//(args['seq_len']-1)
        self.te_batch_perm = np.linspace(0,self.te_batch_num-1,self.te_batch_num)
        np.random.shuffle(self.te_batch_perm)

    def tr_get_batch(self,id=None):
        if id:
            idx_start = self.batch_size*id
            idx_end = self.batch_size*(id+1)
        else:
            id = int(self.tr_batch_perm[self.tr_batch_counter])
            idx_start = self.batch_size*id
            idx_end = self.batch_size*(id+1)
            self.tr_batch_counter = (self.tr_batch_counter+1)%self.tr_batch_num
            if self.tr_batch_counter==0:
                np.random.shuffle(self.tr_batch_perm)
        
        batch = self.tr[:,idx_start:idx_end+1]
        return batch

    def te_get_batch(self,id=None):
        if id:
            idx_start = self.batch_size*id
            idx_end = self.batch_size*(id+1)
        else:
            id = int(self.te_batch_perm[self.te_batch_counter])
            idx_start = self.batch_size*id
            idx_end = self.batch_size*(id+1)
            self.te_batch_counter = (self.te_batch_counter+1)%self.te_batch_num
            if self.te_batch_counter==0:
                np.random.shuffle(self.te_batch_perm)
        
        batch = self.te[:,idx_start:idx_end+1]
        return batch
