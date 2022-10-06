# pylint: disable=E1101
import numpy as np
import apdt

class sin_regression():
    '''Sin regression meta dataset. Generate random functions of form f(x)=Asin(wx+phi)

    Parameters
    -----
        s_size: int, default 5, support sample number.
        q_size: int, default 20, query sample number.
        dataset_size: tuple of int, default (10000, 2000, 10000), number of functions in train/val/test set.
        A_range: tuple of int, default (0.1, 5.0), uniform random range of parameter A.
        phi_range: tuple of int, default (0.0, np.pi), uniform random range of parameter phi.
        w_range: tuple of int, default (8.0, 12.0), uniform random range of parameter w.
        *_range_test: parameter range of test set, default to be same as train/val. in case of you need train-test shift.
    '''
    def __init__(self, **kwarg):
        if "s_size" not in kwarg.keys():
            kwarg["s_size"] = 5
        if "q_size" not in kwarg.keys():
            kwarg["q_size"] = 20
        if "dataset_size" not in kwarg.keys():
            kwarg["dataset_size"] = (10000, 2000, 10000)
        if "A_range" not in kwarg.keys():
            kwarg["A_range"] = (0.1, 5.0)
        if "phi_range" not in kwarg.keys():
            kwarg["phi_range"] = (0.0, np.pi)
        if "w_range" not in kwarg.keys():
            kwarg["w_range"] = (8.0, 12.0)
        if "A_range_test" not in kwarg.keys():
            kwarg["A_range_test"] = kwarg["A_range"]
        if "phi_range_test" not in kwarg.keys():
            kwarg["phi_range_test"] = kwarg["phi_range"]
        if "w_range_test" not in kwarg.keys():
            kwarg["w_range_test"] = kwarg["w_range"]
        self.kwarg = kwarg
        self.tr_batch_num = kwarg["dataset_size"][0]
        self.val_batch_num = kwarg["dataset_size"][1]
        self.te_batch_num = kwarg["dataset_size"][2]
    
    def get_data(self, A, phi, w, batch_size):
        x_s = np.random.rand(batch_size, self.kwarg["s_size"], 1)
        y_s = A * np.sin(x_s * w + phi)
        x_q = np.random.rand(batch_size, self.kwarg["q_size"], 1)
        y_q = A * np.sin(x_q * w + phi)
        return [x_s, y_s, x_q, y_q]
    
    def tr_get_batch(self, batch_size=1):
        A = np.random.rand(batch_size, 1, 1) * (self.kwarg["A_range"][1] - self.kwarg["A_range"][0]) + self.kwarg["A_range"][0]
        phi = np.random.rand(batch_size, 1, 1) * (self.kwarg["phi_range"][1] - self.kwarg["phi_range"][0]) + self.kwarg["phi_range"][0]
        w = np.random.rand(batch_size, 1, 1) * (self.kwarg["w_range"][1] - self.kwarg["w_range"][0]) + self.kwarg["w_range"][0]
        return self.get_data(A, phi, w, batch_size)

    def val_get_batch(self, batch_size=1):
        A = np.random.rand(batch_size, 1, 1) * (self.kwarg["A_range"][1] - self.kwarg["A_range"][0]) + self.kwarg["A_range"][0]
        phi = np.random.rand(batch_size, 1, 1) * (self.kwarg["phi_range"][1] - self.kwarg["phi_range"][0]) + self.kwarg["phi_range"][0]
        w = np.random.rand(batch_size, 1, 1) * (self.kwarg["w_range"][1] - self.kwarg["w_range"][0]) + self.kwarg["w_range"][0]
        return self.get_data(A, phi, w, batch_size)

    def te_get_batch(self, batch_size=1):
        A = np.random.rand(batch_size, 1, 1) * (self.kwarg["A_range_test"][1] - self.kwarg["A_range_test"][0]) + self.kwarg["A_range_test"][0]
        phi = np.random.rand(batch_size, 1, 1) * (self.kwarg["phi_range_test"][1] - self.kwarg["phi_range_test"][0]) + self.kwarg["phi_range_test"][0]
        w = np.random.rand(batch_size, 1, 1) * (self.kwarg["w_range_test"][1] - self.kwarg["w_range_test"][0]) + self.kwarg["w_range_test"][0]
        return self.get_data(A, phi, w, batch_size)

class ContinualDataset():
    ''' A Meta Dataset for Continual Learning purpose. It first construct a large dataset, and then return subset as sub-dataset.

    Parameter
    ------
        datapack: an apdt.DataPack
        task_num: the total number of child dataset

    Method
    ------
        get_dataset: return a sub-dataset.
    '''
    def __init__(self, datapack, **kwarg):
        if 'seed' not in kwarg.keys():
            kwarg['seed'] = None
        if 'task_num' not in kwarg.keys():
            kwarg['task_num'] = 10

        self._kwarg = kwarg

        if type(datapack) is apdt.DataPack:
            self.pre_process(datapack, **kwarg)
        else:
            raise NotImplementedError

        if kwarg['task_num'] > self.tr_batch_num:
            raise Exception("No enough data to construct ", kwarg['task_num'], " sub-datasets.")

    def pre_process(self, datapack, **kwarg):
        if 'method' not in kwarg.keys():
            raise Exception("You have to re-define pre_process function or use a pre-defined one using 'method' argument.")

        if kwarg['method'] == 'window':
            if 'time-aligned' not in datapack.tag:
                raise Exception("datapack should be aligned along time axis first.")
            if 'fixed-location' not in datapack.tag:
                raise Exception("window method only support fixed station data.")
            
            if 'seq_len' not in kwarg.keys():
                kwarg['seq_len'] = 100
            if 'strides' not in kwarg.keys():
                kwarg['strides'] = kwarg['seq_len']
            if 'normalize' not in kwarg.keys():
                kwarg['normalize'] = True
            if 'normalize_method' not in kwarg.keys():
                kwarg['normalize_method'] = '99pt'
            if 'feature' not in kwarg.keys():
                kwarg['feature'] = [x for x in datapack.data.columns if x.startswith('data')]

            self.method = 'window'
            T = len(datapack.data.index.drop_duplicates()) 
            N = len(datapack.data.site_id.drop_duplicates())
            D = len(kwarg['feature'])
            if kwarg['normalize']:
                datapack = apdt.proc.linear_normalize(datapack, kwarg['feature'], kwarg['normalize_method'])
            self.data = datapack.data.reset_index().sort_values(['datetime', 'site_id'])[kwarg['feature']].values.reshape((T, N, D))
            
            tr_list = []
            self.tr_batch_num = (T - kwarg['seq_len']) // kwarg['strides'] + 1
            for i in range(self.tr_batch_num):
                tr_list.append(self.data[i*kwarg['strides']:i*kwarg['strides']+kwarg['seq_len']])
            self.tr = np.stack(tr_list)
            self.tr = np.transpose(self.tr, (0,2,1,3))

            self.seq_len = kwarg['seq_len']
            self.strides = kwarg['strides']

    def get_dataset(self, idx1, idx2=None, split_ratio=None):
        ''' Return a dataset that include samples from task idx1 (or a combined dataset from task idx1 - idx2 (include))

        Parameters
        -----
            idx1: target task index, 0<=idx1<=self.task_num
            idx2: if given, task idx1 - idx2 (include both ends) will be combined into one and returned
            split_ratio: the ratio to bu used as training set. note that this is a random split instead of split-along-time.
        '''
        sample_num_per_task = self.tr.shape[0] // self._kwarg['task_num']
        idx_start = idx1 * sample_num_per_task
        if idx2 is None:
            idx2 = idx1
        idx_end = (idx2 + 1) * sample_num_per_task
        return _ContinualDataset_child(self, idx_start, idx_end, split_ratio, self._kwarg['seed'])

class _ContinualDataset_child():
    '''A dataset object generated from ContinualDataset, provide basic interface same as apdt.DataSet, e.g., shuffle, get_batch.

    Note: This dataset use a random split of train/val instead of split-along-time
    '''
    def __init__(self, father, idx_start, idx_end, split_ratio=None, seed=None):
        '''
        Parameters
        -----
            father: ContinualDataset instance.
            idx_start: start sample index (include)
            idx_end: start sample index (exclude)
            split_ratio: the ratio of training set, rest will be used as validation. default no validation.
            seed: seed for randomness, default none.
        '''
        if split_ratio is not None:
            if seed is not None:
                np.random.seed(seed)
            perm = np.arange(idx_start, idx_end)
            np.random.shuffle(perm)
            tr_idx = perm[:int((idx_end-idx_start)*split_ratio)]
            val_idx = perm[int((idx_end-idx_start)*split_ratio):]
            self.tr = father.tr[tr_idx]
            self.val = father.tr[val_idx]
            self.te = father.tr[idx_start: idx_end]
        else:
            self.tr = father.tr[idx_start: idx_end]
            self.val = father.tr[idx_start: idx_end]
            self.te = father.tr[idx_start: idx_end]

        self.tr_batch_num = self.tr.shape[0]
        self.val_batch_num = self.val.shape[0]
        self.te_batch_num = self.te.shape[0]

        self._init_counter(seed=seed)

    def _init_counter(self, **kwarg):
        if kwarg['seed'] is not None:
            np.random.seed(kwarg['seed'])
        self.tr_batch_counter = 0
        self.val_batch_counter = 0
        self.te_batch_counter = 0
        self.tr_batch_perm = np.linspace(0, self.tr_batch_num-1, self.tr_batch_num, dtype=np.int32)
        self.val_batch_perm = np.linspace(0, self.val_batch_num-1, self.val_batch_num, dtype=np.int32)
        self.te_batch_perm = np.linspace(0, self.te_batch_num-1, self.te_batch_num, dtype=np.int32)
        np.random.shuffle(self.tr_batch_perm)

    def tr_get_batch(self, batch_size=1):
        if batch_size > self.tr_batch_num:
            batch_size = self.tr_batch_num
        if self.tr_batch_counter + batch_size > self.tr_batch_num:
            np.random.shuffle(self.tr_batch_perm)
            self.tr_batch_counter = 0
        target_index = self.tr_batch_perm[self.tr_batch_counter: self.tr_batch_counter + batch_size]
        self.tr_batch_counter = self.tr_batch_counter + batch_size
        batch = self.tr[target_index]
        batch = self.post_process(batch, mode='train')
        return batch

    def val_get_batch(self, batch_size=1):
        if batch_size > self.val_batch_num:
            batch_size = self.val_batch_num
        if self.val_batch_counter + batch_size > self.val_batch_num:
            self.val_batch_counter = 0
        target_index = self.val_batch_perm[self.val_batch_counter: self.val_batch_counter + batch_size]
        self.val_batch_counter = self.val_batch_counter + batch_size
        batch = self.val[target_index]
        batch = self.post_process(batch, mode='validate')
        return batch
    
    def te_get_batch(self, batch_size=1):
        if batch_size > self.te_batch_num:
            batch_size = self.te_batch_num
        if self.te_batch_counter + batch_size > self.te_batch_num:
            self.te_batch_counter = 0
        target_index = self.te_batch_perm[self.te_batch_counter: self.te_batch_counter + batch_size]
        self.te_batch_counter = self.te_batch_counter + batch_size
        batch = self.te[target_index]
        batch = self.post_process(batch, mode='test')
        return batch

    def post_process(self, batch, **kwarg):
        '''Define the action on post_process of batches from dataset, default doing nothing.

        Parameters
        ------
            batch, tensor
            mode, in ['train, 'validate', 'test]
        '''
        return batch