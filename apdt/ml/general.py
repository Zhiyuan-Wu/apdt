# pylint: disable=E1101
import numpy as np
import pandas as pd
import os
import time
import apdt
import pickle
from apdt.ml.utils import _unzip_list

try:
    import tensorflow as tf
except:
    pass

class DataSet():
    '''DataSet can make DataPack object being Machine Learning ready.
    Description
    -----------
        Dataset object consist of three parts: pre_process, get_data, and post_process.

        - pre_process is to make DataPack obejct into a ML-ready ndarray self.tr, self.val and self.te, which have size N*D_1*D_2*...; The first dimension is considered as independent sample index; Beside this, three variables self.tr_batch_num, self.val_batch_num and self.te_batch_num which claims the coressponding sample number should also be defined.
        - get_data has three key methods tr_get_batch, val_get_batch and te_get_batch, will return a B*D_1*D_2*... as a batch of training/validation/testing data, or list of ndarray when you need multi-input.
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
                - split_ratio=0.7: the ratio of training set. Example: when split_ratio=0.7, training set will take first 70% data, validation set and testing set will be identical and take last 30%. when split_ration=(0.5, 0.25, 0.25), training set will take first 50% data, validation set will take middle 25% data, and testing set take last 25% data. When given None, there's no splition, training/validation/testing will use same entire dataset.
                - seq_len=100: the window length;
                - strides=seq_len-1: the sliding step;
                - sub_sample=1: the sub_sample rate. The data time span should be at least seq_len*sub_sample (instead of (seq_len-1)*sub_sample+1) to ensure one sample
                - normalize=True: if we do 0-1 normalize, Warning: dataset should always be properly normalized, and this operation is a in-place operation.
                - feature=['data0', 'data1', ...]: the feature list to be used. default to all features avaliable.
        seed: int
            The initial random seed, default no seed.
        shuffle_train_only: bool, default False
            if False, Train/Val/Test set will be shuffled at every glance. if True, only Train set will.

    '''
    def __init__(self, datapack, **kwarg):
        if 'supervised' not in kwarg.keys():
            kwarg['supervised'] = False
        if 'seed' not in kwarg.keys():
            kwarg['seed'] = None
        if 'shuffle_train_only' not in kwarg.keys():
            kwarg['shuffle_train_only'] = False

        self._kwarg = kwarg
        self.supervised = kwarg['supervised']

        if type(datapack) is apdt.DataPack:
            self.pre_process(datapack, **kwarg)
        elif type(datapack) is str:
            if datapack=='MNIST':
                self.MNIST(**kwarg)
            else:
                self.load(datapack, **kwarg)

        self._init_counter(**kwarg)

    def dump(self, path):
        dumped_data = {'data': self.data, 'tr': self.tr, 'val': self.val, 'te': self.te}
        with open(path, 'wb') as f:
            pickle.dump(dumped_data, f, 1)

    def load(self, path, **kwarg):
        with open(path, 'rb') as f:
            dumped_data = pickle.load(f)
        self.data = dumped_data['data']
        self.tr = dumped_data['tr']
        if 'val' in dumped_data.keys():
            self.val = dumped_data['val']
        else:
            self.val = dumped_data['te']
            print('WARNING: ' + path + 'is generated by early version of apdt and will not be supported soon.')
        self.te = dumped_data['te']
        self.tr_batch_num = self.tr.shape[0]
        self.val_batch_num = self.val.shape[0]
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
            if 'sub_sample' not in kwarg.keys():
                kwarg['sub_sample'] = 1
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

            if kwarg['split_ratio'] is None:
                self.tr = self.data
                self.val = self.data
                self.te = self.data
            elif type(kwarg['split_ratio']) is float:
                split_point = int(T * kwarg['split_ratio'])
                self.tr = self.data[:split_point]
                self.val = self.data[split_point:]
                self.te = self.data[split_point:]
            elif type(kwarg['split_ratio']) is tuple:
                split_point_1 = int(T * kwarg['split_ratio'][0])
                split_point_2 = split_point_1 + int(T * kwarg['split_ratio'][1])
                self.tr = self.data[:split_point_1]
                self.val = self.data[split_point_1:split_point_2]
                self.te = self.data[split_point_2:]
            self.tr_batch_num = (self.tr.shape[0]//kwarg['sub_sample']-kwarg['seq_len']+kwarg['strides'])//(kwarg['strides'])*kwarg['sub_sample']
            self.val_batch_num = (self.val.shape[0]//kwarg['sub_sample']-kwarg['seq_len']+kwarg['strides'])//(kwarg['strides'])*kwarg['sub_sample']
            self.te_batch_num = (self.te.shape[0]//kwarg['sub_sample']-kwarg['seq_len']+kwarg['strides'])//(kwarg['strides'])*kwarg['sub_sample']
            if self.tr_batch_num <= 0 or self.val_batch_num <= 0 or self.te_batch_num <= 0:
                raise Exception("time_length is not enough to construct a window.")
            
            self.tr_list = []
            self.val_list = []
            self.te_list = []
            for i in range(self.tr_batch_num//kwarg['sub_sample']):
                for j in range(kwarg['sub_sample']):
                    self.tr_list.append(self.tr[i*(kwarg['strides']*kwarg['sub_sample'])+j:i*(kwarg['strides']*kwarg['sub_sample'])+kwarg['seq_len']*kwarg['sub_sample']:kwarg['sub_sample']])
            for i in range(self.val_batch_num//kwarg['sub_sample']):
                for j in range(kwarg['sub_sample']):
                    self.val_list.append(self.val[i*(kwarg['strides']*kwarg['sub_sample'])+j:i*(kwarg['strides']*kwarg['sub_sample'])+kwarg['seq_len']*kwarg['sub_sample']:kwarg['sub_sample']])
            for i in range(self.te_batch_num//kwarg['sub_sample']):
                for j in range(kwarg['sub_sample']):
                    self.te_list.append(self.te[i*(kwarg['strides']*kwarg['sub_sample'])+j:i*(kwarg['strides']*kwarg['sub_sample'])+kwarg['seq_len']*kwarg['sub_sample']:kwarg['sub_sample']])
            self.tr = np.array(self.tr_list)
            self.val = np.array(self.val_list)
            self.te = np.array(self.te_list)
            self.tr = np.transpose(self.tr, (0,2,1,3))
            self.val = np.transpose(self.val, (0,2,1,3))
            self.te = np.transpose(self.te, (0,2,1,3))

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
        if not kwarg['shuffle_train_only']:
            np.random.shuffle(self.val_batch_perm)
            np.random.shuffle(self.te_batch_perm)

    def tr_get_batch(self, batch_size=1):
        if batch_size > self.tr_batch_num:
            batch_size = self.tr_batch_num
        if self.tr_batch_counter + batch_size > self.tr_batch_num:
            np.random.shuffle(self.tr_batch_perm)
            self.tr_batch_counter = 0
        target_index = self.tr_batch_perm[self.tr_batch_counter: self.tr_batch_counter + batch_size]
        self.tr_batch_counter = self.tr_batch_counter + batch_size
        if self.supervised:
            batchx = self.tr[target_index]
            batchy = self.tr_y[target_index]
            batch = self.post_process([batchx, batchy], mode='train')
        else:
            batch = self.tr[target_index]
            batch = self.post_process(batch, mode='train')
        return batch

    def val_get_batch(self, batch_size=1):
        if batch_size > self.val_batch_num:
            batch_size = self.val_batch_num
        if self.val_batch_counter + batch_size > self.val_batch_num:
            if not self._kwarg['shuffle_train_only']:
                np.random.shuffle(self.val_batch_perm)
            self.val_batch_counter = 0
        target_index = self.val_batch_perm[self.val_batch_counter: self.val_batch_counter + batch_size]
        self.val_batch_counter = self.val_batch_counter + batch_size
        if self.supervised:
            batchx = self.val[target_index]
            batchy = self.val_y[target_index]
            batch = self.post_process([batchx, batchy], mode='validate')
        else:
            batch = self.val[target_index]
            batch = self.post_process(batch, mode='validate')
        return batch
    
    def te_get_batch(self, batch_size=1):
        if batch_size > self.te_batch_num:
            batch_size = self.te_batch_num
        if self.te_batch_counter + batch_size > self.te_batch_num:
            if not self._kwarg['shuffle_train_only']:
                np.random.shuffle(self.te_batch_perm)
            self.te_batch_counter = 0
        target_index = self.te_batch_perm[self.te_batch_counter: self.te_batch_counter + batch_size]
        self.te_batch_counter = self.te_batch_counter + batch_size
        if self.supervised:
            batchx = self.te[target_index]
            batchy = self.te_y[target_index]
            batch = self.post_process([batchx, batchy], mode='test')
        else:
            batch = self.te[target_index]
            batch = self.post_process(batch, mode='test')
        return batch

    def post_process(self, batch, **kwarg):
        return batch

    def MNIST(self, **kwarg):
        # Load data from keras interface
        import tensorflow as tf
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train = (np.outer(np.ones((y_train.shape[0],)), np.arange(10)) == y_train.reshape([-1, 1])) * 1.0
        y_test = (np.outer(np.ones((y_test.shape[0],)), np.arange(10)) == y_test.reshape([-1, 1])) * 1.0

        # Construct dataset
        self.supervised = True
        self.tr = x_train
        self.val = x_test
        self.te = x_test
        self.tr_y = y_train
        self.val_y = y_test
        self.te_y = y_test
        self.tr_batch_num = self.tr.shape[0]
        self.val_batch_num = self.val.shape[0]
        self.te_batch_num = self.te.shape[0]

class TFModel():
    '''Provide model constructure interface based on Tensorflow (1.x).
    Description
    -----------
        There are some method of this class:

        - def_model(): you HAVE TO re-define this function to build your deep network. this fuction should define:
            - self.input, self.input should be a placeholder tensor or a list of such, where training batch passed to TFModel will be sent into;
            - self.metric, self.metric should be a list of tensor, which will be evaluated and printed during training, and will by default be set to [self.loss];
            - self.loss, self.loss should be a tensor of single float value, which will be minimized by optimizer;
            - self.pred. self.pred should be a tensor or a list of such, which will be the output to be evaluated given some input.
            - self.learning_rate (optional), self.learning_rate should be a placeholder tensor of single float value, which will control the gradient scale;
            - self.pretrain_var_map (optional), self.pretrain_var_map is only needed when pre-train model is given. The map between model variables, which need pre-train values instead of random initialization, to its name in pretrain model. By default {v.op.name: v for v in tf.trainable_variables()}.

        - fit(dataset): fit the model using given DataSet object. Note that please make sure the dataset structure is compatible with model's input interface.

        - eval(data): evaluate the model using given data (ndarray or a list of such, depends on self.input).

        - load(path): load parameters at given path.

    Parameters
    ----------
        - model.__init__()
            - l2_norm, float, default None.
                A L2-Regularization will apllied with weight decay parameter as l2_norm, It will be disabled when given None.
            - clip_gvs, bool, default False.
                all gradients in network will be cliiped to [-1,1] when set True.
            - seed, int, default to currrent time.
                The random seed to be set in Numpy and Tensorflow.
            - pretrain_model, str, default None.
                The pretrain model path.
        
        - model.fit():
            - model_name='NewModel', str.
            - lr=1e-3, float.
            - baseline=0, float. if the model loss is smaller than this, it will be automatically saved.
            - epoch=100, int. How many times should we go through the dataset.
            - print_every_n_epochs=1. int.
            - test_every_n_epochs=1. int.
            - lr_annealing='constant'. str. How to decay learning rate during training. optional: 'constant', 'step', 'cosine'.
            - lr_annealing_step_length=epoch/4. int. Avaliable when lr_annealing is set to 'step'. How often do we decay the learning rate.
            - lr_annealing_step_divisor=2.0. float. Avaliable when lr_annealing is set to 'step', we will apply lr=lr/lr_annealing_step_divisor to slow down the training process.
            - early_stop=None. int. Stop training if there is no better validation result since last recorder. This parameter decides the waiting epoch number. If None this feature will be disabled.
            - repeat=1. int. If set >1, we will randomly reset parameters and retrain the model for certain times, and compute statistical performance.
            - verbose=0. int. The verbose level. 0-print all. 1-only summary. 2-mute.
    '''
    def __init__(self, **kwarg):
        # Parameter Check
        if 'seed' not in kwarg.keys():
            kwarg['seed'] = None
        if 'pretrain_model' not in kwarg.keys():
            kwarg['pretrain_model'] = None
        self.kwarg = kwarg
        
        # Initial tensor
        self.training = tf.placeholder(tf.bool)
        self.learning_rate = tf.placeholder(tf.float32)
        self.training_process = tf.placeholder(tf.float32)
        self.input = [None]
        self.pred = None
        self.loss = None
        self.metric = None
        self.metric_name = None
        self.pretrain_var_map = None
        self.summary_merged = None

        # Define Model
        self.def_model(**kwarg)
        if self.loss is not None:
            if self.metric is None:
                self.metric = [self.loss]
            elif type(self.metric) is not list:
                self.metric = [self.metric]
            if self.metric_name is None:
                self.metric_name = ['metric_' + str(i) for i in range(len(self.metric))]
            if len(self.metric_name) < len(self.metric):
                _diff = len(self.metric) - len(self.metric_name)
                self.metric_name = self.metric_name + ['metric_' + str(i) for i in range(_diff)]
            self.setup_train_op(**kwarg)
        else:
            raise Exception('self.loss not defined in your model.')        
        if self.pred is None:
            self.pred = self.metric[0]
        if self.summary_merged is None and len(tf.get_collection(tf.GraphKeys.SUMMARIES)) > 0:
            self.summary_merged = tf.summary.merge_all()
        self.variables_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        
        # Start Engine
        self.saver = tf.train.Saver(max_to_keep=None)
        if kwarg['pretrain_model'] is not None:
            if type(kwarg['pretrain_model']) is str:
                if self.pretrain_var_map is None:
                    self.pretrain_var_map = {v.op.name: v for v in tf.trainable_variables()}
                self._pre_saver = tf.train.Saver(var_list=self.pretrain_var_map, max_to_keep=None)
            elif type(kwarg['pretrain_model']) is list:
                if self.pretrain_var_map is None:
                    self.pretrain_var_map = [{v.op.name: v for v in tf.trainable_variables()} for _ in range(len(kwarg['pretrain_model']))]
                self._pre_saver = [tf.train.Saver(var_list=x, max_to_keep=None) for x in self.pretrain_var_map]
            else:
                raise Exception("pretrain_model must be a path or a list of path.")
        self.sess = tf.Session()
        self.init_parameter()

    def init_parameter(self, seed=None):
        # Parameter Initialize
        if seed is None:
            if self.kwarg['seed'] is not None:
                seed = self.kwarg['seed']
            else:
                seed = int(time.time())
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.sess.run(tf.global_variables_initializer())
        if self.kwarg['pretrain_model'] is not None:
            if type(self.kwarg['pretrain_model']) is str:
                self._pre_saver.restore(self.sess, save_path=self.kwarg['pretrain_model'])
            elif type(self.kwarg['pretrain_model']) is list:
                for i in range(len(self.kwarg['pretrain_model'])):
                    self._pre_saver[i].restore(self.sess, save_path=self.kwarg['pretrain_model'][i])

    def def_model(self, **kwarg):
        pass

    def setup_train_op(self, **kwarg):
        # Parameter check
        if 'l2_norm' not in kwarg.keys():
            kwarg['l2_norm'] = None
        if 'keep_norm_loss' not in kwarg.keys():
            kwarg['keep_norm_loss'] = True
        if 'clip_gvs' not in kwarg.keys():
            kwarg['clip_gvs'] = False

        # Set up optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        # Add global regulizer
        if kwarg['l2_norm']:
            l2_loss =  kwarg['l2_norm'] * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
            if kwarg['keep_norm_loss']:
                self.loss = self.loss + l2_loss
                gvs = optimizer.compute_gradients(self.loss)
            else:
                gvs = optimizer.compute_gradients(self.loss + l2_loss)
        else:
            gvs = optimizer.compute_gradients(self.loss)
        
        # Clip Gradients
        if kwarg['clip_gvs']:
            clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
            gvs = [clipped_gvs[i] for i in range(0,len(clipped_gvs))]
        
        # Check update ops.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.apply_gradients(gvs)

    def load(self, path):
        self.saver.restore(self.sess, save_path=path)

    def update(self, feed_dict):
        '''Apply an update on network based on given feed_dict and return statistics.
        Parameters
        ----------
            feed_dict, dict
                a dictionary that maps all necessary input tensor to their input value 
        
        Returns
        ----------
            out, a list of list, where: 
                out[0] is a list of real value, considered as loss in report;
                out[1+k] is a list of real value, considered as metric[k] in report;
                out[1+K] is a list of summary result, if self.summary_merged is not None;
        '''
        target = [self.train_op, self.loss] + self.metric
        if self.summary_merged is not None:
            target = target + [self.summary_merged]
        _re = self._zip_run(target, feed_dict)
        _re = _unzip_list(_re)
        return _re[1:]

    def eval(self, data):
        if type(self.input) is list:
            feed_dict = {self.input[i]: data[i] for i in range(len(self.input))}
            feed_dict.update({self.learning_rate: 0.0, self.training: False, self.training_process: 0.0})
        else:
            feed_dict = {self.learning_rate: 0.0, self.training: False, self.training_process: 0.0, self.input: data}
        result = self.sess.run(self.pred, feed_dict)
        return result

    def _zip_run(self, target, feed_dict, _r=0):
        '''Recurrently unzip lists in feed_dict to generate samples.
        Description
        -----------
            Compared to self.sess.run(target, feed_dict), self._zip_run(target, feed_dict) support list values in feed_dict, where the average of unziped feed_dict will be returned.
        
        Parameters
        ----------
            target, TF tensor or a list of TF tensor
                the tensorflow tensor to be evaluated.
            feed_dict, dict
                the feed data, list values supported.
        
        Returns
        ----------
            ndarray or a multi-level list of ndarray

        Examples
        ----------
            If we run self._zip_run(self.loss, {imput:[x1,x2]}) it is equivalent to do: [self.session.run(self.loss, {input: x}) for x in [x1,x2]]
        '''
        if _r == len(feed_dict.keys()):
            return self.sess.run(target, feed_dict)
        else:
            if type(list(feed_dict.values())[_r]) is list:
                _result = []
                for i in range(len(list(feed_dict.values())[_r])):
                    _temp_dict = feed_dict.copy()
                    _temp_dict[list(feed_dict.keys())[_r]] = list(feed_dict.values())[_r][i]
                    # change _r+1 to _r to support multi-level lists.
                    _result.append(self._zip_run(target, _temp_dict, _r+1))
                return _result
            else:
                return self._zip_run(target, feed_dict, _r+1)

    def test(self, dataset, **kwarg):
        '''Test the model on given dataset.
        Parameters
        ------
            dataset, DataSet object.
                the dataset.
            mode='te', str.
                on which part of dataset we test on. optional: ['tr', 'val', 'te']
            target='metric', str
                which tensor we test on. optional: ['metric', 'loss']
            batch_size=1, int.
                the batch size.

        Returns
        ------
            float,
                the test result.
        '''
        if 'mode' not in kwarg.keys():
            kwarg['mode'] = 'te'
        if 'target' not in kwarg.keys():
            kwarg['target'] = 'metric'
        if 'batch_size' not in kwarg.keys():
            kwarg['batch_size'] = 1
        if kwarg['batch_size'] > min(dataset.tr_batch_num, dataset.val_batch_num, dataset.te_batch_num):
            kwarg['batch_size'] = min(dataset.tr_batch_num, dataset.val_batch_num, dataset.te_batch_num)
            print('WARNING: batch_size should be smaller than dataset size. Automatically set batch_size =', kwarg['batch_size'])
        
        if kwarg['mode'] in ['tr', 'train', 'training']:
            batch_num = dataset.tr_batch_num
            get_batch = dataset.tr_get_batch
        elif kwarg['mode'] in ['val', 'validate', 'validation']:
            batch_num = dataset.val_batch_num
            get_batch = dataset.val_get_batch
        elif kwarg['mode'] in ['te', 'test', 'testing']:
            batch_num = dataset.te_batch_num
            get_batch = dataset.te_get_batch
        else:
            raise Exception('Unsupport test mode.')
        
        if kwarg['target']=='metric':
            target = self.metric
        elif kwarg['target']=='loss':
            target = [self.loss]
        else:
            raise Exception('Unsupport test target.')
        
        me_list = [[] for _ in range(len(target))]
        for _ in range(batch_num//kwarg['batch_size']):
            batch = get_batch(kwarg['batch_size'])
            if type(self.input) is list:
                feed_dict = {self.input[i]: batch[i] for i in range(len(self.input))}
                feed_dict.update({self.learning_rate: 0.0, self.training: False, self.training_process: 0.0})
            else:
                feed_dict = {self.learning_rate: 0.0, self.training: False, self.training_process: 0.0, self.input: batch}
            # ls = self.sess.run(self.loss, feed_dict)
            _re = self._zip_run(target, feed_dict)
            _re = _unzip_list(_re)
            for _i in range(len(target)):
                me = np.mean(_re[_i])
                me_list[_i].append(me)
        me_list = [np.mean(x) for x in me_list]

        return me_list

    def fit(self, dataset, **kwarg):
        '''Train the model on given dataset.

        This function will train the model on the training set, keep the best model on validation set, and report its result on testing set. This proceduremay repeat for statistical perfomence analysis.

        Parameters
        ------
            dataset, DataSet object.
                the dataset.
        Returns
        ------
            list of float.
                the final test result. list length equals to the repeat number.
            list of str.
                the model name list. list length equals to the repeat number.
        '''
        # parameter checking
        if 'model_name' not in kwarg.keys():
            kwarg['model_name'] = 'NewModel'
        if 'lr' not in kwarg.keys():
            if 'learning_rate' in kwarg.keys():
                kwarg['lr'] = kwarg['learning_rate']
            else:
                kwarg['lr'] = 1e-3
        if 'baseline' not in kwarg.keys():
            kwarg['baseline'] = 0
        if 'epoch' not in kwarg.keys():
            kwarg['epoch'] = 100
        if 'batch_size' not in kwarg.keys():
            kwarg['batch_size'] = 1
        if kwarg['batch_size'] > min(dataset.tr_batch_num, dataset.val_batch_num, dataset.te_batch_num):
            kwarg['batch_size'] = min(dataset.tr_batch_num, dataset.val_batch_num, dataset.te_batch_num)
            print('WARNING: batch_size should be smaller than dataset size. Automatically set batch_size =', kwarg['batch_size'])
        if 'print_type' not in kwarg.keys():
            kwarg['print_type'] = 'metric'
        if 'print_every_n_epochs' not in kwarg.keys():
            kwarg['print_every_n_epochs'] = 1
        if 'val_every_n_epochs' not in kwarg.keys():
            kwarg['val_every_n_epochs'] = 1
        if 'lr_annealing' not in kwarg.keys():
            kwarg['lr_annealing'] = 'constant'
        if 'lr_annealing_step_length' not in kwarg.keys():
            kwarg['lr_annealing_step_length'] = int(kwarg['epoch']/4)
        if 'lr_annealing_step_divisor' not in kwarg.keys():
            kwarg['lr_annealing_step_divisor'] = 2.0
        if 'early_stop' not in kwarg.keys():
            kwarg['early_stop'] = None
        if 'repeat' not in kwarg.keys():
            kwarg['repeat'] = 1
        if 'verbose' not in kwarg.keys():
            kwarg['verbose'] = 0
        if 'validate_on' not in kwarg.keys():
            kwarg['validate_on'] = 'val'
        if 'higher_better' not in kwarg.keys():
            kwarg['higher_better'] = False


        repeat_name_set = []
        repeat_metric_set = []
        repeat_time_set = []

        for _repeat in range(kwarg['repeat']):
            if not os.path.exists('model/'):
                os.mkdir('model/')
            while 1:
                version = time.strftime('%Y%m%d_%H%M%S')
                try:
                    model_path = 'model/' + kwarg['model_name'] + version
                    os.mkdir(model_path)
                    if self.summary_merged is not None:
                        self.summary_writer = tf.summary.FileWriter(model_path + '/log', self.sess.graph)
                        summary_counter = 0
                    break
                except FileExistsError:
                    time.sleep(1)           
            lr = float(kwarg['lr'])
            if kwarg['higher_better']:
                performance_recorder = -1e10
            else:
                performance_recorder = 1e10
            epoch_recorder = 1e10
            start_time = time.time()

            # Reset parameters for repeat
            if kwarg['repeat'] > 1:
                self.init_parameter(seed=_repeat)
                if kwarg['verbose'] < 1:
                    print('Repeat ', _repeat, 'Start.')

            # Start training
            for epoch in range(kwarg['epoch']):
                # update an epoch
                train_ls = []
                train_me = [[] for _ in range(len(self.metric))]
                for _ in range(dataset.tr_batch_num//kwarg['batch_size']):
                    batch = dataset.tr_get_batch(kwarg['batch_size'])
                    if type(self.input) is list:
                        feed_dict = {self.input[i]: batch[i] for i in range(len(self.input))}
                        feed_dict.update({self.learning_rate: lr, self.training: True,
                             self.training_process: float((epoch+1)/kwarg['epoch'])})
                    else:
                        feed_dict = {self.learning_rate: lr, self.training: True, self.training_process: float((epoch+1)/kwarg['epoch']), self.input: batch}
                    # _re is a list of list. _re[0] is list of self.loss; _re[1+k] is list of metric[k];
                    # _re[1+K] is list of summary record (if defined). 
                    _re = self.update(feed_dict)
                    ls = np.mean(_re[0])
                    train_ls.append(ls)
                    for _i in range(len(self.metric)):
                        me = np.mean(_re[1+_i])
                        train_me[_i].append(me)
                    if self.summary_merged is not None:
                        for x in _re[1+len(self.metric)]:
                            self.summary_writer.add_summary(x, summary_counter)
                            summary_counter += 1
                    
                # print log
                if (epoch+1)%kwarg['print_every_n_epochs'] == 0 and kwarg['verbose'] < 1:
                    train_ls = np.mean(train_ls)
                    train_me = [np.mean(m) for m in train_me]
                    if kwarg['print_type']=='metric':
                        for _i in range(len(self.metric)):
                            print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Done, Train '+self.metric_name[_i],round(train_me[_i],4))
                    elif kwarg['print_type']=='loss':
                        print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Done, Train loss ',round(train_ls,4))
                    else:
                        print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Done, Train loss ',round(train_ls,4))
                        for _i,_x in enumerate(train_me):
                            print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Done, Train '+self.metric_name[_i],round(_x,4))
                
                # learning_rate_decay
                if kwarg['lr_annealing']=='step':
                    if (epoch+1)%kwarg['lr_annealing_step_length'] == 0:
                            lr = lr/kwarg['lr_annealing_step_divisor']
                            if kwarg['verbose'] < 1:
                                print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],', Learning rate decay to ',lr)
                elif kwarg['lr_annealing']=='cosine':
                    lr = float(kwarg['lr'])*(np.cos(epoch/kwarg['epoch']*np.pi)+1.0)/2

                # validate
                if (epoch+1)%kwarg['val_every_n_epochs'] == 0:
                    # validate
                    val_me = self.test(dataset, mode=kwarg['validate_on'], target='metric', batch_size=kwarg['batch_size'])
                    if kwarg['verbose'] < 1:
                        for _i,_x in enumerate(val_me):
                            print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Done, Val '+self.metric_name[_i],round(_x,4))

                    # save model
                    target = val_me[0]
                    if (target < performance_recorder) ^ kwarg['higher_better']:
                        performance_recorder = target
                        epoch_recorder = epoch
                        self.saver.save(self.sess,'model/'+kwarg['model_name']+version+'/model')
                        if target < kwarg['baseline'] and kwarg['verbose'] < 1:
                            print('['+kwarg['model_name']+version+']epoch ',epoch,'/',kwarg['epoch'],' Model Save Success. New record ',target)

                    # Early stop
                    if kwarg['early_stop']:
                        if epoch >= epoch_recorder + kwarg['early_stop']:
                            if kwarg['verbose'] < 1:
                                print('Early stop.')
                            break
            
            # Final Test
            self.load('model/'+kwarg['model_name']+version+'/model')
            test_me = self.test(dataset, mode='te', target='metric', batch_size=kwarg['batch_size'])

            repeat_metric_set.append(test_me)
            repeat_name_set.append(kwarg['model_name']+version)
            repeat_time_set.append(time.time()-start_time)
            if kwarg['verbose'] < 2:
                print('=======Training Summary=======')
                print('Time used: ', time.strftime('%H:%M:%S',time.gmtime(time.time()-start_time)))
                if performance_recorder < 1e10:
                    print('Best model at epoch ', epoch_recorder, ', with:')
                    print('Validation metric ', performance_recorder)
                    print('Test metric ', test_me)
                if performance_recorder < kwarg['baseline']:
                    print('Best Model saved as: ', 'model/'+kwarg['model_name']+version+'/model')

            # Here ends a repeat.
        
        # Compute repeat summary
        if kwarg['repeat'] > 1:
            test_ls_mean = np.mean(repeat_metric_set, 0)
            test_ls_std = np.std(repeat_metric_set, 0)
            total_time = np.sum(repeat_time_set)
            if kwarg['verbose'] < 2:
                print('=======Repeat Summary=======')
                print('Repeat: ', kwarg['repeat'])
                print('Model list: ', repeat_name_set)
                print('Time used: ', time.strftime('%H:%M:%S',time.gmtime(total_time)))
                print('Test loss: ', repeat_metric_set)
                for _i in range(len(self.metric)):
                    print('Average '+self.metric_name[_i], test_ls_mean[_i])
                    print('95 confidence interval: ', test_ls_std[_i]*1.96, '(',test_ls_mean[_i]-test_ls_std[_i]*1.96,'~',test_ls_mean[_i]+test_ls_std[_i]*1.96,')')

        if kwarg['verbose'] < 1:
            print('Done.')            
        
        return repeat_metric_set, repeat_name_set
