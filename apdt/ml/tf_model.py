# pylint: disable=E1101
import tensorflow as tf
import numpy as np

def wavenet_weight(**kwarg):
    '''Construct weight for wavenet.
    Parameter
    ---------
        name: str
            the variable name scope to be used.
    Return
    ------
        dict
            the dictionary of weight tensor.
    '''
    if 'name' not in kwarg:
        kwarg['name'] = 'wavenet'
    if 'res_channel' not in kwarg:
        kwarg['res_channel'] = 32
    if 'skip_channel' not in kwarg:
        kwarg['skip_channel'] = 16
    if 'input_dim' not in kwarg:
        kwarg['input_dim'] = 1
    if 'DilatedConvLayers' not in kwarg:
        kwarg['DilatedConvLayers'] = 4
    if 'n_hidden' not in kwarg:
        kwarg['n_hidden'] = 128
    if 'trainable' not in kwarg:
        kwarg['trainable'] = True
    if 'share_param_between_layers' not in kwarg:
        kwarg['share_param_between_layers'] = False

    with tf.variable_scope(kwarg['name']):
        res_channel=kwarg['res_channel']
        skip_channel=kwarg['skip_channel']
        bits = 5
        w = {}
        if kwarg['share_param_between_layers']:
            k = 0
            w['kernel_l'+str(k)] = tf.get_variable('kernel_l'+str(k),[3, res_channel, 2*res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['1_by_1_res_l'+str(k)] = tf.get_variable('1_by_1_res_l'+str(k),[1, res_channel, res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['1_by_1_skip_l'+str(k)] = tf.get_variable('1_by_1_skip_l'+str(k),[1, res_channel, skip_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            for k in range(1, kwarg['DilatedConvLayers']):
                w['kernel_l'+str(k)] = w['kernel_l'+str(0)]
                w['1_by_1_res_l'+str(k)] = w['1_by_1_res_l'+str(0)]
                w['1_by_1_skip_l'+str(k)] = w['1_by_1_skip_l'+str(0)]
        else:
            for k in range(0, kwarg['DilatedConvLayers']):
                w['kernel_l'+str(k)] = tf.get_variable('kernel_l'+str(k),[3, res_channel, 2*res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
                w['1_by_1_res_l'+str(k)] = tf.get_variable('1_by_1_res_l'+str(k),[1, res_channel, res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
                w['1_by_1_skip_l'+str(k)] = tf.get_variable('1_by_1_skip_l'+str(k),[1, res_channel, skip_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['1_by_1_x'] = tf.get_variable('1_by_1_x', [1, kwarg['input_dim'], res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['1_by_1_skip1'] = tf.get_variable(
                    '1_by_1_skip1', [1, skip_channel*kwarg['DilatedConvLayers']+kwarg['input_dim'], kwarg['n_hidden']],
                    initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['1_by_1_skip2'] = tf.get_variable(
                    '1_by_1_skip2', [1, kwarg['n_hidden'], 2**bits],
                    initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        return w

def WaveNet(input, weights, name, **kwarg):
    '''Map a time series into another using WaveNet.
    Parameter
    ---------
        input: tensor
            a tensor shaped [batch_size, time_length, feature_dim]
        weights: dict
            a dictionary of weights generated by apdt.ml.wavenet_weight
        name:
            the variable name scope to be used.
    Return
    ------
        tensor
            The corresponding prediction series. a tensor shaped [batch_size, time_length, 1]
        tensor
            The loss function.
    '''
    if 'res_channel' not in kwarg.keys():
        kwarg['res_channel'] = 32
    if 'skip_channel' not in kwarg.keys():
        kwarg['skip_channel'] = 16
    if 'input_dim' not in kwarg.keys():
        kwarg['input_dim'] = 1
    if 'DilatedConvLayers' not in kwarg.keys():
        kwarg['DilatedConvLayers'] = 4
    if 'n_hidden' not in kwarg.keys():
        kwarg['n_hidden'] = 128
    if 'dilated' not in kwarg.keys():
        kwarg['dilated'] = 3
    if 'loss' not in kwarg.keys():
        kwarg['loss'] = 'MAE'

    def CasualResUnit(x,k,res_channel=kwarg['res_channel'],skip_channel=kwarg['skip_channel'],name='CasualResUnit'):
        k = int(np.log(k)/np.log(kwarg['dilated']))
        w = weights['kernel_l'+str(k)]
        _x = tf.pad(x,[[0,0],[2,0],[0,0]])
        y = tf.nn.conv1d(_x, w, 1, 'VALID')
        h,g = tf.split(y, 2, axis=-1)
        h = tf.tanh(h)
        g = tf.sigmoid(g)
        h = h*g
        w = weights['1_by_1_res_l'+str(k)]
        o = tf.nn.conv1d(h, w, 1, 'SAME')
        o = o+x
        w = weights['1_by_1_skip_l'+str(k)]
        skip = tf.nn.conv1d(h, w, 1, 'SAME')
        return o,skip
    
    def DilatedConv(x,dilated=kwarg['dilated'],res_channel=kwarg['res_channel'],skip_channel=kwarg['skip_channel'],name='DilatedConv'):
        n = x.shape[-1].value
        l = x.shape[1].value
        if l%dilated != 0:
            print('Dilated Error, when dilated at '+str(dilated))
            exit()
        num = l//dilated
        with tf.variable_scope(name) as scope:
            x = tf.reshape(x,[-1,num,dilated,n])
            _out = []
            _skip = []
            for i in range(dilated):
                out, skip = CasualResUnit(x[:,:,i],dilated,res_channel=res_channel,skip_channel=skip_channel,name='CasualResUnit#'+str(i))
                _out.append(out)
                _skip.append(skip)
                if i==0:
                    scope.reuse_variables()
            o = tf.stack(_out,axis=2)
            o = tf.reshape(o,[-1,l,n])
            skip = tf.stack(_skip,axis=2)
            skip = tf.reshape(skip,[-1,l,skip_channel])
            return o,skip
    
    def u_law_encoder(x,bits=5):
        bits = 2**bits
        x = tf.clip_by_value(x,0.0,0.99999)
        x = tf.floor(tf.log(1+bits*x)*bits/np.log(1+bits))
        x = tf.one_hot(tf.cast(tf.squeeze(x,-1),tf.int32),depth=bits)
        return x
    
    def u_law_decoder(x,bits=5):
        x = tf.cast(x,tf.float32)
        x = (tf.exp((x+0.5)/(2**bits)*np.log(1.0+(2**bits)))-1.0)/(2**bits)
        return tf.expand_dims(x,-1)
    
    def SingleChannelNetwork(x,channel,bits,encoder,decoder,name='Channel'):
        with tf.variable_scope(name):
            y = x[:,1:,channel:channel+1]
            x = x[:,:-1,:]
            _skip = [x]
            w = weights['1_by_1_x']
            x = tf.nn.conv1d(x, w, 1, 'SAME')
            for i in range(kwarg['DilatedConvLayers']):
                if i==0:
                    o,skip = DilatedConv(x,dilated=3**i,name='DilatedConv'+str(i))
                    _skip.append(skip)
                else:
                    o,skip = DilatedConv(o,dilated=3**i,name='DilatedConv'+str(i))
                    _skip.append(skip)
            skip = tf.concat(_skip,axis=-1)
            skip = tf.nn.relu(skip)
            w = weights['1_by_1_skip1']
            skip = tf.nn.conv1d(skip, w, 1, 'SAME')
            skip = tf.nn.relu(skip)
            w = weights['1_by_1_skip2']
            pred = tf.nn.conv1d(skip, w, 1, 'SAME')
            pred_prob = tf.nn.softmax(pred)
            pred = tf.nn.conv1d(pred_prob, decoder(tf.constant(np.arange(2**bits).reshape(1,2**bits))), 1, 'SAME')
            if kwarg['loss']=='MAE':
                loss = tf.reduce_mean(tf.abs(pred-y))
            elif kwarg['loss']=='RMSE':
                loss = tf.sqrt(tf.reduce_mean(tf.square(pred-y)))
            return pred,loss
    
    pred,loss = SingleChannelNetwork(input,0,5,u_law_encoder,u_law_decoder,name=name)
    return pred,loss
pass