# pylint: disable=E1101
import numpy as np

try:
    import tensorflow as tf
except:
    pass

def Transformer_ar_decoder_weight(**kwarg):
    '''Construct weight for Transformer.
    Parameter
    ---------
        name: str
            the variable name scope to be used.
        input_dim: int, default 10
            the number of input neurals
        output_dim: int, default 10
            the number of output neurals
        n_hidden: list of int, default 128
            the number of hidden units
        n_layers: list of int, default 3
            the number of multihead-attention + dense blocks
        trainable: bool, default True
            if these parameters are trainable.
    Return
    ------
        dict
            the dictionary of weight tensor.
    '''
    if 'name' not in kwarg:
        kwarg['name'] = 'Transformer_ar_decoder_weight'
    if 'input_dim' not in kwarg:
        kwarg['input_dim'] = 10
    if 'output_dim' not in kwarg:
        kwarg['output_dim'] = 10
    if 'n_hidden' not in kwarg:
        kwarg['n_hidden'] = 128
    if 'n_layers' not in kwarg:
        kwarg['n_layers'] = 3
    if 'trainable' not in kwarg:
        kwarg['trainable'] = True

    w = {}
    with tf.variable_scope(kwarg['name']):
        w['input_project_w'] = tf.get_variable('input_project_w', [1, kwarg['input_dim'], kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['input_project_b'] = tf.get_variable('input_project_b', [kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        for i in range(kwarg['n_layers']):
            w['w_q_'+str(i)] = tf.get_variable('w_q_'+str(i), [1, kwarg['n_hidden'], kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['b_q_'+str(i)] = tf.get_variable('b_q_'+str(i), [kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['w_k_'+str(i)] = tf.get_variable('w_k_'+str(i), [1, kwarg['n_hidden'], kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['b_k_'+str(i)] = tf.get_variable('b_k_'+str(i), [kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['w_v_'+str(i)] = tf.get_variable('w_v_'+str(i), [1, kwarg['n_hidden'], kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['b_v_'+str(i)] = tf.get_variable('b_v_'+str(i), [kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['dense_w1_'+str(i)] = tf.get_variable('dense_w1_'+str(i), [1, kwarg['n_hidden'], kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['dense_b1_'+str(i)] = tf.get_variable('dense_b1_'+str(i), [kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['dense_w2_'+str(i)] = tf.get_variable('dense_w2_'+str(i), [1, kwarg['n_hidden'], kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['dense_b2_'+str(i)] = tf.get_variable('dense_b2_'+str(i), [kwarg['n_hidden']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['out_w'] = tf.get_variable('out_w', [1, kwarg['n_hidden'], kwarg['output_dim']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['out_b'] = tf.get_variable('out_b', [kwarg['output_dim']], 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
    return w

def scaled_dot_product_attention(Q, K, V, causality=False,
                                 name="scaled_dot_product_attention"):
    '''Compute dot production attention. H = softmax(QK^T/sqrt(d_k))V
    Parameter
    ---------
        Q: tensor
            Packed queries. 3d tensor. [N, T_q, d_k].
        K: tensor
            Packed keys. 3d tensor. [N, T_k, d_k].
        V: tensor
            Packed values. 3d tensor. [N, T_k, d_v].
        causality: bool, default False
            If True, applies masking for future blinding. A query wont use keys and values after its index.
        name: str
            the variable name scope.
    Return
    ------
        tensor
            attention result. 3d tensor. [N, T_q, d_v]. 
    '''
    with tf.variable_scope(name):
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
        outputs /= (Q.shape[-1].value) ** 0.5
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            outputs = outputs * tril - 1e10 * (1 - tril)
        outputs = tf.nn.softmax(outputs)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
    return outputs

def multihead_attention(Q, K, V, w_q, w_k, w_v,
                                 b_q, b_k, b_v,
                                 causality=False,
                                 num_head=1,
                                 name="multihead_attention"):
    '''Compute multi-head attention.
    Parameter
    ---------
        Q: tensor
            Packed queries. 3d tensor. [N, T_q, d_k].
        K: tensor
            Packed keys. 3d tensor. [N, T_k, d_k].
        V: tensor
            Packed values. 3d tensor. [N, T_k, d_v].
        w_q, w_k, w_v: tensor
            weight for linear projection. 3d tensor. [1, d_k, d_k] (d_v for w_v). the first dimension 1 is not ignorable because we use conv1D for dense mapping.
        b_q, b_k, b_v: tensor
            bias for linear projection. 1d tensor. [d_k] (d_v for w_v).
        causality: bool, default False
            If True, applies masking for future blinding. A query wont use keys and values after its index.
        num_head: int, default 1
            The number of heads. d_k and d_v should be exact divided by num_head.
        name: str
            the variable name scope.
    Return
    ------
        tensor
            attention result. 3d tensor. [N, T_q, d_v]. 
    '''
    with tf.variable_scope(name):
        Q_ = tf.nn.conv1d(Q, w_q, 1, 'SAME') + b_q
        K_ = tf.nn.conv1d(K, w_k, 1, 'SAME') + b_k
        V_ = tf.nn.conv1d(V, w_v, 1, 'SAME') + b_v
        Q_ = tf.concat(tf.split(Q_, num_head, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K_, num_head, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V_, num_head, axis=2), axis=0) # (h*N, T_k, d_model/h)
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality=causality) # (h*N, T_q, d_model/h)
        outputs = tf.concat(tf.split(outputs, num_head, axis=0), axis=2) # (N, T_q, d_model)
        outputs += Q
    return outputs

def positional_encoding(inputs, scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5 in paper Attention is all you need.
    Parameter
    ---------
        inputs: tensor
            the feature tensor to be encoded. 3d tensor. (N, T, E)
        scope: str
            Optional scope for `variable_scope`.
    Return
    -------
        tensor
            3d tensor that has the same shape as inputs.
    '''
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    maxlen = inputs.shape[1].value
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        return tf.to_float(outputs)

def Transformer_ar_decoder(input, weights=None, name='Transformer_ar_decoder', **kwarg):
    '''Apply Transformer on a single series auto-regressively.
    Parameter
    ---------
        inputs: tensor
            the feature tensor to be encoded. 3d tensor. (N, T, D)
        weights: dict, default None
            the network weights generated by Transformer_ar_decoder_weight(). If given None, 
            network structure have to be clearified and corresponding weights will be newly generated.
        n_hidden: int, default 128
            the feature dimension in the model. If weights is not None, this is ignored.
        n_layers: int, default 3
            the number of multihead-attention + dense blocks. If weights is not None, this is ignored.
        n_head: int, default 1
            the number of heads in multihead-attention. If weights is not None, this is ignored.
        output_dim: int, default 128
            the dimension of output. If weights is not None, this is ignored.
        positional_encoding: bool, default True
            if True, the positional_encoding given by Sec.3.5 in paper Attention is all you need will be used.
        name: str
            Optional scope for `variable_scope`.
    Return
    -------
        tensor
            3d tensor that has the same shape as inputs.
    ''' 

    if 'n_hidden' not in kwarg:
        kwarg['n_hidden'] = 128
    if 'n_layers' not in kwarg:
        kwarg['n_layers'] = 3
    if 'n_head' not in kwarg:
        kwarg['n_head'] = 4
    if 'output_dim' not in kwarg:
        kwarg['output_dim'] = 128
    if 'positional_encoding' not in kwarg:
        kwarg['positional_encoding'] = True

    if weights is None:
        weights = Transformer_ar_decoder_weight(name=name, input_dim=input.shape[2].value, n_hidden=kwarg['n_hidden'], n_layers=kwarg['n_layers'], output_dim=kwarg['output_dim'])
    else:
        kwarg['n_layers'] = len([0 for x in weights.keys() if x.startswith('w_q_')])

    with tf.variable_scope(name):
        if kwarg['positional_encoding']:
            input = input + positional_encoding(input)
        x = tf.nn.conv1d(input, weights['input_project_w'], 1, 'SAME') + weights['input_project_b']
        for i in range(kwarg['n_layers']):
            h = multihead_attention(x, x, x,
                                    weights['w_q_'+str(i)], weights['w_k_'+str(i)], weights['w_v_'+str(i)],
                                    weights['b_q_'+str(i)], weights['b_k_'+str(i)], weights['b_v_'+str(i)],
                                    causality=True, num_head=kwarg['n_head'], name="multihead_attention"+str(i))
            x = tf.nn.conv1d(h, weights['dense_w1_'+str(i)], 1, 'SAME') + weights['dense_b1_'+str(i)]
            x = tf.nn.relu(x)
            x = tf.nn.conv1d(x, weights['dense_w2_'+str(i)], 1, 'SAME') + weights['dense_b2_'+str(i)]
            x = x + h
        o = tf.nn.conv1d(x, weights['out_w'], 1, 'SAME') + weights['out_b']
    return o

def lstm_weight(**kwarg):
    '''Construct weight for LSTM.
    Parameter
    ---------
        name: str
            the variable name scope to be used.
        input_dim: int, default 10
            the number of input neurals
        output_dim: int, default 10
            the number of output neurals
        n_hidden: list of int, default 128
            the number of hidden units
    Return
    ------
        dict
            the dictionary of weight tensor.
    '''
    if 'name' not in kwarg:
        kwarg['name'] = 'lstm'
    if 'input_dim' not in kwarg:
        kwarg['input_dim'] = 10
    if 'output_dim' not in kwarg:
        kwarg['output_dim'] = 10
    if 'n_hidden' not in kwarg:
        kwarg['n_hidden'] = 128
    if 'trainable' not in kwarg:
        kwarg['trainable'] = True
    
    w = {}
    with tf.variable_scope(kwarg['name']):
        w['weight_r'] = tf.get_variable('weight_r',[2*kwarg['n_hidden'], kwarg['n_hidden']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['bias_r'] = tf.get_variable('bias_r',[kwarg['n_hidden']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['weight_u'] = tf.get_variable('weight_u',[2*kwarg['n_hidden'], kwarg['n_hidden']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['bias_u'] = tf.get_variable('bias_u',[kwarg['n_hidden']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['weight_c'] = tf.get_variable('weight_c',[2*kwarg['n_hidden'], kwarg['n_hidden']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['bias_c'] = tf.get_variable('bias_c',[kwarg['n_hidden']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['input_embedding'] = tf.get_variable('input_embedding',[1, kwarg['input_dim'], kwarg['n_hidden']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['output_transform'] = tf.get_variable('output_transform',[1, kwarg['n_hidden'], kwarg['output_dim']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
    
    return w

def basic_lstm_cell(input, h, weight, name):
    with tf.variable_scope(name):
        x = tf.concat([input,h],axis=-1)
        r = tf.sigmoid(tf.matmul(x, weight['weight_r']) + weight['bias_r'])
        u = tf.sigmoid(tf.matmul(x, weight['weight_u']) + weight['bias_u'])
        x = tf.concat([input,h*r],axis=-1)
        c = tf.sigmoid(tf.matmul(x, weight['weight_c']) + weight['bias_c'])
        h_new = u*h + (1-u)*c
    return c,h_new

def LSTM(input, state=None, output_dim=None, n_hidden=None, weights=None, name='lstm', **kwarg):
    '''Map a time series into another using LSTM.

    parameter set (output_dim, n_hidden) will be inferenced from weights if it exists. However, in the case you have modified weights from mlp_weight (e.g. combine two weights together) things may comes wrong. Please also give (output_dim, n_hidden) in such case.

    Parameter
    ---------
        input: tensor
            a tensor shaped [batch_size, seq_length, feature_dim]
        state: tensor
            a tensor shaped [batch_size, n_hidden]
        output_dim: int
            the dimension of output vector.
        n_hidden: int
            the number of hidden neural in cell.
        weights: dict
            a dictionary of weights generated by apdt.ml.lstm_weight
    Return
    ------
        tensor
            The output vector, shaped [batch_size, seq_length, feature_dim_output]
    Note
    ----
        multi layer lstm is not supported now.
        customer rnn cell is not supported now.
        bidirectional lstm is not supported now.
    '''
    N = input.shape[0].value
    T = input.shape[1].value
    C = input.shape[2].value

    if weights is None:
        if output_dim is None:
            output_dim = 10
        if n_hidden is None:
            n_hidden = 128
        weights = lstm_weight(name=name,input_dim=C,output_dim=output_dim,n_hidden=n_hidden)
    else:
        if n_hidden is None:
            n_hidden = weights['bias_r'].shape[0].value

    if state is None:
        h = tf.zeros([N, n_hidden])
    else:
        h = state

    if 'activation' not in kwarg:
        kwarg['activation'] = tf.nn.relu
    if 'dropout' not in kwarg:
        kwarg['dropout'] = False
    if 'dropout_ratio' not in kwarg:
        kwarg['dropout_ratio'] = 0.5
    
    with tf.variable_scope(name):
        input_embedding = tf.nn.conv1d(input, weights['input_embedding'], 1, 'SAME')
        o_list = []
        for time_step in range(T):
            o, h = basic_lstm_cell(input_embedding[:,time_step], h, weights, name)
            o_list.append(o)
        pred = tf.stack(o_list, 1)
        pred = tf.nn.conv1d(pred, weights['output_transform'], 1, 'SAME')

    return pred, h

def mlp_weight(**kwarg):
    '''Construct weight for MLP.
    Parameter
    ---------
        name: str
            the variable name scope to be used.
        input_dim: int, default 10
            the number of input neurals
        output_dim: int, default 10
            the number of output neurals
        n_hidden: list of int, default [128,128]
            the number of hidden units, the length of list decide the number of layers.
    Return
    ------
        dict
            the dictionary of weight tensor.
    '''
    if 'name' not in kwarg:
        kwarg['name'] = 'mlp'
    if 'input_dim' not in kwarg:
        kwarg['input_dim'] = 10
    if 'output_dim' not in kwarg:
        kwarg['output_dim'] = 10
    if 'n_hidden' not in kwarg:
        kwarg['n_hidden'] = [128, 128]
    if 'trainable' not in kwarg:
        kwarg['trainable'] = True
    
    w = {}
    with tf.variable_scope(kwarg['name']):
        hn = kwarg['input_dim']
        for i in range(len(kwarg['n_hidden'])):
            w['weight'+str(i)] = tf.get_variable('weight'+str(i),[hn, kwarg['n_hidden'][i]],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['bias'+str(i)] = tf.get_variable('bias'+str(i),[kwarg['n_hidden'][i],],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            hn = kwarg['n_hidden'][i]
        w['weightout'] = tf.get_variable('weightout',[hn, kwarg['output_dim']],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['biasout'] = tf.get_variable('biasout',[kwarg['output_dim'],],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
    
    return w

def MLP(input, n_layers=None, output_dim=None, n_hidden=None, weights=None, name='mlp', **kwarg):
    '''Map a vector into another using MLP.

    parameter set (n_layers, output_dim, n_hidden) will be inferenced from weights if it exists. However, in the case you have modified weights from mlp_weight (e.g. combine two weights together) things may comes wrong. Please also give (n_layers, output_dim, n_hidden).

    Parameter
    ---------
        input: tensor
            a tensor shaped [batch_size, feature_dim]
        n_layers: int
            the number of hidden layers.
        output_dim: int
            the dimension of output vector.
        n_hidden: list of int
            the number of hidden neural in each layer.
        weights: dict
            a dictionary of weights generated by apdt.ml.mlp_weight
    Return
    ------
        tensor
            The output vector, shaped [batch_size, feature_dim_output]
    Note
    ----
        Dropout is still not available now, because it shoule support automatic switch with training/testing condition.
        BN is not supported now.
    '''
    
    if weights is None:
        if n_layers is None:
            n_layers = 2
        if output_dim is None:
            output_dim = 10
        if n_hidden is None:
            n_hidden = [128]*n_layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden]*n_layers
        weights = mlp_weight(name=name,input_dim=input.shape[-1].value,output_dim=output_dim,n_hidden=n_hidden)
    else:
        if n_layers is None:
            n_layers = len(weights.keys())//2 - 1
        if output_dim is None:
            output_dim = weights['biasout'].shape[0].value

    if 'activation' not in kwarg:
        kwarg['activation'] = tf.nn.relu
    if 'dropout' not in kwarg:
        kwarg['dropout'] = False
    if 'dropout_ratio' not in kwarg:
        kwarg['dropout_ratio'] = 0.5
    
    x = input
    for i in range(n_layers):
        x = tf.tensordot(x, weights['weight'+str(i)], 1) + weights['bias'+str(i)]
        x = kwarg['activation'](x)
        if kwarg['dropout']:
            x = tf.nn.dropout(x, kwarg['dropout_ratio'])
    x = tf.tensordot(x, weights['weightout'], 1) + weights['biasout']
    return x
    
    

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
    if 'kernel_width' not in kwarg:
        kwarg['kernel_width'] = 3
    if 'output_channel' not in kwarg:
        kwarg['output_channel'] = 32

    with tf.variable_scope(kwarg['name']):
        res_channel = kwarg['res_channel']
        skip_channel = kwarg['skip_channel']
        output_channel = kwarg['output_channel']
        w = {}
        if kwarg['share_param_between_layers']:
            k = 0
            w['kernel_l'+str(k)] = tf.get_variable('kernel_l'+str(k),[kwarg['kernel_width'], res_channel, 2*res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['1_by_1_res_l'+str(k)] = tf.get_variable('1_by_1_res_l'+str(k),[1, res_channel, res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            w['1_by_1_skip_l'+str(k)] = tf.get_variable('1_by_1_skip_l'+str(k),[1, res_channel, skip_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
            for k in range(1, kwarg['DilatedConvLayers']):
                w['kernel_l'+str(k)] = w['kernel_l'+str(0)]
                w['1_by_1_res_l'+str(k)] = w['1_by_1_res_l'+str(0)]
                w['1_by_1_skip_l'+str(k)] = w['1_by_1_skip_l'+str(0)]
        else:
            for k in range(0, kwarg['DilatedConvLayers']):
                w['kernel_l'+str(k)] = tf.get_variable('kernel_l'+str(k),[kwarg['kernel_width'], res_channel, 2*res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
                w['1_by_1_res_l'+str(k)] = tf.get_variable('1_by_1_res_l'+str(k),[1, res_channel, res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
                w['1_by_1_skip_l'+str(k)] = tf.get_variable('1_by_1_skip_l'+str(k),[1, res_channel, skip_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['1_by_1_x'] = tf.get_variable('1_by_1_x', [1, kwarg['input_dim'], res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['1_by_1_skip1'] = tf.get_variable(
                    '1_by_1_skip1', [1, skip_channel*kwarg['DilatedConvLayers']+kwarg['input_dim'], kwarg['n_hidden']],
                    initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        w['1_by_1_skip2'] = tf.get_variable(
                    '1_by_1_skip2', [1, kwarg['n_hidden'], output_channel],
                    initializer=tf.contrib.layers.xavier_initializer(), trainable=kwarg['trainable'])
        return w

def _WaveNet(input, weights, name, **kwarg):
    '''Map a time series into another using WaveNet.

    This implementation has been depracted and will be removed in future version. Please check WaveNet().

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
    if 'bits' not in kwarg.keys():
        kwarg['bits'] = 5
    if 'return_type' not in kwarg.keys():
        kwarg['return_type'] = 'pred+loss'

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
            skip_raw = skip
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
            return pred,loss,skip_raw
    
    pred,loss,skip_raw = SingleChannelNetwork(input,0,kwarg['bits'],u_law_encoder,u_law_decoder,name=name)
    if kwarg['return_type'] == 'pred+loss':
        return pred,loss
    elif kwarg['return_type'] == 'feature':
        return skip_raw

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

def WaveNet(input, weights=None, name='WaveNet', **kwarg):
    '''Causally map a time series into another using WaveNet.
    Parameter
    ---------
        input: tensor
            a tensor shaped [batch_size, time_length, feature_dim]
        weights: dict, default None
            a dictionary of weights generated by apdt.ml.wavenet_weight. If given None, network parameters should be further clearified.
        name: str, default 'WaveNet'
            the variable name scope to be used.
        input_dim: int, default input.shape[-1]
            the dimension of input feature.
        res_channel: int, default 32
            the dimension of residual connection. More channel leads to better fit capability.
        skip_channel: int, default 16
            the dimension of skip connection. More channel leads to better fit capability.
        DilatedConvLayers: int, default 4
            the number of conv layers. More layer leads to larger receptive fields and better fit capability. It is highly suggested that input series length should far larger than receptive fields (=dilated**DilatedConvLayers)
        n_hidden: int, default 128
            the dimension of mlp hidden layers. More channel leads to better fit capability.
        dilated: int, default 3
            the dilated rate and conv kernel width. Larger number leads to more parameters and better fit capability.
        loss: str, optional: 'MAE' (default), 'RMSE'
            the loss function type.
        valid_loss: bool, default False
            if True, only predictions with full observations in receptive fields will be used to calculate loss.
        bits: int, default 5
            the u-law quatilization bit number. More bits leads to finner predcition resolution.
        return_type: str, optional 'pred+loss' (default), 'feature'
            the return type of function. 'feature' stands for raw causal features, 'pred' will further map it into a real number with mlp and u-law qutilization. 'loss' standis for the loss.
        batch_norm: bool, default False
            if set True, a batch normalization layer will be add into conv kernel.
        training: bool tensor
            required only when batch_norm is set True. Typically you can use training=self.training.

    Return
    ------
        tensor 'pred'
            The corresponding prediction series. a tensor shaped [batch_size, time_length, 1]
        tensor 'loss'
            The loss value.
        tensor 'feature'
            the raw causal features. a tensor shaped [batch_size, time_length, skip_channel*DilatedConvLayers+input_dim]
    '''
    if 'res_channel' not in kwarg.keys():
        kwarg['res_channel'] = 32
    if 'skip_channel' not in kwarg.keys():
        kwarg['skip_channel'] = 16
    if 'input_dim' not in kwarg.keys():
        try:
            kwarg['input_dim'] = input.shape[-1].value
        except:
            kwarg['input_dim'] = 1
    if 'DilatedConvLayers' not in kwarg.keys():
        kwarg['DilatedConvLayers'] = 4
    if 'n_hidden' not in kwarg.keys():
        kwarg['n_hidden'] = 128
    if 'dilated' not in kwarg.keys():
        kwarg['dilated'] = 3
    if 'kernel_width' not in kwarg.keys():
        kwarg['kernel_width'] = 3
    if 'loss' not in kwarg.keys():
        kwarg['loss'] = 'MAE'
    if 'bits' not in kwarg.keys():
        kwarg['bits'] = 5
    if 'return_type' not in kwarg.keys():
        kwarg['return_type'] = 'pred+loss'
    if 'batch_norm' not in kwarg.keys():
        kwarg['batch_norm'] = False
    if kwarg['batch_norm'] and 'training' not in kwarg.keys():
        Exception('training parameters is required when batch norm is enabled.')
    if 'valid_loss' not in kwarg.keys():
        kwarg['valid_loss'] = False
    
    if weights is None:
        weights = wavenet_weight(name=name, res_channel=kwarg['res_channel'], skip_channel=kwarg['skip_channel'],
            input_dim=kwarg['input_dim'], DilatedConvLayers=kwarg['DilatedConvLayers'], n_hidden=kwarg['n_hidden'],
            kernel_width=kwarg['kernel_width'],output_channel=2**kwarg['bits'])
    else:
        # Code to auto decision of parameters like kwarg['DilatedConvLayers']
        ...

    def SingleChannelNetwork(x,channel,bits,encoder,decoder,name='Channel'):
        with tf.variable_scope(name):
            # x_input = x
            y = x[:,1:,channel:channel+1]
            x = x[:,:-1,:]
            _skip = [x]
            w = weights['1_by_1_x']
            x = tf.nn.conv1d(x, w, 1, 'SAME')
            for i in range(kwarg['DilatedConvLayers']):
                _x = tf.pad(x,[[0,0],[kwarg['dilated']**i * (kwarg['kernel_width'] - 1),0],[0,0]])
                w = weights['kernel_l'+str(i)]
                h = tf.nn.conv2d(tf.expand_dims(_x,1), tf.expand_dims(w,0), [1,1,1,1], 'VALID', dilations=[1,1,kwarg['dilated']**i,1])
                if kwarg['batch_norm']:
                    h = tf.layers.batch_normalization(h, training=kwarg['training'])
                h,g = tf.split(h[:,0], 2, axis=-1)
                h = tf.tanh(h)
                g = tf.sigmoid(g)
                h = h*g
                w = weights['1_by_1_res_l'+str(i)]
                o = tf.nn.conv1d(h, w, 1, 'SAME')
                x = o+x
                w = weights['1_by_1_skip_l'+str(i)]
                skip = tf.nn.conv1d(h, w, 1, 'SAME')
                _skip.append(skip)

            skip = tf.concat(_skip,axis=-1)
            skip = tf.nn.relu(skip)
            skip_raw = skip
            w = weights['1_by_1_skip1']
            skip = tf.nn.conv1d(skip, w, 1, 'SAME')
            skip = tf.nn.relu(skip)
            w = weights['1_by_1_skip2']
            pred = tf.nn.conv1d(skip, w, 1, 'SAME')
            pred_prob = tf.nn.softmax(pred)
            pred = tf.nn.conv1d(pred_prob, decoder(tf.constant(np.arange(2**bits).reshape(1,2**bits))), 1, 'SAME')
            if not kwarg['valid_loss']:
                shift = 0
            else:
                shift = kwarg['dilated'] ** kwarg['DilatedConvLayers']
            # residual = pred[:, shift:-1, channel] - x_input[:, 1+shift:, channel]
            residual = pred - y
            if kwarg['loss']=='MAE':
                loss = tf.reduce_mean(tf.abs(residual))
            elif kwarg['loss']=='RMSE':
                loss = tf.sqrt(tf.reduce_mean(tf.square(residual)))
            return pred,loss,skip_raw
    
    pred,loss,skip_raw = SingleChannelNetwork(input,0,kwarg['bits'],u_law_encoder,u_law_decoder,name=name)
    if kwarg['return_type'] == 'pred+loss':
        return pred,loss
    elif kwarg['return_type'] == 'feature':
        return skip_raw
pass