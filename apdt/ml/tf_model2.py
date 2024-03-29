# pylint: disable=E1101
import numpy as np
from apdt.ml.tf_model import mlp_weight, MLP

try:
    import tensorflow as tf
except:
    pass

def deepsets_weight(**kwarg):
    '''Construct weight for deepsets.

    Parameter
    ---------
        name: str
            the variable name scope to be used.
        input_dim: int, default 10
            the number of input neurals
        output_dim: int, default 10
            the number of output neurals
        n_hidden: list of int, default 128
            the dimension of hidden states.
        mlp_n_hidden: list of int, default [128,128]
            the number of mlp hidden units, the length of list decide the number of layers.
    
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
        kwarg['n_hidden'] = 128
    if 'mlp_n_hidden' not in kwarg:
        kwarg['mlp_n_hidden'] = [128, 128]   
    if 'trainable' not in kwarg:
        kwarg['trainable'] = True

    w = {}
    with tf.variable_scope(kwarg['name']):
        mlp_weight_encoder = mlp_weight(name="encoder_mlp", input_dim=kwarg['input_dim'], output_dim=kwarg['n_hidden'],
                                        n_hidden=kwarg['mlp_n_hidden'],trainable=kwarg['trainable'])
        mlp_weight_decoder = mlp_weight(name="decoder_mlp", input_dim=kwarg['n_hidden'], output_dim=kwarg['output_dim'],
                                        n_hidden=kwarg['mlp_n_hidden'],trainable=kwarg['trainable'])
    w.update({'pf0'+key: value for key, value in mlp_weight_encoder.items()})
    w.update({'pf1'+key: value for key, value in mlp_weight_decoder.items()})

    return w

def deepsets(x, axis, weights=None, name='deepsets', **kwarg):
    '''Map a set into one feature using deepsets.

    Parameter
    ---------
        x: tensor
            the input tensor
        axis: int
            along which axis is the set going
        weights: dict
            a dictionary of weights generated by apdt.ml.deepsets_weight
    
    Return
    ------
        tensor
            The output tensor
    
    '''
    
    if weights is None:
        if "output_dim" not in kwarg:
            raise Exception("output_dim should be given if weights is None.")
        weights = deepsets_weight(name=name, input_dim=x.shape[-1].value, output_dim=kwarg["output_dim"])
    
    mlp_weight_encoder = {key[3:]: value for key, value in weights.items() if key.startswith('pf0')}
    mlp_weight_decoder = {key[3:]: value for key, value in weights.items() if key.startswith('pf1')}
    with tf.variable_scope(name):
        a = MLP(x, weights=mlp_weight_encoder, name="encoder_mlp")
        h = tf.reduce_mean(a, axis)
        z = MLP(h, weights=mlp_weight_decoder, name="decoder_mlp")

    return z

def time2vec_weight(**kwarg):
    '''Construct weight for time2vec block.

    Parameter
    ---------
        name: str
            the variable name scope to be used.
        input_dim: int, default 1
            the number of input neurals
        output_dim: int, default 16
            the number of output neurals
    
    Return
    ------
        dict
            the dictionary of weight tensor.
    '''
    if 'name' not in kwarg:
        kwarg['name'] = 'mlp'
    if 'input_dim' not in kwarg:
        kwarg['input_dim'] = 1
    if 'output_dim' not in kwarg:
        kwarg['output_dim'] = 10  
    if 'trainable' not in kwarg:
        kwarg['trainable'] = True

    _init = tf.contrib.layers.xavier_initializer()
    w = {}
    with tf.variable_scope(kwarg['name']):
        w['weight'] = tf.get_variable('weight',[kwarg['input_dim'], kwarg['output_dim']],initializer=_init, trainable=kwarg['trainable'])
        w['bias'] = tf.get_variable('bias',[kwarg['output_dim']],initializer=_init, trainable=kwarg['trainable'])

    return w

def time2vec(x, weights=None, name='time2vec', **kwarg):
    '''Construct Time2Vec embedings for index-like input.
    Time2Vec(t) = | w_i*t + b_i       ,(0<=i<output_dim/2)
                  | sin(w_i*t + b_i)  ,(output_dim/2<=i<output_dim)

    Parameter
    ---------
        x: tensor
            the input tensor, the last dimension is supposed to be 1.
        weights: dict
            a dictionary of weights generated by apdt.ml.deepsets_weight
    
    Return
    ------
        tensor
            The output tensor
    
    '''
    
    if x.shape[-1].value != 1:
        x = x[..., None]
    
    if weights is None:
        if "output_dim" not in kwarg:
            raise Exception("output_dim should be given if weights is None.")
        weights = time2vec_weight(name=name, input_dim=1, output_dim=kwarg["output_dim"])
    
    with tf.variable_scope(name):
        x = tf.tensordot(x, weights['weight'], 1) + weights['bias']
        _dim = x.shape[-1].value
        x1 = x[..., :_dim//2]
        x2 = x[..., _dim//2:]
        x2 = tf.sin(x2)
        x = tf.concat([x1, x2], -1)

    return x