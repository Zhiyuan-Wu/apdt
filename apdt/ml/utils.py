import numpy as np
import itertools

try:
    import tensorflow as tf
except:
    pass

def batch_norm(x, training_flag, gamma=0.99, learnable=True, name='bn'):
    '''Apply batch norm to input x, i.e., turn each feature channel into a (learnable) gaussian distribution. 
    This function will introduce new in-trainable variables into models. 
    Its Useful to insert this function between linear mapping and activation to get magical improvements/degradation.

    Parameters
    ------
        x, tensor
            input time series of shape [batch_size, spatial_dim_1, ..., spatial_dim_S, feature_dim]
        training_flag, tensor
            a bool scalar tensor indicating if its training phase, usually TFModel.training
        gamma, float, default 0.99
            moving average decay weight.
        learnable, bool, default True
            whether to introduce learnable mean and var to target distribution.
        name, str, default "bn"
            the name prefix for new variables
    Returns
    ------
        tensor
            same shape with x
    Notes
    ------
        This function include in-trainable variables and maybe incompitable with current pretrain model loading mechanism.
    '''
    num_dims = x.shape.ndims
    D = x.shape[-1].value

    running_mean = tf.get_variable(name+'_running_mean', [D,], initializer=tf.constant_initializer(0.0),trainable=False)
    running_var = tf.get_variable(name+'_running_var', [D,], initializer=tf.constant_initializer(1.0),trainable=False)
    if learnable:
        learnable_mean = tf.get_variable(name+'_learnable_mean', [D,], initializer=tf.constant_initializer(0.0))
        learnable_var = tf.get_variable(name+'_learnable_var', [D,], initializer=tf.constant_initializer(1.0))
    mean, var = tf.nn.moments(x, [i for i in range(num_dims-1)])
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(running_mean, running_mean * gamma + mean * (1-gamma)))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(running_var, running_var * gamma + var * (1-gamma)))
    mean, var = tf.cond(training_flag, lambda: (mean, var), lambda: (running_mean, running_var))
    x = (x - mean) / tf.sqrt(var + 1e-6)
    if learnable:
        x = x * tf.sqrt(learnable_var + 1e-6) + learnable_mean

    return x

def layer_norm(x, K=1, learnable=True, name='ln'):
    '''Apply layer norm to input x, i.e., turn output of each sample into a (learnable) gaussian distribution. 
    This function will introduce new in-trainable variables into models. 
    Its Useful to insert this function between layers to get magical improvements/degradation.

    Parameters
    ------
        x, tensor
            input time series of shape [batch_size, spatial_dim_1, ..., spatial_dim_S, feature_dim]
        K, int, default 1
            along last K dimensions to average on
        learnable, bool, default True
            whether to introduce learnable mean and var to target distribution. Becareful that layer norm use 
            element-wise learnable parameter (different from batch norm that use scalar), which may result in
            very large parameter size.
        name, str, default "bn"
            the name prefix for new variables
    Returns
    ------
        tensor
            same shape with x
    '''
    num_dims = x.shape.ndims
    N = list(map(lambda _x: _x.value, x.shape))

    if learnable:
        learnable_mean = tf.get_variable(name+'_learnable_mean', N[-K:], initializer=tf.constant_initializer(0.0))
        learnable_var = tf.get_variable(name+'_learnable_var', N[-K:], initializer=tf.constant_initializer(1.0))
    mean, var = tf.nn.moments(x, list(range(num_dims))[-K:], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + 1e-6)
    if learnable:
        x = x * tf.sqrt(learnable_var + 1e-6) + learnable_mean

    return x

def _unzip_list(nested):
    '''unzip a list generated by TFModel._zip_run.
    Example: [[a1,a2,a3],[b1,b2,b3]] -> [[a1,b1],[a_2,b_2],[a3,b_3]]
    the basic elements should not be empty list [].
    '''
    # detect depth
    depth = 0
    _temp = nested
    while 1:
        if type(_temp) is list:
            depth = depth + 1
            _temp = _temp[0]
        else:
            break
    
    # unzip nested
    if depth==1:
        nested = [nested]
    elif depth>2:
        for _ in range(depth-2):
            nested = list(itertools.chain(*nested))
    nested = list(map(list, zip(*nested)))
    return nested

def stacked_window(x, width, shift=0):
    '''Compute moving stacked window on a given time series.

    Parameters
    ------
        x, tensor
            input time series of shape [batch_size, time_length, feature_dim]
        width, int
            window width
        shift, int, default 0
            decide the index of given time stamp in the output. For example, if shift=0, then x[1, 2, 3] will show at result[1, 2, 0, 3]
            minus value is also supported.
    Returns
    ------
        tensor
            output stacked time series of shape [batch_size, time_length, width, feature_dim]
    '''
    N = x.shape[0].value
    T = x.shape[1].value
    D = x.shape[2].value
    _x = tf.pad(x, [[0, 0], [max(shift, 0), max(width - 1 - shift, 0)], [0, 0]])
    _T = _x.shape[1].value
    _x = tf.transpose(_x, [0, 2, 1])
    _x = tf.reshape(_x, [N*D, _T, 1])
    w = tf.constant(np.eye(width).reshape([width, 1, width]), dtype=x.dtype)
    result = tf.nn.conv1d(_x, w, 1, 'VALID')
    result = result[:, max(-shift, 0): max(-shift, 0) + T, :]
    result = tf.reshape(result, [N, D, T, width])
    result = tf.transpose(result, [0, 2, 3, 1])
    return result

def pearson_corr(x, y, axis=-1, keepdims=True):
    ''' Compute the pearson correlation coefficient.

    Parameters
    ------
        x, tensor
            the first input tensor
        y, tensor
            the second input tensor
        axis, int, default -1
            along which index to compute correlation
        keepdims, bool, default True
            if keep corresponding dim.
    Returns
    ------
        tensor
            output stacked time series of shape [batch_size, time_length, width, feature_dim]
    '''
    x_mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    y_mean = tf.reduce_mean(y, axis=axis, keepdims=True)
    x_norm = tf.sqrt(tf.reduce_sum((x-x_mean)**2, axis=axis, keepdims=True) + 1e-8)
    y_norm = tf.sqrt(tf.reduce_sum((y-y_mean)**2, axis=axis, keepdims=True) + 1e-8)
    _x = (x - x_mean) / x_norm
    _y = (y - y_mean) / y_norm
    corr = tf.reduce_sum(_x*_y, axis=axis, keepdims=keepdims)

    return corr