import numpy as np
import itertools

try:
    import tensorflow as tf
except:
    pass

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