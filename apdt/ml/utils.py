import numpy as np

try:
    import tensorflow as tf
except:
    pass

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
    w = tf.constant(np.eye(width).reshape([width, 1, width]) * 1.0)
    result = tf.nn.conv1d(_x, w, 1, 'VALID')
    result = result[:, max(-shift, 0): max(-shift, 0) + T, :]
    result = tf.reshape(result, [N, D, T, width])
    result = tf.transpose(result, [0, 2, 3, 1])
    return result
