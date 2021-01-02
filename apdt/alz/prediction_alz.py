# pylint: disable=E1136
import numpy as np

def moving_predict(x, func, par={}, width=None, step=10, strides=1, **kwarg):
    '''apply a auto-regressive moving prediction on given series.
    Parameters
    -------
        x, ndarray
            the input time series, shaped [batch_size, time_length, feature_dims]
        func, callable
            the predictor, given a time series shaped [batch_size, observe_length@(0, ..., T-1), feature_dims], return a time series shaped [batch_size, predict_length@(T, ..., T+strides-1), predict_dim]
        par, dict, default {}
            the kwarg dictionary passed to func beside x.
        width, int, default to the length of input series.
            the observation window width T of func.
        step, int, default 10
            the number of moving steps.
        strides, int, default 1
            the strides of every moving steps.
        predict_dim, int, default 1
            the dimension returned by func.
        loss, str, default 'MAE'
            the loss function, available: 'MAE'.

    Returns
    -------
        ndarray
            the prediction.
        float
            the average prediction loss.
    '''
    # Parameter Check
    if 'predict_dim' not in kwarg.keys():
        kwarg['predict_dim'] = 1
    if 'loss' not in kwarg.keys():
        kwarg['loss'] = 'MAE'

    if width is None:
        width = x.shape[1]

    r = np.array(x)
    N, L, D = x.shape
    r = np.concatenate([r, np.zeros([N, step * strides - (L - width) % (step * strides), D])], axis=1)
    pt = 0
    while pt+width<=L:
        slides = np.array(x[:, pt : pt + width + step * strides, :])
        if slides.shape[1] < width + step * strides:
            slides = np.concatenate([slides, np.zeros([N, width + step * strides - slides.shape[1], D])], axis=1)
        for s in range(step):
            obv = slides[:, s*strides:width+s*strides, :]
            pred = func(obv, **par)
            slides[:, width+s*strides:width+(s+1)*strides, :kwarg['predict_dim']] = pred
        r[:, pt+width:pt + width + step * strides, :] = slides[:, width: width + step * strides, :]
        pt = pt + step * strides

    residual = r[:N, width: L, :kwarg['predict_dim']] - x[:N, width: L, :kwarg['predict_dim']]
    if kwarg['loss'] == 'MAE':
        loss = np.mean(np.abs(residual))
    return r, loss