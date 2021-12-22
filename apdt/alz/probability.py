import numpy as np

def _gauss_ll(x, mean, sigma):
    D = x.shape[-1]
    sigma = sigma + np.abs(mean) * 1e-8
    ll = - 1/2*np.sum((x-mean)**2/sigma**2, -1) - np.sum(np.log(sigma), -1) - 1/2*D*np.log(2*np.pi)
    return ll

def kde_gauss_density_estimator(support, target, sigma):
    '''Using Gaussian kernel to estimate the mean likelihood of N target samples, on the distribution where K support samples is from.

    Parameter
    -----
        support: ndarry, shape (..., K, D)
        target: ndarry, shape (..., N, D)
        sigma: ndarry, shape (..., D)

    Return
    -----
        ndarry, shape (...)
    '''
    _support = support[..., None, :, :]
    _target = target[..., :, None, :]
    _sigma = sigma[..., None, None, :]
    ll = _gauss_ll(_target, _support, _sigma) # (..., N, K)
    ll_max = np.max(ll, -1, keepdims=True)
    estimate_ll = np.log(np.mean(np.exp(ll - ll_max), -1, keepdims=True)) + ll_max
    mean_ll = np.mean(estimate_ll, (-1, -2))
    return mean_ll

def kde_find_diagonal_sigma(support, folds=5, min=0.1, max=1.0, n=20):
    '''Use K-folds cross-validation to find best diagonal gauss kernel width. (one sigma for all distributions in a batch)

    Parameter
    -----
        support: ndarry
            input support samples, shape (..., K, D)
        folds: int, default 5
            number of k-folds cross-validation
        min: float, default 0.1
            start of search interval (include)
        max: float, default 1.0
            end of search interval (include)
        n: int, default 20
            number of points included in search interval

    Return
    -----
        ndarry
            shape (D,)
    '''
    N, D = support.shape[-2:]
    N = N//folds*folds
    _support = support[..., :N, :]
    _support_split = np.split(_support, folds, -2)
    search_range = np.linspace(min, max, n)

    best_sigma = []
    for i in range(D):
        val_result = []
        for k in range(folds):
            _support_train = np.concatenate([_support_split[j][..., i:i+1] for j in range(folds) if j!=k], -2)
            _support_val = _support_split[k][..., i:i+1]
            _val_result = [np.mean(kde_gauss_density_estimator(_support_train, _support_val, np.array([h]))) for h in search_range]
            val_result.append(_val_result)
        val_result = np.mean(val_result, 0)
        best_sigma.append(search_range[np.argmax(val_result)])

    return np.array(best_sigma)

def kde_find_identity_sigma(support, folds=5, min=0.1, max=1.0, n=20):
    '''Use K-folds cross-validation to find best identity gauss kernel width. (one sigma for all distributions in a batch)

    Parameter
    -----
        support: ndarry
            input support samples, shape (..., K, D)
        folds: int, default 5
            number of k-folds cross-validation
        min: float, default 0.1
            start of search interval (include)
        max: float, default 1.0
            end of search interval (include)
        n: int, default 20
            number of points included in search interval

    Return
    -----
        ndarry
            shape (1,)
    '''
    N, D = support.shape[-2:]
    N = N//folds*folds
    _support = support[..., :N, :]
    _support_split = np.split(_support, folds, -2)
    search_range = np.linspace(min, max, n)

    val_result = []
    for k in range(folds):
        _support_train = np.concatenate([_support_split[j] for j in range(folds) if j!=k], -2)
        _support_val = _support_split[k]
        _val_result = [np.mean(kde_gauss_density_estimator(_support_train, _support_val, np.array([h]))) for h in search_range]
        val_result.append(_val_result)
    val_result = np.mean(val_result, 0)
    best_sigma = search_range[np.argmax(val_result)]

    return np.array([best_sigma])

def kde_find_rule_of_thumb_sigma(support):
    '''Use rule-of-thumb (which use gaussian assumption) to find best diagonal gauss kernel width. This provide different estimation for different distribution.

    Parameter
    -----
        support: ndarry, shape (..., K, D)

    Return
    -----
        ndarry
            shape (..., D)
    '''
    K = support.shape[-2]
    std = np.std(support, -2)
    best_sigma = 1.06 * std * np.power(K, -0.2)
    return best_sigma
