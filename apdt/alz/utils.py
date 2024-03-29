import numpy as np

def rbf_kernel(x1,x2,sigma=1.0):
    '''Compute rbf kernel between two points.
    k(x1,x2)=exp(-|x1-x2|_2/2sigma^2)

    Parameters
    ----------
        x1: ndarray
            shape (...,D), batch of D-Dimension vectors.
        x2: ndarray
            shape (...,D), batch of D-Dimension vectors.
        sigma: float
            default 1.0
    Return
    ------
        ndarray
            shape (...). 
    '''
    return np.exp(-np.sqrt(np.sum((x1-x2)**2, -1))/(2*sigma**2))

def mmd(P,Q,k=None,sample_size=None):
    '''Compute MMD (Maximum Mean Discrepancy) between two sample sets.
    mmd^2(P,Q) = 1/m^2 sum_{i,j} k(p_i,p_j) + 1/n^2 sum_{i,j} k(q_i,q_j) - 2/mn sum_{i,j} k(p_i,q_j)

    Parameters
    ----------
        P: ndarray
            shape (...,D), batch of D-Dimension vectors.
        Q: ndarray
            shape (...,D), batch of D-Dimension vectors.
        k: callable
            return kernel result, called by k(x,y), default rbf kernel
        sample_size: int
            use a random subset to compute mmd, default use all avaliable. The memory consumption of mmd computation is square to sample_size
    Return
    ------
        float
            mmd value. 
    '''
    if k is None:
        k = lambda x,y: rbf_kernel(x, y, 1.0)
    
    D = P.shape[-1]
    p = np.reshape(P, [-1, D])
    q = np.reshape(Q, [-1, D])
    if sample_size is not None:
        perm = np.arange(p.shape[0])
        np.random.shuffle(perm)
        p = p[perm[:sample_size]]
        perm = np.arange(q.shape[0])
        np.random.shuffle(perm)
        q = q[perm[:sample_size]]

    term1 = np.mean(k(p[:, None], p))
    term2 = np.mean(k(q[:, None], q)) 
    term3 = np.mean(k(p[:, None], q))*2
    return np.sqrt(term1+term2-term3)

def gauss_kl_distance(P,Q,epsilon=1e-3):
    '''Compute KL divengence between two gaussians fitted from given sample set.

    Parameters
    ----------
        P: ndarray
            shape (...,D), batch of D-Dimension vectors.
        Q: ndarray
            shape (...,D), batch of D-Dimension vectors.
        epsilon: float
            a small numper to avoid singular covariance
    
    Return
    ------
        float
            KL divengence of the fitted gaussian. 
    '''
    D = P.shape[-1]
    p = np.reshape(P, [-1, D])
    q = np.reshape(Q, [-1, D])

    mu_p = np.mean(p, 0)
    sigma_p = (p-mu_p).T @ (p-mu_p) / p.shape[0] + epsilon
    mu_q = np.mean(q, 0)
    sigma_q = (q-mu_q).T @ (q-mu_q) / q.shape[0] + epsilon

    # KL[p1||p2] = 1/2 * {log(det(sigma2)/det(sigma1)) - d + tr(sigma2^(-1)sigma1) + (mu2-mu1)'sigma2^(-1)(mu2-mu1)}
    term1 = np.linalg.slogdet(sigma_q)[1] - np.linalg.slogdet(sigma_p)[1]
    term2 = D
    term3 = np.trace(np.linalg.inv(sigma_q) @ sigma_p)
    term4 = (mu_q-mu_p) @ np.linalg.inv(sigma_q) @ (mu_q-mu_p)
    result = 0.5 * (term1 - term2 + term3 + term4)

    return result

def haversine(loc1, loc2):
    '''Compute the harversine distance between two points.
    Parameters
    ----------
        loc1: ndarray
            shape N*2, N pairs of (lon, lat).
        loc2: ndarray
            shape M*2, M pairs of (lon, lat).
    Return
    ------
        ndarray
            shape N*M, [i,j] denote the distance between loc1[i] and loc2[j]. in unit km. 
    '''
    loc1 = np.array(loc1)/180.0*np.pi
    loc2 = np.array(loc2)/180.0*np.pi
    R = 6378.137
    result = []
    for i in range(loc1.shape[0]):
        h = np.sin((loc1[i,1]-loc2[:,1])/2)**2 + np.cos(loc1[i,1])*np.cos(loc2[:,1])*np.sin((loc1[i,0]-loc2[:,0])/2)**2
        d = 2*R*np.arcsin(np.sqrt(h))
        result.append(d)
    return np.array(result)

def IDW(sloc, svalue, tloc, scale=0.05, kernel='exp'):
    '''The Inverse Distance Weighted interplotation method.
    Parameters
    ----------
        sloc: ndarray
            shape N*2, N pairs of scource location (lon, lat)
        svalue: ndarray
            shape N*K, corresponding scource value (or vector length K).
        tloc: ndarray
            shape M*2, M pairs of target location (lon, lat)
        scale: float, >0, default 0.1
            determine the influence range of a scource, the larger the larger.
        kernel: str, default 'exp'
            the inverse kernel function, optional: 'exp', '1/x'
    Return
    ------
        ndarray
            shape M*K, corresponding target value.
    '''
    kernel = {'exp': np.exp, '1/x': lambda x: 1/(x+1e-5)}[kernel]
    
    distance = haversine(sloc, tloc)
    distance_std = np.std(distance)
    if distance_std==0:
        # This happens when there are only one scource, we return a constant result in this case
        result = np.ones((tloc.shape[0],svalue.shape[1]))*svalue[0]
        return result
    weight = []
    for i in range(sloc.shape[0]):
        _w = kernel(-distance[i,:]/(distance_std*scale))
        weight.append(_w)
    weight = np.array(weight) + 1e-99
    weight = weight/np.sum(weight,0)
    result = np.matmul(svalue.T, weight).T
    return result

def projsplx(x, epsilon=0.0, s=1.0):
    '''Project x, a n-dim real vector, onto the n-dim simplx:
    min ||x-y||_2
    s.t.  y_i>=epsilon
          Sigma y_i = s

    Ref: [1] Chen, Yunmei, and Xiaojing Ye. "Projection onto a simplex." arXiv preprint arXiv:1101.6081 (2011).
         [2] https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex

    Parameters
    ----------
        x, ndarray
            need of shape (n,)
        epsilon, float
            boundary value of y, default 0.0
        s, float
            required sum of y, default 1.0

    Return
    ------
        ndarray
    '''
    n = x.shape[0]
    x_sorted = np.sort(x)
    assert s >= epsilon * n

    i = n-1
    while 1:
        t_i = (np.sum(x_sorted[i:]) + i*epsilon - s) / (n - i)
        if t_i >= x_sorted[i - 1]:
            t_hat = t_i
            break
        i = i - 1
        if i >= 1:
            continue
        else:
            t_hat = (np.sum(x) - s) / (n)
            break
    y = x - t_hat
    y[y<epsilon] = epsilon

    return y
