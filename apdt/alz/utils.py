import numpy as np

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
    weight = np.array(weight)
    weight = weight/np.sum(weight,0)
    result = np.matmul(svalue.T, weight).T
    return result