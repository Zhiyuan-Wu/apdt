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