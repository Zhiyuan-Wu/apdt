import numpy as np

def mcl(M, expand_factor = 2, inflate_factor = 2, max_loop = 100, mult_factor = 1, eps = 1e-5):
    '''Markov Cluster Algorithm (MCL).
    MCL is a very simple clustering algorithm on graph, which intuitively believe a random walk is more likely to be trapped in a cluster. Note that in MCL a single node could be in different clusters.

    Parameters
    -----
        M, ndarray.
            The input graph adjacency matrix.
        expand_factor, int, default 2.
            The expand factor. random walk step num, larger number tends to fewer clusters.
        inflate_factor, float, default 2.
            The inflate factor. larger number tends to more clusters.
        max_loop, int, defalut 100.
            The max  number of loops to stop the algorithm.
        mult_factor, float, default 1.
            The number adds to the diagnal of matrix. tends to make algorithm stable.
        eps, float, default 1e-5.
            A small number used to check convergence.
    Returns
    -----
        clusters, dict
    Reference
    -----
        Van Dongen, S. (2000) Graph Clustering by Flow Simulation. PhD Thesis, University of Utrecht, The Netherlands.
    '''
    assert type(M) is np.ndarray and M.ndim==2
    M = M + mult_factor * np.identity(M.shape[0])
    M = M / M.sum(0)

    for i in range(max_loop):
        _M = M
        # inflate step
        M = np.power(M, inflate_factor)
        M = M / M.sum(0)
        # expend step
        M = np.linalg.matrix_power(M, expand_factor)
        # Convergence Check
        if np.abs(M - _M).max() < eps:
            break
    
    # get_cluster
    clusters = []
    for i, r in enumerate((M>0).tolist()):
        if r[i]:
            clusters.append(M[i,:]>0)

    clust_map  ={}
    for cn , c in enumerate(clusters):
        for x in  [ i for i, x in enumerate(c) if x ]:
            clust_map[cn] = clust_map.get(cn, [])  + [x]

    return M, clust_map