from apdt.general import DataPack
import numpy as np
import pandas as pd
from sklearn import gaussian_process as gp

def gp_data(time_length, site_num, dimension=1, kernel_weight=None, seed=None):
    '''Generate a fake PM2.5-like data for quick start using gaussian process.
    Parameters
    ----------
        time_length: int. How many hours should data have.
        site_num: int. How many sites should data have.
        dimension: int, default 1, How many independent sample each ST-point have.
        kernel_weight: list of three float numbers, default [1.0, 1.0, 1.0], the relevant variance of three components: long-term trend, short-term fluction and period wave.
        seed: int. The random seed. 
    Return
    ------
        DataPack
    Issue
    -----
        The generated data's statistics dont agree with therotical value???
    '''
    if seed is not None:
        np.random.seed(seed)
    if kernel_weight is None:
        kernel_weight = [1.0, 1.0, 1.0]
    
    # Decrease this number if this program stuck.
    generation_step = 2000

    xs = np.arange(generation_step*2).reshape((generation_step*2,1))
    k1 = gp.kernels.RBF(length_scale=100.0)
    k2 = gp.kernels.Matern(length_scale=30.0, nu=0.5)
    k3 = gp.kernels.ExpSineSquared(length_scale=1, periodicity=200)
    k = kernel_weight[0]*k1 + kernel_weight[1]*k2 + kernel_weight[2]*k3
    C = k(xs)
    C_11 = C[:generation_step,:generation_step]
    C_11_inv = np.linalg.inv(C_11)
    C_21 = C[generation_step:,:generation_step]
    sample = np.zeros((generation_step,site_num*dimension))
    mu_cond = 0 + np.matmul(np.matmul(C_21, C_11_inv), sample)
    C_cond = C_11 - np.matmul(np.matmul(C_21.T, C_11_inv), C_21)
    u, s, _ = np.linalg.svd(C_cond)
    us = np.matmul(u, np.diag(np.sqrt(s)))
    time_now = 0
    sample_list = []
    while time_now < time_length:
        mu_cond = 0 + np.matmul(np.matmul(C_21, C_11_inv), sample)
        sample = np.matmul(us, np.random.randn(generation_step,site_num*dimension))
        sample = sample + mu_cond
        sample_list.append(sample)
        time_now = time_now + generation_step
        
    sample_list = np.concatenate(sample_list)[:time_length].reshape((time_length*site_num,dimension))
    datetime_list = pd.date_range(start='2000-1-1',periods=time_length,freq='H')
    site_name_list = ['virtual_site'+str(i) for i in range(site_num)]
    idx = pd.MultiIndex.from_product([datetime_list, site_name_list],names=('datetime', 'site_name'))
    data = pd.DataFrame(index=idx, columns=['data'+str(i) for i in range(dimension)], data=sample_list)
    data = data.reset_index()
    site_list = pd.DataFrame(data=np.random.randn(site_num, 2)+np.array([39.8673, 116.3660]),
        columns=['lat','lon'])
    site_list['site_name'] = site_name_list
    site_list['site_id'] = ['V'+str(i).zfill(4) for i in range(site_num)]
    data = data.merge(site_list, how='left', on='site_name')
    data = data.set_index('datetime')

    datapack = DataPack()
    datapack.raw_data = data
    datapack.data = data.copy()
    datapack.site_info = site_list
    datapack.data_type = ['virtual_type_'+str(i) for i in range(dimension)]
    datapack.sample_unit = 'H'
    datapack.tag.append('fixed-location')
    datapack.tag.append('time-aligned')
    datapack.time_length = time_length
    datapack.site_num = site_num

    return datapack




    
