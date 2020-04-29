from apdt.general import DataPack
import numpy as np
import pandas as pd
from sklearn import gaussian_process as gp

def gp_data(time_length, site_num, dimension=1, kernel_weight=None, noise_level=None, seed=None):
    '''Generate a fake PM2.5-like data for quick start using gaussian process.
    Parameters
    ----------
        - time_length: int. How many hours should data have.
        - site_num: int. How many sites should data have.
        - dimension: int, default 1, How many independent sample each ST-point have.
        - kernel_weight: list of three float numbers, default [1.0, 1.0, 1.0], the relevant variance of three components: long-term trend, short-term fluction and period wave. or list of six float numbers, where addional 3 will be considered as lenth_scale parameter of three kernels(0-1, default 1).
        - noise_level: float, default 0.01, the white noise add to kernel, note that this is neccessary for long time generation.
        - seed: int. The random seed. 
    Return
    ------
        DataPack
    Issue
    -----
    '''
    if seed is not None:
        np.random.seed(seed)
    if kernel_weight is None:
        kernel_weight = [[1.0, 1.0, 1.0]]
    if np.array(kernel_weight).ndim==1:
        kernel_weight = [kernel_weight]
    if noise_level is None:
        noise_level = 0.01
    if len(kernel_weight[0])==3:
        kernel_weight = [kernel_weight[i]+[1.0, 1.0, 1.0] for i in range(len(kernel_weight))]
    
    # Decrease this number if this program stuck.
    generation_step = 1000

    xs = np.arange(generation_step*2).reshape((generation_step*2,1))
    

    if len(kernel_weight)==1:
        # Case: Only one parameter is given, site_num will be treated as new dimension
        k1 = gp.kernels.RBF(length_scale=50.0+100.0*kernel_weight[0][3])
        k2 = gp.kernels.Matern(length_scale=20.0+10.0*kernel_weight[0][4], nu=0.5)
        k3 = gp.kernels.ExpSineSquared(length_scale=1, periodicity=100+200*kernel_weight[0][5])
        kw = gp.kernels.WhiteKernel(noise_level=noise_level)
        k = kernel_weight[0][0]*k1 + kernel_weight[0][1]*k2 + kernel_weight[0][2]*k3 + kw
        C = k(xs)
        C_11 = C[:generation_step,:generation_step]
        C_11_inv = np.linalg.inv(C_11)
        C_21 = C[generation_step:,:generation_step]
        u, s, _ = np.linalg.svd(C_11)
        us = np.matmul(u, np.diag(np.sqrt(s)))
        sample = np.matmul(us, np.random.randn(generation_step,site_num*dimension))
        C_cond = C_11 - np.matmul(np.matmul(C_21.T, C_11_inv), C_21)
        u, s, _ = np.linalg.svd(C_cond)
        us = np.matmul(u, np.diag(np.sqrt(s)))
        time_now = generation_step
        sample_list = [sample]
        while time_now < time_length:
            mu_cond = 0 + np.matmul(np.matmul(C_21, C_11_inv), sample)
            sample = np.matmul(us, np.random.randn(generation_step,site_num*dimension))
            sample = sample + mu_cond
            sample_list.append(sample)
            time_now = time_now + generation_step
        sample_list = np.concatenate(sample_list)[:time_length].reshape((time_length*site_num,dimension))
    else:
        # Case: A list of parameter is given, generate site_num samples one by one
        sample_list_all = []
        for i in range(site_num):
            k1 = gp.kernels.RBF(length_scale=50.0+100.0*kernel_weight[i][3])
            k2 = gp.kernels.Matern(length_scale=20.0+10.0*kernel_weight[i][4], nu=0.5)
            k3 = gp.kernels.ExpSineSquared(length_scale=1, periodicity=100+200*kernel_weight[i][5])
            kw = gp.kernels.WhiteKernel(noise_level=noise_level)
            k = kernel_weight[i][0]*k1 + kernel_weight[i][1]*k2 + kernel_weight[i][2]*k3 + kw
            C = k(xs)
            C_11 = C[:generation_step,:generation_step]
            C_11_inv = np.linalg.inv(C_11)
            C_21 = C[generation_step:,:generation_step]
            u, s, _ = np.linalg.svd(C_11)
            us = np.matmul(u, np.diag(np.sqrt(s)))
            sample = np.matmul(us, np.random.randn(generation_step,1*dimension))
            C_cond = C_11 - np.matmul(np.matmul(C_21.T, C_11_inv), C_21)
            u, s, _ = np.linalg.svd(C_cond)
            us = np.matmul(u, np.diag(np.sqrt(s)))
            time_now = generation_step
            sample_list = [sample]
            while time_now < time_length:
                mu_cond = 0 + np.matmul(np.matmul(C_21, C_11_inv), sample)
                sample = np.matmul(us, np.random.randn(generation_step,1*dimension))
                sample = sample + mu_cond
                sample_list.append(sample)
                time_now = time_now + generation_step
            sample_list = np.concatenate(sample_list)[:time_length].reshape((time_length*1,dimension))
            sample_list_all.append(sample_list)
        sample_list = np.stack(sample_list_all, 1).reshape((time_length*site_num, dimension))
    
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




    
