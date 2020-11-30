import numpy as np
import pandas as pd
import apdt

from apdt.config import _config

def spatial_interplot(datapack, method='NN', columns='ALL', **kwarg):
    '''Using spatial interplotion to complete the missing value.
    Parameters
    ----------
        datapack: DataPack
        method: str
            The interplotation method to be used. 
            Supported: 'NN'(Nearest Neighboor), 'IDW'(Inverse Distance Weighted)
        columns: str or list
            which data columns should we processed on. if 'ALL' is given, all data columns will be considered.
            
    Return
    ------
        DataPack
    Note
    ----
        WARNING: very slow for large scale data (> 2 days for ~3M records in our test)
        To be supported: 'Kriging'(Kriging interplotation).
        deal with data1, data2, ...
        To Do: Why multi-thread dont work on this function?

    ''' 
    worker_num = 1
    if method=='NN':
        # A small enough scale IDW will equal to nearest neighboor.
        IDW_scale = 1e-5
    elif method=='IDW':
        try:
            IDW_scale = kwarg['IDW_scale']
        except:
            IDW_scale = 0.05
    else:
        raise Exception("Unknown interpolation method: " + method)
    if columns=='ALL':
        columns = [x for x in datapack.data.columns if x.startswith('data')]
    elif type(columns) is str:
        columns = [columns]
    
    def worker(k):
        for i, timestamp in enumerate(datapack.data.index.drop_duplicates()):
            if i%worker_num != k:
                continue
            slice_data = datapack.data.loc[timestamp].copy()
            for data_type in columns:
                nan_index = slice_data[data_type].isna()
                if not any(nan_index) or all(nan_index):
                    continue
                sloc = slice_data[~nan_index][['lon', 'lat']].values.reshape((-1,2))
                svalue = slice_data[~nan_index][data_type].values.reshape((-1,1))
                tloc = slice_data[nan_index][['lon', 'lat']].values.reshape((-1,2))
                tvalue = apdt.alz.IDW(sloc, svalue, tloc, IDW_scale)
                slice_data.loc[nan_index,data_type] = tvalue
            datapack.data.loc[timestamp] = slice_data
    
    workers = []
    for i in range(worker_num):
        workers.append(apdt.general.SubThread(worker,(i,)))
        workers[-1].start()
    for i in range(worker_num):
        workers[i].join()
    
    return datapack

def temporal_interpolate(datapack, site='ALL', method='linear'):
    '''Using temporal interplotion to complete the missing value. Note that this only works for fixed site.
    Parameters
    ----------
        datapack: DataPack
            site_info required.
        site: str or list, default 'ALL'
            the list of site to be interpolated. If given 'ALL', all available sites will be processed according to site_info.
        method: str, default 'linear'
            The interplotation method to be used. 
            Supported: 'linear'
            Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.resample.Resampler.interpolate.html?highlight=interpolate#pandas.core.resample.Resampler.interpolate      
    Return
    ------
        DataPack
    Note
    ----
    '''
    if site=='ALL':
        site = list(datapack.site_info['site_id'])
    elif type(site) is str:
        site = [site]

    result = []
    mask = pd.Series(False, index=datapack.data.index)
    for target in site:
        _mask = datapack.data['site_id']==target
        series = datapack.data[_mask].resample(datapack.sample_unit).interpolate(method=method).bfill()
        mask = mask | _mask
        result.append(series)
    datapack.data = pd.concat([datapack.data[~mask]]+result).sort_index()
    if all(mask):
        datapack.tag.append('time-aligned')
        datapack.time_length = len(datapack.data.index.drop_duplicates())
        datapack.site_num = datapack.site_info.shape[0]
    return datapack