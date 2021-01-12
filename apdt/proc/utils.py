import pandas as pd
import numpy as np

def linear_normalize(datapack, columns='data0', method='99pt'):
    '''Normalize data linealy column-wise.
    Parameter
    ---------
        datapack: DataPack
        columns: str or list, default: 'data0'
            The column names to be normalized. if given 'ALL', all field start with 'data' will be processed.
        method: str, default: '99pt'
            The method used to calculate zoom boundary.
            Supported: 'max-min'(max-min), '99pt'(99 percentail value), '3std'(3 times of std range)
    Return
    ------
        DataPack
    '''
    if columns=='ALL':
        columns = [str(x) for x in datapack.data.columns if str(x).startswith('data')]
    elif type(columns) is str:
        columns = [columns]
    
    datapack.normalize_factor = {}
    for c in columns:
        series = datapack.data[c].values
        if method=='max-min':
            series_max = np.max(series)
            series_min = np.min(series)
        if method=='99pt':
            series_max = np.percentile(series, 99)
            series_min = np.percentile(series, 1)
        if method=='999pt':
            series_max = np.percentile(series, 99.9)
            series_min = np.percentile(series, 0.1)
        if method=='3std':
            series_max = np.mean(series) + 3 * np.std(series)
            series_min = series_max - 6 * np.std(series)
        if series_max==series_min:
            series_max = series_min + 1
        series = (series - series_min) / (series_max - series_min)
        datapack.data[c] = series
        datapack.normalize_factor.update({c: (series_min, series_max)})

    return datapack

def time_stamp_feature(datapack, item='ALL'):
    '''Add sin and cos embedding to data as external feature.
    Parameters
    ------
        datapack, Datapack
            the datapack to be processed.
        item, list of str, default 'ALL'
            a sub-list of ['month_cos', 'month_sin', 'weekday_cos', 'weekday_sin', 'hour_cos', 'hour_sin'], deciding which feature to append.
    Returns
    ------
        datapack
    '''
    if item=='ALL':
        item = ['month_cos', 'month_sin', 'weekday_cos', 'weekday_sin', 'hour_cos', 'hour_sin']
    if type(item) is str:
        item = [item]

    K = len(datapack.data_type)
    time_stamp = datapack.data.index
    datapack.data_type = datapack.data_type + item
    if 'month_cos' in item:
        datapack.data['data'+str(K)] = np.cos(2*np.pi*time_stamp.month.to_numpy()/12.0)
        K += 1
    if 'month_sin' in item:
        datapack.data['data'+str(K)] = np.sin(2*np.pi*time_stamp.month.to_numpy()/12.0)
        K += 1
    if 'weekday_cos' in item:
        datapack.data['data'+str(K)] = np.cos(2*np.pi*time_stamp.weekday.to_numpy()/7.0)
        K += 1
    if 'weekday_sin' in item:
        datapack.data['data'+str(K)] = np.sin(2*np.pi*time_stamp.weekday.to_numpy()/7.0)
        K += 1
    if 'hour_cos' in item:
        datapack.data['data'+str(K)] = np.cos(2*np.pi*time_stamp.hour.to_numpy()/24.0)
        K += 1
    if 'hour_sin' in item:
        datapack.data['data'+str(K)] = np.sin(2*np.pi*time_stamp.hour.to_numpy()/24.0)
        K += 1
    return datapack
