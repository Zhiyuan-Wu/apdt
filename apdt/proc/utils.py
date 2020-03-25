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
        if method=='3std':
            series_max = np.mean(series) + 3 * np.std(series)
            series_min = series_max - 6 * np.std(series)
        if series_max==series_min:
            series_max = series_min + 1
        series = (series - series_min) / (series_max - series_min)
        datapack.data[c] = series
        datapack.normalize_factor.update({c: (series_min, series_max)})

    return datapack

