"""Some general definations of apdt.
"""
import numpy as np
import pandas as pd
import threading

class DataPack():
    """DataPack is the standard data structure used by apdt.
    """
    def __init__(self):
        _empty_data = pd.DataFrame(columns=['datetime', 'lon', 'lat', 'data0'])
        _empty_data['datetime'] = pd.to_datetime(_empty_data['datetime'])
        _empty_data = _empty_data.set_index('datetime')
        self.raw_data = _empty_data
        self.data = _empty_data.copy()
        self.site_info = None
        self.data_type = ['data0']
        self.sample_unit = None
        self.tag = []

    def reset(self):
        '''Reset data into just-loaded status.
        '''
        self.data = self.raw_data.copy()

    def concat(self, other):
        self.data.rename(columns={'data' + str(i): j for i,j in enumerate(self.data_type)}, inplace=True)
        target_data = other.data.rename(columns={'data' + str(i): j for i,j in enumerate(other.data_type)}, inplace=False)
        self.data = pd.concat([self.data, target_data], sort=True)
        self.data_type = list({}.fromkeys(self.data_type + other.data_type).keys())
        self.data.rename(columns={j: 'data' + str(i) for i,j in enumerate(self.data_type)}, inplace=True)
        
        if other.site_info is not None:
            if self.site_info is None:
                self.site_info = other.site_info.copy()
            else:
                self.site_info = pd.concat([self.site_info, other.site_info]).drop_duplicates()

        sample_unit_list = ['S', 'M', 'H', 'D', 'W', 'M', 'Y']
        self.sample_unit = sample_unit_list[min(sample_unit_list.index(self.sample_unit), 
                sample_unit_list.index(other.sample_unit))]
        return self
    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def __add__(self, other):
        return self.concat(other)

class SubThread(threading.Thread):
    """SubThread is an interface of python threading.
    """
    def __init__(self, func, args = ()):
        super(SubThread, self).__init__()
        self.func = func
        self.args = args
    
    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


