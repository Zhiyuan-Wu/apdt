"""Some general definations of apdt.
"""
import numpy as np
import pandas as pd
import pickle
import threading

class DataPack():
    """DataPack is the standard data structure used by apdt.
    """
    def __init__(self, path=None):
        if path is None:
            _empty_data = pd.DataFrame(columns=['datetime', 'lon', 'lat', 'data0'])
            _empty_data['datetime'] = pd.to_datetime(_empty_data['datetime'])
            _empty_data = _empty_data.set_index('datetime')
            self.raw_data = _empty_data
            self.data = _empty_data.copy()
            self.site_info = None
            self.data_type = ['data0']
            self.sample_unit = None
            self.tag = []
        elif type(path) is str:
            self.load(path)

    def reset(self):
        '''Reset data into just-loaded status.
        '''
        self.data = self.raw_data.copy()

    def concat(self, other):
        '''combine two datapack into one.
        To Do
        -----
            - self.tag should be properly processed.
            - Non-inplace version?
        '''
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

        if self.site_info is not None:
            self.site_num = self.site_info.shape[0]

        sample_unit_list = ['S', 'M', 'H', 'D', 'W', 'M', 'Y']
        self.sample_unit = sample_unit_list[min(sample_unit_list.index(self.sample_unit), 
                sample_unit_list.index(other.sample_unit))]
        return self
    
    def merge(self, other, on=['datetime','lat','lon']):
        '''Merge two datapack into one, or concat two along dimension 1.
        '''
        for x in self.data_type:
            if x in other.data_type:
                raise Exception('Merge error: two datapack have same data type ' + x)
        self.data.rename(columns={'data' + str(i): j for i,j in enumerate(self.data_type)}, inplace=True)
        target_data = other.data.rename(columns={'data' + str(i): j for i,j in enumerate(other.data_type)}, inplace=False)
        self.data = self.data.reset_index()
        target_data = target_data.reset_index()
        target_data = target_data[on+other.data_type]
        self.data = pd.merge(self.data, target_data, how='left',on=['datetime','lat','lon'])
        self.data_type = list({}.fromkeys(self.data_type + other.data_type).keys())
        self.data.rename(columns={j: 'data' + str(i) for i,j in enumerate(self.data_type)}, inplace=True)
        self.data = self.data.set_index('datetime')
        return self

    def dump(self, path):
        '''Dump Datapack into pickle file. Be careful only standard class member will be stored.
        '''
        dumped_data = {'raw_data':self.raw_data, 'data': self.data, 'site_info': self.site_info,
            'data_type': self.data_type, 'sample_unit': self.sample_unit, 'tag': self.tag}
        with open(path, 'wb') as f:
            pickle.dump(dumped_data, f, 1)

    def load(self, path):
        with open(path, 'rb') as f:
            dumped_data = pickle.load(f)
        self.raw_data = dumped_data['raw_data']
        self.data = dumped_data['data']
        self.site_info = dumped_data['site_info']
        self.data_type = dumped_data['data_type']
        self.sample_unit = dumped_data['sample_unit']
        self.tag = dumped_data['tag']

    def add_row(self, datetime, lat=None, lon=None, value=None):
        '''Append rows into datapack. This fuction is usually used as virtual-sampling.
        '''
        pass

    def drop_site(self, site_id):
        '''Drop rows at given site.
        '''
        if type(site_id) is str:
            site_id = [site_id]

        for x in site_id:
            self.data = self.data.reset_index()
            remove_idx = self.data.index[self.data.site_id==x]
            self.data.drop(index=remove_idx, inplace=True)
            self.data = self.data.set_index('datetime')
            remove_idx = self.site_info.index[self.site_info.site_id==x]
            self.site_info.drop(index=remove_idx, inplace=True)
    
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


