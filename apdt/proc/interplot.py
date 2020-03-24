import numpy as np
import apdt
import yaml

with open('apdt/config.yaml') as f:
    _config = yaml.load(f, Loader=yaml.FullLoader)

def spatial_interplot(datapack, method='NN', **kwarg):
    '''Using spatial interplotation to complete the missing value.
    Parameters
    ----------
        datapack: DataPack
        method: str
            The interplotation method to be used. 
            Supported: 'NN'(Nearest Neighboor), 'IDW'(Inverse Distance Weighted)
            
    Return
    ------
        DataPack
    Note
    ----
        To be supported: 'Kriging'(Kriging interplotation).
        deal with data1, data2, ...

    '''
    datapack.tag.append('idw')
    # To Do: Why multi-thread dont work on this function?
    worker_num = 1#_config['subthread_max_num']
    def worker(k):
        for i, timestamp in enumerate(datapack.data.index.drop_duplicates()):
            if i%worker_num != k:
                continue
            if i==13:
                _debug = 233
            slice_data = datapack.data.loc[timestamp].copy()
            nan_index = slice_data['data0'].isna()
            if not any(nan_index) or all(nan_index):
                continue
            sloc = slice_data[~nan_index][['lon', 'lat']].values.reshape((-1,2))
            svalue = slice_data[~nan_index]['data0'].values.reshape((-1,1))
            tloc = slice_data[nan_index][['lon', 'lat']].values.reshape((-1,2))
            tvalue = apdt.alz.IDW(sloc, svalue, tloc)
            slice_data.loc[nan_index,'data0'] = tvalue
            datapack.data.loc[timestamp] = slice_data
    workers = []
    for i in range(worker_num):
        workers.append(apdt.general.SubThread(worker,(i,)))
        workers[-1].start()
    for i in range(worker_num):
        workers[i].join()
    return datapack