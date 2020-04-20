# -*- coding: UTF-8 -*-
from apdt.general import DataPack, SubThread
import numpy as np
import pandas as pd
import yaml
import time
import json

import os
current_path = os.path.dirname(__file__)
config_path = os.path.abspath(current_path + '/../config.yaml')
with open(config_path) as f:
    _config = yaml.load(f, Loader=yaml.FullLoader)

def load_nms(site, start_date, end_date=None, gas='pm2d5'):
    '''Load national monitoring stations (NMS) data.
    Parameters
    ----------
        site: string or list
            The station ID of queried stations like "1001A", or list of such. Sepcifically, if given 'ALL', all available sites will be used.
        start_date: string
            The start time (include) like "2017-01-01".
        end_date: string, default: same as start_date
            The end time (include) like "2018-12-31"
        gas: string or list, default: 'pm2d5'
            The gas type to be collected.
    Return
    ------
        DataPack
            The loaded data that can be used by apdt.
    '''
    worker_num = _config['subthread_max_num']
    data_path = _config['nms_data_path']
    location = pd.read_csv(data_path+'site_location.csv')

    if site=='ALL':
        site = list(location['监测点编码'])
    elif type(site) is str:
        site = [site]
    gas = {'pm2d5': 'PM2.5', 'pm10': 'PM10', 'aqi': 'AQI', 'so2': 'SO2', 'no2': 'NO2',\
        'o3': 'O3', 'co': 'CO', 'co2': 'CO2'}[gas]
    if end_date is None:
        end_date = start_date

    
    location = location[[location.iloc[i,0] in site for i in range(location.shape[0])]]
    location['监测点名称'] = location['城市'] + '-' + location['监测点名称']
    location = location.rename(columns={'监测点编码': 'site_id', '监测点名称': 'site_name', '经度': 'lon', '纬度': 'lat'})
    location = location[['site_id', 'site_name', 'lat', 'lon']]

    
    #Collect from file
    start_time = time.mktime((time.strptime(start_date+' 12:00:00', '%Y-%m-%d %H:%M:%S')))
    end_time = time.mktime((time.strptime(end_date+' 12:00:00', '%Y-%m-%d %H:%M:%S')))
    template = pd.DataFrame({'hour': [i for i in range(24)]})

    def worker(time_now, end_time):
        data_list = []
        while time_now < end_time:
            file = data_path + 'site' + \
                    str(time.gmtime(time_now).tm_year) + \
                    '/china_sites_' + \
                    time.strftime('%Y%m%d', time.gmtime(time_now)) + '.csv'
            try:
                data = pd.read_csv(file)
                data = data.reindex(columns=list(set(site).union(set(data.columns))))
                data = data.loc[data['type']==gas][['hour'] + site]
                data = template.merge(data, on='hour', how='left')
                data['hour'] = time.strftime('%Y%m%d', time.gmtime(time_now)) + ' ' + data['hour'].astype(str) + ':00:00'
                data['hour'] = pd.to_datetime(data['hour'])
                data = data.rename(columns={'hour': 'datetime'}).set_index('datetime')
                data = data.stack(dropna=False).reset_index(level = 1, drop = False).rename(columns={'level_1': 'site_id', 0: 'data0'})
                data_list.append(data)
            except:
                pass
            time_now += 24*3600
        return data_list
    workers = []
    data_list = []
    time_list = [start_time] + [int((end_time-start_time)/3600/24/worker_num)*k*24*3600+start_time for k in range(1,worker_num)] + [end_time+1]
    for i in range(worker_num):
        workers.append(SubThread(worker,(time_list[i],time_list[i+1])))
        workers[-1].start()
    for i in range(worker_num):
        workers[i].join()
        data_list = data_list+workers[i].get_result()
    data_list = pd.concat(data_list)
    data_list = data_list.reset_index().merge(location, on='site_id', how='left').set_index('datetime')
    data_list = data_list.dropna(subset=['lat', 'lon'])
    location = location.dropna(subset=['lat', 'lon'])

    artifact = DataPack()
    artifact.raw_data = data_list
    artifact.data = data_list.copy()
    artifact.site_info = location
    artifact.data_type = [gas]
    artifact.sample_unit = 'H'
    artifact.tag.append('fixed-location')

    return artifact


def load_weather(site, start_date, end_date=None, feature='temperature', **kwarg):
    '''Load national monitoring stations (NMS) weather data.
    Parameters
    ----------
        site: string or list
            The station ID of queried stations like "1001A", or list of such.
        start_date: string
            The start time (include) like "2017-01-01".
        end_date: string, default: same as start_date
            The end time (include) like "2018-12-31"
        feature: string or list, default: 'temperature'
            The data type to be collected.
    Return
    ------
        DataPack
            The loaded data that can be used by apdt.
    To Do
    -----
        Deal with alias.
        Deal with unit transform.
        Deal with data download.
        Deal with coordinates input.
        Deal with multi-thread
    '''

    if 'enable_warning' not in kwarg.keys():
        kwarg['enable_warning'] = True

    worker_num = _config['subthread_max_num']
    data_path = _config['nms_data_path']
    location = pd.read_csv(data_path+'site_location.csv')
    # feature = ['temperature','humidity','windSpeed','windBearing',\
    #            'cloudCover','visibility']

    if site=='ALL':
        # site = list(location['监测点编码'])
    elif type(site) is str:
        site = [site]
    if end_date is None:
        end_date = start_date

    
    location = location[[location.iloc[i,0] in site for i in range(location.shape[0])]]
    location['监测点名称'] = location['城市'] + '-' + location['监测点名称']
    location = location.rename(columns={'监测点编码': 'site_id', '监测点名称': 'site_name', '经度': 'lon', '纬度': 'lat'})
    location = location[['site_id', 'site_name', 'lat', 'lon']]

    data_bag = []
    for site_id in site:
        #Collect from file
        start_time = time.mktime((time.strptime(start_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
        end_time = time.mktime((time.strptime(end_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
        time_now = start_time
        stamp_bias = time.mktime((time.strptime(start_date+' 00:00:00','%Y-%m-%d %H:%M:%S')))
        
        data_list = [[] for _ in range(len(feature))]
        time_stamp = [[] for _ in range(len(feature))]
        while time_now<=end_time:
            file = 'data/weather/'+site_id+'/'+time.strftime('%Y-%m-%d',time.gmtime(time_now))+'.json'
            with open(file) as f:
                data = json.load(f)
                for k in range(len(data['hourly']['data'])):
                    _stamp = int((data['hourly']['data'][k]['time']-stamp_bias)/3600)
                    _data = data['hourly']['data'][k]
                    for m in range(len(feature)):
                        try:
                            data_list[m].append(_data[feature[m]])
                            time_stamp[m].append(_stamp)
                        except:
                            pass
            time_now += 24*3600

        # Linear interploation
        Length = int((end_time-start_time)//3600 + 24)
        data_ratio = np.array([len(data_list[m]) for m in range(len(feature))])/Length
        if np.any(data_ratio < 0.2) and kwarg['enable_warning']:
            print(__file__+' Runtime Warning: ')
            print('Too much missing value in site ' + site_id + ' Weather Data, the returned interpolated data may be not meaningful.')
        data_interp = []
        for i in range(len(feature)):
            data_interp.append(np.interp(np.arange(Length), time_stamp[i], data_list[i]))
        data = np.stack(data_interp,-1)

        data_table = pd.DataFrame(data=data,columns=['data'+str(i) for i in range(len(feature))])
        data_table['site_id'] = site_id
        data_table['datetime'] = pd.date_range(start=start_date,periods=Length,freq='H')
        data_bag.append(data_table)

    data_bag = pd.concat(data_bag)
    data_bag = data_bag.merge(location, on='site_id', how='left').set_index('datetime')
    data_bag = data_bag.dropna(subset=['lat', 'lon'])
    location = location.dropna(subset=['lat', 'lon'])

    artifact = DataPack()
    artifact.raw_data = data_bag
    artifact.data = data_bag.copy()
    artifact.site_info = location
    artifact.data_type = [feature]
    artifact.sample_unit = 'H'
    artifact.tag.append('fixed-location')
    artifact.tag.append('time-aligned')

    return artifact

    debug = np.array([2,3,3])