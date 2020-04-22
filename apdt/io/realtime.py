from apdt.general import DataPack
import numpy as np
import pandas as pd
import yaml
import time
import json
import os
import requests

current_path = os.path.dirname(__file__)
config_path = os.path.abspath(current_path + '/../config.yaml')
with open(config_path) as f:
    _config = yaml.load(f, Loader=yaml.FullLoader)

def get_weather_history(t,lon,lat,feature='temperature',secret_key=None):
    '''Get weather data in the past. This function will have 1 darksky API use. 
    Parameters
    ----------
        t: str, a date string like '2017-01-01'. It will be considered as UTC+8 (Beijing) time.
        lon: float, the target longtitude.
        lat: float, the target latitude.
        feature: str or list of str, the desired feature list.
        secret_key: str, a darksky secret key. if None is given, darksky_secret_key_0 in config file will be tried.
    Return
    ------
        ndarray, of shape 24*D, where D is the number of feature.
    '''

    # Parameter Check
    if not os.path.exists('data/.temp/'):
        os.mkdir('data/.temp/')
    if pd.to_datetime(t) > pd.to_datetime((time.time()+8*3600)*1e9):
        raise Exception('Input date should in history: '+t)
    if type(feature) is str:
        feature = [feature]
    alia_dict = {}
    alia_dict.update({x: 'temperature' for x in ['temperature','temp','tmp','wendu']})
    alia_dict.update({x: 'humidity' for x in ['humidity','hmd','hum','shidu']})
    alia_dict.update({x: 'windSpeed' for x in ['windSpeed','speed','spd','fengsu']})
    alia_dict.update({x: 'windBearing' for x in ['windBearing','direction','angel','fengxiang']})
    alia_dict.update({x: 'visibility' for x in ['visibility','kejiandu','keshidu']})
    alia_dict.update({x: 'pressure' for x in ['pressure','press','yali','qiya']}) 
    for i,x in enumerate(feature):
        if x not in alia_dict.keys():
            raise Exception(x+' is not supported.')
        feature[i] = alia_dict[x]
    if secret_key is None:
        if 'darksky_secret_key_0' in _config:
            secret_key = _config['darksky_secret_key_0']
        else:
            raise Exception('Secret key is not provide.')

    time_stamp = time.mktime((time.strptime(t+' 00:00:01','%Y-%m-%d %H:%M:%S')))
    req_loc = '{0:.2f},{1:.2f},'.format(lat,lon) + str(round(time_stamp))
    req = 'https://api.darksky.net/forecast/'+secret_key+'/'+req_loc
    file = 'data/.temp/'+req_loc+'.json'
    if os.path.exists(file):
        with open(file, 'r') as result_file:
            data = json.load(result_file)
    else:
        r = requests.get(req)
        with open(file, 'w') as result_file:
            result_file.write(r.text)
        data = json.loads(r.text)
    data_list = [[] for _ in range(len(feature))]
    time_stamp = [[] for _ in range(len(feature))]    
    stamp_bias = time.mktime((time.strptime(t+' 00:00:00','%Y-%m-%d %H:%M:%S')))
    for k in range(len(data['hourly']['data'])):
        _stamp = int((data['hourly']['data'][k]['time']-stamp_bias)/3600)
        _data = data['hourly']['data'][k]
        for m in range(len(feature)):
            try:
                data_list[m].append(_data[feature[m]])
                time_stamp[m].append(_stamp)
            except:
                pass    
    data_interp = []
    for i in range(len(feature)):
        data_interp.append(np.interp(np.arange(24), time_stamp[i], data_list[i]))
    data = np.stack(data_interp,-1)      
    return data    

def get_weather_pred(lon,lat,feature='temperature',secret_key=None):
    '''Get weather data in the future 48h. This function will have 1 darksky API use. 
    Parameters
    ----------
        lon: float, the target longtitude.
        lat: float, the target latitude.
        feature: str or list of str, the desired feature list.
        secret_key: str, a darksky secret key. if None is given, darksky_secret_key_0 in config file will be tried.
    Return
    ------
        ndarray, of shape 48*D, where D is the number of feature.
    '''
    # Parameter Check
    if not os.path.exists('data/.temp/'):
        os.mkdir('data/.temp/')
    if type(feature) is str:
        feature = [feature]
    alia_dict = {}
    alia_dict.update({x: 'temperature' for x in ['temperature','temp','tmp','wendu']})
    alia_dict.update({x: 'humidity' for x in ['humidity','hmd','hum','shidu']})
    alia_dict.update({x: 'windSpeed' for x in ['windSpeed','speed','spd','fengsu']})
    alia_dict.update({x: 'windBearing' for x in ['windBearing','direction','angel','fengxiang']})
    alia_dict.update({x: 'visibility' for x in ['visibility','kejiandu','keshidu']})
    alia_dict.update({x: 'pressure' for x in ['pressure','press','yali','qiya']}) 
    for i,x in enumerate(feature):
        if x not in alia_dict.keys():
            raise Exception(x+' is not supported.')
        feature[i] = alia_dict[x]
    if secret_key is None:
        if 'darksky_secret_key_0' in _config:
            secret_key = _config['darksky_secret_key_0']
        else:
            raise Exception('Secret key is not provide.')

    time_stamp = time.time()
    req_loc = '{0:f},{1:f}'.format(lat,lon)
    req = 'https://api.darksky.net/forecast/'+secret_key+'/'+req_loc
    file = 'data/.temp/'+req_loc+str(time.gmtime(time_stamp+8*3600).tm_hour)+'.json'
    if os.path.exists(file):
        with open(file, 'r') as result_file:
            data = json.load(result_file)
    else:
        r = requests.get(req)
        with open(file, 'w') as result_file:
            result_file.write(r.text)
        data = json.loads(r.text)    
       
    data_list = [[] for _ in range(len(feature))]
    time_stamp = [[] for _ in range(len(feature))]    
    stamp_bias = data['hourly']['data'][0]['time']
    Length = len(data['hourly']['data'])
    for k in range(Length):
        _stamp = int((data['hourly']['data'][k]['time']-stamp_bias)/3600)
        _data = data['hourly']['data'][k]
        for m in range(len(feature)):
            try:
                data_list[m].append(_data[feature[m]])
                time_stamp[m].append(_stamp)
            except:
                pass    
    data_interp = []
    for i in range(len(feature)):
        data_interp.append(np.interp(np.arange(Length), time_stamp[i], data_list[i]))
    data2 = np.stack(data_interp,-1)       
    return data2[1:]


def real_time_weather(lon, lat, past_hours=81, future_hours=48, feature='temperature', secret_key=None):
    '''Get real time weather data. This function will have some darksky API use depends on how many days involved. 
    Parameters
    ----------
        lon: float, the target longtitude.
        lat: float, the target latitude.
        past_hours: int, default 81, how many hours shoud we check back from now.
        future_hours: int, default 48, how many hours shoud we go on from now, up to 48.
        feature: str or list of str, the desired feature list.
        secret_key: str, a darksky secret key. if None is given, darksky_secret_key_0 in config file will be tried.
    Return
    ------
        ndarray, of shape (past_hours+future_hours)*D, where D is the number of feature.
    To Do
    -----
        make result into a DataPack object.
    '''
    if future_hours>0:
        data_pred = get_weather_pred(lon,lat,feature,secret_key)[:future_hours]
        data_pred = [data_pred]
    elif future_hours==0:
        data_pred = []
    else:
        raise Exception('future_hours should > 0.')
    if past_hours>0:
        TH = time.gmtime(time.time()+8*3600).tm_hour + 1
        PH = (past_hours-TH)
        N = int(np.ceil(PH/24.0))
        data_past = []
        for i in range(int(N),-1,-1):
            a = get_weather_history(time.strftime('%Y-%m-%d',time.gmtime(time.time()+8*3600-i*24*3600)),lon,lat,feature,secret_key)
            data_past.append(a)
        data_past = np.concatenate(data_past,0)[N*24-PH:-(24-TH)]
        data_past = [data_past]
    elif future_hours==0:
        data_past = []
    else:
        raise Exception('past_hours should > 0.')
    data = np.concatenate(data_past+data_pred,0)
    return data
