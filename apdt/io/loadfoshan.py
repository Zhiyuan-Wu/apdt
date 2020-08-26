# -*- coding: UTF-8 -*-
from apdt.general import DataPack
import numpy as np
import pandas as pd
import time
import json
import os

from apdt.config import _config

def foshan_data(site, start_date, end_date=None, gas='pm2d5'):
    '''Load fixed sensor data collected from foshan.
    Parameters
    ----------
        site: string or list
            The station name of queried stations like "体育馆岗亭", or list of such. Sepcifically, if given 'ALL', all available sites will be used.
        start_date: string
            The start time (include) like "2018-11-01".
        end_date: string, default: same as start_date
            The end time (include) like "2018-11-30"
        gas: string or list, default: 'pm2d5'
            The gas type to be collected.
    Return
    ------
        DataPack
            The loaded data that can be used by apdt.
    '''
    # Parameter Check
    data_path = _config['foshan_data_path']
    air = gas # To Do: support a list of gas type.

    if site=='ALL':
        site = sorted(os.listdir(data_path))
    elif type(site) is str:
        site = [site]
    site = [x if x.endswith('.xls') else x+'.xls' for x in site]

    if start_date < '2018-11-01' or start_date > '2018-11-30':
        raise Exception("Foshan data range between '2018-11-01'-'2018-11-30'")
    if end_date is None:
        end_date = start_date
    if end_date < '2018-11-01' or end_date > '2018-11-30':
        raise Exception("Foshan data range between '2018-11-01'-'2018-11-30'")
    if end_date < start_date:
        raise Exception('end_date shoule larger than start_date.')
    
    datalist = []
    location = []
    
    for id,target in enumerate(site):
        data = pd.read_excel(data_path+target, index_col=0)
        dat = data[[air]+['lon', 'lat']].dropna()
        dat = dat.reset_index().rename(columns={'index':'datetime',air:'data0', 'lon':'lon', 'lat':'lat'}).set_index('datetime')
        dat = dat.resample("T").asfreq()
        dat = dat[dat.index>=start_date + ' 00:00:00']
        dat = dat[dat.index<=end_date + ' 23:59:59']
        # nanidx=dat.index[np.isnan(dat[air])]
        dat=dat.dropna()
        dat=dat.resample("T").bfill()
        # dat["isnan"]=0
        # dat["isnan"].loc[nanidx]=1
        dat['site_id'] = 'foshan' + str(id)
        dat['site_name'] = target.split('.')[0]
        dat["date"]=dat.index.map(lambda x:x.date())
        
        datalist.append(dat)
        location.append(['foshan' + str(id), target.split('.')[0], dat['lon'][0], dat['lat'][0]])

    datalist = pd.concat(datalist)
    location = pd.DataFrame(location, columns=['site_id', 'site_name', 'lon', 'lat'])
    artifact = DataPack()
    artifact.raw_data = datalist
    artifact.data = datalist.copy()
    artifact.site_info = location
    artifact.data_type = gas
    artifact.sample_unit = 'M'
    artifact.tag.append('fixed-location')
    artifact.tag.append('time-aligned')
    return artifact
    