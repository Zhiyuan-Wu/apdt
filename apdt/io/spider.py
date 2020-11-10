# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import time
import json
import os
import requests

from apdt.config import _config

class DarkSkyAPI():
    def __init__(self, sites, start_date, end_date, **kwarg):
        if type(sites) is str:
            sites = [sites]
        if 'DailyAccountQuota' not in kwarg.keys():
            kwarg['DailyAccountQuota'] = 990

        self.site_list = pd.read_csv(_config['nms_data_path'] + 'site_location.csv')
        self.sites = sites
        self.start_date = start_date
        self.end_date = end_date
        self.quota = kwarg['DailyAccountQuota']
        self.req_pool = self._requests_pool()
        self.account_num = _config['darksky_secret_key_num']
        self.secret_key_list = [_config['darksky_secret_key_' + str(i)] for i in range(self.account_num)]
        self.entry_num = len(self.req_pool)
        self.est_days = self.entry_num // (self.account_num * self.quota) + 1
        ...

    def _requests_pool(self):
        req_pool = []
        for s in self.sites:
            start_time = time.mktime((time.strptime(self.start_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
            end_time = time.mktime((time.strptime(self.end_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
            lon = self.site_list.loc[self.site_list['监测点编码'] == s, '经度'].values[0]
            lat = self.site_list.loc[self.site_list['监测点编码'] == s, '纬度'].values[0]
            req_loc = '{0:f},{1:f},'.format(lat, lon)
            time_now = start_time
            while time_now <= end_time:
                req_0 = 'https://api.darksky.net/forecast/'
                req_1 = '/'+req_loc+str(round(time_now))
                file = 'data/weather/'+s+'/'+time.strftime('%Y-%m-%d',time.gmtime(time_now))+'.json'
                req_pool.append([req_0, req_1, file])
                time_now += 3600*24
        return req_pool

    def run(self, flag=False):
        print('This task need ', self.est_days, ' days using ', self.account_num, ' accounts.')
        if flag:
            key = 0
            key_count = 0
            for i in range(self.entry_num):
                if key_count >= self.quota:
                    # print('key ', key, ' run out of quota')
                    key_count = 0
                    key = key + 1
                    if key >= self.account_num:
                        key = 0
                        print('Waiting 24 hours...')
                req = self.req_pool[i]
                url = req[0] + self.secret_key_list[key] + req[1]
                file = req[2]
                try:
                    r = requests.get(url)
                    with open(file, 'w') as result_file:
                        result_file.write(r.text)
                except:
                    print('Error: ', file)
                    with open('ERROR_LOG_FILE', 'a') as ERROR_LOG_file:
                        ERROR_LOG_file.write(url+'\n')
                        ERROR_LOG_file.write(file+'\n')
                key_count = key_count + 1
        else:
            print('Use DarkSkyAPI.run(flag = True) to confirm and run task.')
