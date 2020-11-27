# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import time
import json
import os
import requests

from apdt.config import _config

class DarkSkyAPI():
    '''DarkSkyAPI is a spider on darsky.net. Help you download weather history data for a set of sites using a set of accounts.

    darksky_secret_key_num and darksky_secret_key_0 should be decleared in config.py.

    Method
    ------
        __init__: Initialize.
            sites: a list of sites to collect data on. e.g. ['1001A', '1414A'].
            start_date: the start date (include) to collect data. e.g. '2019-01-01'.
            end_date: the end date (include) to collect data. e.g. '2019-12-31'.
            DailyAccountQuota: the maximum call quota for each accounts per day. e.g. (default) 990.
            save_path: the path to put download data. e.g. (default) 'data/weather/'.

        run: excute download.
            flag: set True to start download. default: False.
            shift: claim a start point for task, this can be useful for continue a job from previous start. default: 0.

    '''
    def __init__(self, sites, start_date, end_date, **kwarg):
        if type(sites) is str:
            sites = [sites]
        if 'DailyAccountQuota' not in kwarg.keys():
            kwarg['DailyAccountQuota'] = 990
        if type(kwarg['DailyAccountQuota']) is not list:
            kwarg['DailyAccountQuota'] = [kwarg['DailyAccountQuota'] for _ in range(_config['darksky_secret_key_num'])]
        if 'save_path' not in kwarg.keys():
            kwarg['save_path'] = 'data/weather/'

        self.site_list = pd.read_csv(_config['nms_data_path'] + 'site_location.csv')
        self.sites = sites
        self.start_date = start_date
        self.end_date = end_date
        self.quota = kwarg['DailyAccountQuota']
        self.save_path = kwarg['save_path']
        self.req_pool = self._requests_pool()
        self.account_num = _config['darksky_secret_key_num']
        self.secret_key_list = [_config['darksky_secret_key_' + str(i)] for i in range(self.account_num)]
        self.entry_num = len(self.req_pool)
        self.est_days = self.entry_num // sum(kwarg['DailyAccountQuota']) + 1
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
                file = self.save_path + s + '/' + time.strftime('%Y-%m-%d', time.gmtime(time_now)) + '.json'
                req_pool.append([req_0, req_1, file])
                time_now += 3600*24
        return req_pool

    def run(self, flag=False, shift=0, wait_hour=24):
        est_days = (self.entry_num - shift) // sum(self.quota) + 1
        print('This task include', self.entry_num - shift, ' entries and need ', est_days, ' days using ', self.account_num, ' accounts.')
        if flag:
            key = 0
            key_count = 0
            for s in self.sites:
                if not os.path.exists(self.save_path + s + '/'):
                    os.mkdir(self.save_path + s + '/')
            for i in range(shift, self.entry_num):
                if (i+1)%500 == 0:
                    print(i+1, '/', self.entry_num, '...')
                if key_count >= self.quota[key]:
                    # print('key ', key, ' run out of quota')
                    key_count = 0
                    key = key + 1
                    if key >= self.account_num:
                        key = 0
                        print('Waiting', wait_hour,' hours...')
                        print('Current shift:', i)
                        time.sleep(wait_hour*3600)
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

    def retry_error(self, flag=False, shift=0, wait_hour=24):
        with open('ERROR_LOG_FILE', 'r') as ERROR_LOG_file:
            lines = [x.replace('\n','') for x in ERROR_LOG_file.readlines()]
            req_pool = [['https://api.darksky.net/forecast/', lines[2*i].rsplit('/',2)[-1], lines[2*i+1]] for i in range(len(lines)//2)]

        with open('ERROR_LOG_FILE_BCK', 'w') as ERROR_LOG_file:
            for line in lines:
                ERROR_LOG_file.write(line+'\n')
        with open('ERROR_LOG_FILE', 'w') as ERROR_LOG_file:
            ...
        
        entry_num = len(req_pool)
        est_days = (entry_num - shift) // sum(self.quota) + 1
        print('This task include', entry_num - shift, ' entries and need ', est_days, ' days using ', self.account_num, ' accounts.')
        
        if flag:
            key = 0
            key_count = 0
            for s in self.sites:
                if not os.path.exists(self.save_path + s + '/'):
                    os.mkdir(self.save_path + s + '/')
            for i in range(shift, entry_num):
                if (i+1)%500 == 0:
                    print(i+1, '/', entry_num, '...')
                if key_count >= self.quota[key]:
                    key_count = 0
                    key = key + 1
                    if key >= self.account_num:
                        key = 0
                        print('Waiting', wait_hour,' hours...')
                        print('Current shift:', i)
                        time.sleep(wait_hour*3600)
                req = req_pool[i]
                url = req[0] + self.secret_key_list[key] + '/' + req[1]
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
            print('Use DarkSkyAPI.retry_error(flag = True) to confirm and run task.')