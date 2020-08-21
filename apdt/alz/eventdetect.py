from apdt.general import DataPack
import numpy as np
import pywt
import pandas as pd

def groups(dat,type):
    dgroups=dat.groupby(dat[type])
    return dgroups 

def group_info(dgroup,air='None'):
    date=dgroup[0]
    dg=dgroup[1]
    if air=='None':
        return date,dg
    else:
        a=dg[air]
        return date,dg,a
    
def re_mean(array_like):
    try:
        np.nan_to_num(array_like)
        v80=np.percentile(array_like,75)
        v20=np.percentile(array_like,25)
        v_array=array_like[array_like>=v20]
        v_array=v_array[v_array<=v80]
        return np.mean(v_array)
    except IndexError:
        return np.nan

def re_std(array_like):
    try:
        np.nan_to_num(array_like)
        v80=np.percentile(array_like,75)
        v20=np.percentile(array_like,25)
        v_array=array_like[array_like>=v20]
        v_array=v_array[v_array<=v80]
        return np.std(v_array)
    except IndexError:
        return np.nan
        
def VIX_ln(array):
    logarray=np.log(array)
    return np.std(logarray) 

def wavelet_decomp(data, step, w="haar"):
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for _ in range(step):    
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)
    return ca,cd

def wavelet_comp(ca, cd, w="haar"):
    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    return rec_a,rec_d

def get_theta(dg, data_type):
    dg_mean = re_mean(dg[data_type]) # the mean of 25~75 percentage
    dg_vix = VIX_ln(dg[data_type]) # std(log(x))
    dg['daymean'] = dg_mean
    dg['dayVIX'] = dg_vix
    theta = dg_mean * dg_vix
    return theta

def threshold(d,theta):
    for dd in d:
        # dd_mean=re_mean(dd)
        # whats this?????
        dd[dd>theta]=0
        dd[dd<theta]=0


def after_filter(dg,ra,rd,lv=5):
    rdd=[]
    for dd in rd:
        rdd.append(dd[0:len(dg)])
    x=np.array(rdd[0:lv])
    dx=np.sum(x,axis=0)
    ax=ra[lv-1]
    dg["after_filter"]=ax[0:len(dg)]+dx
    return dg

def smooth(dat,affiltdata,dev):
    dat["after_filter"]=affiltdata["after_filter"].rolling(60,min_periods=1,center=True).mean()
    dat["devid"]=dev
    dat["day_mean"]=affiltdata["daymean"]
    dat["day_VIX"]=affiltdata["dayVIX"]
    

def short_event(dat, data_type):
    dat["thresh"]=dat["after_filter"]+dat["day_VIX"]*3
    singleshortdata=dat[dat[data_type]-dat["thresh"]>0]
    return singleshortdata

def longterm_data(dat,dev,level=4,theta=30):
    longdata=pd.DataFrame()
    longtermdat=dat.resample("10T").mean()
    longtermdat["air_10"]=longtermdat['after_filter']
    longdata[dev]=longtermdat["air_10"]
    longtermdat=longtermdat[["air_10","day_mean","day_VIX"]]
    a,d=wavelet_decomp(longtermdat["air_10"],level)
    threshold(d,theta)
    ra,rd=wavelet_comp(a,d)
    rdd=[]
    for dd in rd:
        rdd.append(dd[0:len(longtermdat)])
        
    x=np.array(rdd[0:level])
    dx=np.sum(x,axis=0)
    ax=ra[level-1]
    longtermdat["after_filter"]=ax[0:len(longtermdat)]+dx
    longtermdat["after_filter"]=longtermdat["after_filter"].rolling(18,min_periods=1,center=True).mean()
    longtermdat["thresh"]=longtermdat["after_filter"]+(longtermdat["day_VIX"]*3)
    return longtermdat

def long_event(data, site_name):
    temp=pd.DataFrame()
    simglelongevent=data[data["air_10"]-data["thresh"]>0]
    simglelongevent["devid"]=site_name
    temp=temp.append(simglelongevent)
    return temp


def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def save_csv(path,data):
    data["time"]=data.index
    name=get_df_name(data)
    data.to_csv(path+name+'.csv',index=False)

def small_scale(dorg,data_type='data0',out=False,outpath=None):
    
    # dorg=pd.read_csv(datapack)
    #dat.index=pd.to_datetime(dat.time)
    dorg['time'] = dorg.index
    dgs=dorg.groupby(dorg["devid"])
    allevent=pd.DataFrame()
    for dg in dgs:
        dev=dg[0]
        data=dg[1]
    
        data.loc[:,"time"]=pd.to_datetime(data["time"])
        data=data.sort_values(by=['time'])
        data["tshift"]=data["time"].shift()
        data["tdiff"]=data["time"]-data["tshift"]
        data=data.reset_index()
        
        df = data[data.tdiff > pd.Timedelta("2m")]
        start_idx = [0]
        start_idx.extend(df.index)
        end_idx = list(df.index - 1)
        end_idx.append(len(data) - 1)
        end_time = data.iloc[end_idx]['time'].reset_index(drop=True)
        start_time = data.iloc[start_idx]['time'].reset_index(drop=True)
        con_time=end_time-start_time
        
        env_strength = []
        for i in range(len(start_idx)):
            d = data.iloc[start_idx[i]:end_idx[i]]
            if len(d)>0:
                env_strength.append(np.mean(d[data_type]-d["after_filter"]))
            else:
                env_strength.append(np.nan)
        
        dat=pd.DataFrame()
        dat["starttime"] = data.loc[start_idx]["time"].reset_index(drop=True)
        dat["timedelta"]=con_time.map(lambda x:x.seconds/60)
        
        dat["start_value"] = data.loc[start_idx][data_type].reset_index(drop=True)  
        dat["single_strength"] = env_strength
    
        dat["month"]=dat["starttime"].map(lambda x:x.month)
        dat["weekday"]=dat["starttime"].map(lambda x:x.weekday())
        dat["hour"]=dat["starttime"].map(lambda x:x.hour)
        dat["dev"] = dev  
        allevent=allevent.append(dat)
	
    # sevent=allevent.dropna()
    if out==True:
        if outpath==None:
            allevent.to_csv("gathershortevent.csv")
        else:
            allevent.to_csv(outpath+"gathershortevent.csv")
    return allevent

def big_scale(dorg,time_delta=10,strength=10,out=False,outpath=None):
    
    #dorg=pd.read_csv(datapack)
    #dat.index=pd.to_datetime(dat.time)
    dorg['time'] = dorg.index
    dgs=dorg.groupby(dorg["devid"])
    allevent=pd.DataFrame()
    for dg in dgs:
        dev=dg[0]
        data=dg[1]
        
        data.loc[:,"time"]=pd.to_datetime(data["time"])
        data=data.sort_values(by=['time'])
        data["tshift"]=data["time"].shift()
        data["tdiff"]=data["time"]-data["tshift"]
        data=data.reset_index()
        
        df = data[data.tdiff > pd.Timedelta("20m")]
        start_idx = [0]
        start_idx.extend(df.index)
        end_idx = list(df.index - 1)
        end_idx.append(len(data) - 1)
        end_time = data.iloc[end_idx]['time'].reset_index(drop=True)
        start_time = data.iloc[start_idx]['time'].reset_index(drop=True)
        con_time=end_time-start_time
        
        env_strength = []
        for i in range(len(start_idx)):
            d = data.iloc[start_idx[i]:end_idx[i]]
            if len(d)>0:
                env_strength.append(np.mean(d["air_10"]-d["after_filter"]))
            else:
                env_strength.append(np.nan)
        
        dat=pd.DataFrame()
        dat["starttime"] = data.loc[start_idx]["time"].reset_index(drop=True)
        dat["timedelta"]=con_time.map(lambda x:x.seconds/60)
        
        dat["start_value"] = data.loc[start_idx]["air_10"].reset_index(drop=True)  
        dat["single_strength"] = env_strength
        
        dat["month"]=dat["starttime"].map(lambda x:x.month)
        dat["weekday"]=dat["starttime"].map(lambda x:x.weekday())
        dat["hour"]=dat["starttime"].map(lambda x:x.hour)
        dat["dev"] = dev  
        allevent=allevent.append(dat)
        	
        dd=dat.dropna()
        dd=dd[dd.timedelta>time_delta]
        dd=dd[dd.single_strength>strength]
        
        # print(dev,len(dat.dropna()))
    
    # sevent=allevent.dropna()
    if out==True:
        if outpath==None:
            allevent.to_csv("gatherlongevent.csv")
        else:
            allevent.to_csv(outpath+"gatherlongevent.csv")
    return allevent

def wavelet_detect(datapack, **kwarg):
    '''Event detection based on wavelet filtered threshold.
    Author: Jiayi Huang & boting Lin
    Parameters
    ----------
        datapack: DataPack
        columns: str or list, default: 'data0'
            which data columns should we processed on. if 'ALL' is given, all data columns will be considered.
        outpath: str, dafault: None
            the path to save the result table. Disable when given None
            
    Return
    ------
        DataFrame, DataFrame
            The detected short event and long event will be returned as a table.
    Note
    ------
        1. Early version, we need more detailed test.
        2. make kwargs for adjustable parameters.
        3. Deal with massive 'value set on copy' warning. and log(0) numerical warning.
        4. function threshold doesnt make sense.
        5. better visualization. at least a report.
        6. how do result depends on the sample frequency.
    '''
    # Parameter Check
    if 'columns' not in kwarg.keys():
        kwarg['columns'] = 'data0'
    if 'outpath' not in kwarg.keys():
        kwarg['outpath'] = None
    if 'date' not in datapack.data.columns:
        datapack.data["date"] = datapack.data.index.map(lambda x:x.date())
    # Main
    shortevent=pd.DataFrame()
    longtermdata=pd.DataFrame()
    longevent=pd.DataFrame()
    for _,site in enumerate(datapack.site_info['site_name']):
        dat = datapack.data[datapack.data['site_name']==site].copy()
        dgroups = groups(dat,'date')
        affiltdata = pd.DataFrame()
        for dgroup in dgroups:
            _, dg, da = group_info(dgroup,'data0') # dg: full table of date, da: dg[air_type]
            a,d=wavelet_decomp(da,5)
            theta=get_theta(dg, 'data0')
            threshold(d,theta)
            ra,rd=wavelet_comp(a,d) # ra: reconstructed a
            dg=after_filter(dg,ra,rd)
            affiltdata=affiltdata.append(dg)

        smooth(dat,affiltdata,site) # moving average of window width 60 and write it into dat
        
        shortevent=shortevent.append(short_event(dat, 'data0'))
        longtermdat=longterm_data(dat,site)
        longtermdata[site]=longtermdat["air_10"]
        longevent=longevent.append(long_event(longtermdat, site))

    gather_long = big_scale(longevent,10,10,out=kwarg['outpath'] is not None,outpath=kwarg['outpath'])   
    gather_short = small_scale(shortevent,out=kwarg['outpath'] is not None,outpath=kwarg['outpath'])
    
    return gather_short, gather_long