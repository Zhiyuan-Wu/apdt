'''
APDT: Data IO
--------------------------

Provide useful data fetching/loading/generating tools for standarized data structure.

Check https://github.com/Zhiyuan-Wu/apdt for more information.
'''
from apdt.io.loadnms import load_nms, load_weather
from apdt.io.fake_data import gp_data
from apdt.io.realtime import get_weather_history, get_weather_pred, real_time_weather
from apdt.io.loadfoshan import foshan_data