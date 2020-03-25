"""Some general definations of apdt.
"""
import numpy as np
import pandas as pd
import threading

class DataPack():
    """DataPack is the standard data structure used by apdt.
    """
    def __init__(self):
        self.raw_data = None
        self.data = None
        self.site_info = None
        self.data_type = None
        self.sample_unit = None
        self.tag = []

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


