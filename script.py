import apdt
import numpy as np

t = apdt.io.load_nms('ALL','2017-01-01')
r = apdt.alz.find_sites_range(t, 39.8673, 116.3660, 100)

_debug = 233