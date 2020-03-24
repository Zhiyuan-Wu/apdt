from apdt.general import DataPack
from apdt.alz import haversine
import numpy as np

def find_sites_range(datapack, clat, clon, radius):
    '''Find sites within a circle on the map.
    Parameters
    ----------
        datapack: DataPack
            site_info is required.
        clat: float
            the lat of center point.
        clon: float
            the lon of center point.
        radius: float
            the radius of the circle. in unit km.
    Return
    ------
        list
            the list of site_id.
    '''
    if datapack.site_info is None:
        raise Exception("site_info is required.")
    loc1 = np.array([[clon, clat]])
    loc2 = datapack.site_info[['lon', 'lat']].values
    h_distance = haversine(loc1, loc2)
    h_distance[np.isnan(h_distance)] = radius + 1
    result = list(datapack.site_info['site_id'][np.squeeze(h_distance<radius)])
    return result

