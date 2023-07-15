# -*- coding: utf-8 -*-
"""
@Features:
1.output_surface:['longitude', 'latitude', 'mean_sea_level_pressure', 'u_component_of_wind_10m', 'v_component_of_wind_10m', 'temperature_2m']
                 top/bottom/right/left of the array is 90N/90S/0.125/359.875,respectively

2.output_upper:['longitude', 'latitude', 'geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
               top/bottom/right/left of the array is 90N/90S/0.125/359.875,respectively

@Author: L.F. Zhan
@Date：2023/7/14
"""

import numpy as np
# import netCDF4 as nc
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    data_nc = []
    data_time = []
    # 6h-per-week
    t0 = datetime.datetime(2023, 7, 9, 16)
    for i in range(1, 29):
        data = np.load(r'forecasts\2023-07-09-08-00\output_surface' + str(i * 6) + '.npy')
        data_nc.append(data[3, 245, 464] - 273.15)  # test 1 of grid
        t1 = t0 + timedelta(0.25 * i)
        data_time.append(t1.strftime('%m%d%H%M'))
    data_nc_arr = np.array(data_nc)
    stadata = pd.read_csv('sta_data.csv')
    stadata = np.array(stadata)
    sta_nc = np.empty([0,1])
    for i in range(0,len(stadata),6):
        sta_nc = np.vstack([sta_nc,stadata[i, 6]])
    plt.plot(data_time, data_nc)
    plt.plot(data_time[0:23], sta_nc[1:24])
    plt.legend(['pangu','58606'])
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation=90, tickdir='in')
    plt.xlabel('time/BJ')
    plt.ylabel('t2m/℃')
    plt.subplots_adjust(bottom=0.2)
    plt.show()

