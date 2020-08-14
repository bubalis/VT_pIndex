# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:11:05 2020

@author: benja
"""

import whitebox
import rasterio as rio
from rasterio.plot import show
import numpy as np
import os

wbt=whitebox.WhiteboxTools()

dirr=r"C:\Users\benja\VT_P_index\model\intermediate_data\USLE\041504020304"

dem=os.path.join(dirr,'H2O_shed_DEM.tif')
pf_dem=os.path.join(dirr, 'pit_filed_test.tif')
slope=os.path.join(dirr, 'slope_test.tif')
sca=os.path.join(dirr, 'sca_tst.tif')
ls=os.path.join(dirr, 'ls_test.tif')

wbt.fill_single_cell_pits(dem, pf_dem)
wbt.slope(dem=pf_dem, output=slope)
wbt.d_inf_flow_accumulation(pf_dem, output=sca)
wbt.sediment_transport_index(sca, slope, ls)


def fillNulls(array):
    return np.where(array<-1000, np.nan, array)
    