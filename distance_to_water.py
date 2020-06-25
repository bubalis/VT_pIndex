# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:44:47 2020

@author: benja
"""

import numpy as np
import rasterio
import itertools
import os
import geopandas as gpd
import fiona
from shapely.geometry.point import Point 
import matplotlib.pyplot as plt

#%%

def get_cropFields():
    '''Load Crop Fields geodatabase'''
    gdb_file=os.path.join("P_Index_LandCoverCrops", "P_Index_LandCoverCrops","Crop_DomSoil.dbf")
    layers = fiona.listlayers(gdb_file)
    layer=layers[0]
    
    
    gdf = gpd.read_file(gdb_file,layer=layer)
    return gdf

crop_fields=get_cropFields()
soils, aoi=load_soils(soils_path)
#%%
def get_adj_nums(x, length):
    if x==0:
        return (0,1)
    elif x==length-1:
        return (x,x-1)
    else:
        return (x-1, x, x+1)
        

def is_local_max(y,x, array):
    y_vals=get_adj_nums(y, array.shape[0])
    x_vals=get_adj_nums(x, array.shape[1])
    locs=[(y_val, x_val) for x_val in x_vals for y_val in y_vals
          if not (x_val==x and y_val==y)]
    print(locs)
    print([array[loc] for loc in locs])
    return array[(y,x)]>=max([array[loc] for loc in locs])

#%%
def point_from_raster_cell(y,x, raster):
    top_left=(raster.bounds[3], raster.bounds[0])
    xform=raster.transform
    yloc=top_left[0]+(y+.5)*xform[4]
    xloc= (top_left[1]+(x+.5)*xform[0])
    try:
        return Point(xloc, yloc)
    except:
        print(xloc, yloc)
        break





def get_max_point(raster):
    a=r.read(1)
    y, x=np.where(a==a.max())
    return pd.GeoSeries(point_from_raster_cell(y,x, raster), crs=raster.crs)



def distance_to_water(field_shape, flow_raster):
    
    
#%%
'''
def find_max_outflow(flow_array):
    a1, a2=np.where(a==a.max())
    return ((a1[i], a2[i] for )'''