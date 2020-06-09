# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:34:10 2020

@author: benja
"""

import pygeoprocessing
import pygeoprocessing.routing as pyg_routing
import os
import math
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import gdal





    


def getLS(slope_val, flow_acc):
    '''LS value from a cell.
    From https://jblindsay.github.io/wbt_book/available_tools/geomorphometric_analysis.html'''
    return 1.4*(flow_acc/22.13)**.4*(math.sin(math.tan(slope_val/100))/.0896)**1.3

vgetLS=np.vectorize(getLS)



def enforce_non_neg(x):
    '''Clean data for running raster calculations.'''
    if x<0:
        return 0
    elif x>100:
        return 100
    elif isinstance(x, complex):
        return -9999
    else:
        return x
    

venforce_non_neg= np.vectorize(enforce_non_neg)


main_out_dir=os.path.join('intermediate_data', 'USLE')


data_dir=os.path.join('Source_data', "DEM_rasters")


shp_name= r"C:\Users\benja\VT_P_index\model\intermediate_data\Geologic_SO01_poly.shp"  


def rkls(LS, k):
    return LS*k*113

vect_rkls=np.vectorize(rkls)

   
def extract_K_facs(LS_raster, out_dir):
    x_min,  y_min, x_max, y_max = LS_raster.bounds
    
    ds = gdal.Rasterize(os.path.join(out_dir, 'K_factors.tif'), shp_name, xRes=.7, yRes=.7, 
                        outputBounds=[x_min, y_min, x_max, y_max], 
                        outputType=gdal.GDT_Float64, attribute='K_factor')
    ds= None
    
    return rasterio.open(os.path.join(out_dir, 'K_factors.tif'))
      
        
def calculateRKLS(file, out_dir):
    dem_path=os.path.join(data_dir, file)  
    LS_raster=calculateLS(dem_path, out_dir)
    k_raster=extract_K_facs(LS_raster, out_dir)
    makeRKLS_raster(LS_raster, k_raster, out_dir)
    LS_raster.close()
    k_raster.close()

def makeRKLS_raster(LS_raster, k_raster, out_dir):
    RKLS_vals=rkls(LS_raster.read(1), k_raster.read(1))
    profile=LS_raster.profile
    with rasterio.open(os.path.join(out_dir, 'RKLS.tif'), 'w', **profile) as dst:
        dst.write(RKLS_vals.astype(rasterio.float32), 1)    

def calculateLS(dem_path, out_dir):
    '''Calculate the length-slope factor using routines from pygeoprocessing.'''
    
    stream_path=os.path.join(out_dir, 'stream.img')
    pit_filled_dem_path=os.path.join(out_dir, 'pit_filled_dem.img')
    slope_path=os.path.join(out_dir, 'slope_path.img')
    flow_direction_path=os.path.join(out_dir, 'flow_direction.img')
    #avg_aspect_path=os.path.join(out_dir, 'avg_aspect.img')
    flow_accumulation_path=os.path.join(out_dir, 'flow_acc.img')
    #ls_factor_path=os.path.join(out_dir, 'ls.img')
    #rkls_path=os.path.join(out_dir, 'rkls.img')
    threshold_flow_accumulation=1
    #aligned_drainage_path=False
    #stream_and_drainage_path=stream_path
    
    
    #all these functions create a raster file at the location specified in the last 'path' arg
    pyg_routing.fill_pits((dem_path, 1), pit_filled_dem_path)
    pygeoprocessing.calculate_slope((pit_filled_dem_path, 1),
            slope_path)
    pyg_routing.flow_dir_mfd(
            (pit_filled_dem_path, 1),
            flow_direction_path)
    pyg_routing.flow_accumulation_mfd((flow_direction_path, 1),
            flow_accumulation_path)
    pyg_routing.extract_streams_mfd(
            (flow_accumulation_path, 1),
            (flow_direction_path, 1),
            float(threshold_flow_accumulation),
            stream_path,)
    
    slope_raster=rasterio.open(slope_path)
    flow_acc_raster=rasterio.open(flow_accumulation_path)
    
    LS=vgetLS(venforce_non_neg(slope_raster.read(1)), venforce_non_neg(flow_acc_raster.read(1)))
    
    profile=rasterio.open(dem_path).profile
    print(profile)
    with rasterio.open(os.path.join(out_dir, 'LS.tif'), 'w', **profile) as dst:
        dst.write(LS.astype(rasterio.float32), 1)
    
    LS_raster=rasterio.open(os.path.join(out_dir, 'LS.tif'))
    
    return LS_raster
#%%


'''
for file in [f for f in os.listdir(data_dir) if (f[-4:]=='.img' and len(f)>10)]:
    print(file)
    out_dir=os.path.join(main_out_dir, file.split('.')[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        calculateRKLS(file, out_dir)
'''
 

#getting layer information of shapefile.
#shp_layer = input_shp.GetLayer()

#pixel_size determines the size of the new raster.
#pixel_size is proportional to size of shapefile.

#get extent values to set size of output raster.



