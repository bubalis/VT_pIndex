# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:56:11 2020

@author: benja
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio 
from rasterio.merge import merge
import math
import pygeoprocessing
import pygeoprocessing.routing as pyg_routing
import os
import rasterio
import matplotlib.pyplot as plt
import gdal
import time
from rasterio.enums import Resampling



#%%
def get_tileNum(row):
     for col in [c for c in row.index if 'TILENUM' in c]:
         if row[col]:
             return row[col]
         
def give_first_valid(iterable):
    for i in iterable:
        if i:
            return i

def merge_tiles(tiles_1, tiles_2):
    tiles_2.to_crs(tiles_1.crs)
    merged=gpd.overlay(tiles_1, tiles_2, how='union')
    merged['TILENUMBER']=merged.apply(get_tileNum, axis=1)
    return merged[['TILENUMBER', 'geometry']]





def getLS(slope_val, flow_acc):
    '''LS value from a cell.
    From https://jblindsay.github.io/wbt_book/available_tools/geomorphometric_analysis.html'''
    if flow_acc<0:
        return 0
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

def rkls(LS, k):
    r=113 #rainfall erosivity value for Western VT
    return LS*k*r

vect_rkls=np.vectorize(rkls)

   
def extract_K_facs(LS_raster, out_dir):
    x_min,  y_min, x_max, y_max = LS_raster.bounds
    xRes, yRes=LS_raster.transform[0], -LS_raster.transform[4]
    ds = gdal.Rasterize(os.path.join(out_dir, 'K_factors.tif'), soil_shp, xRes=xRes, yRes=yRes, 
                        outputBounds=[x_min, y_min, x_max, y_max], 
                        outputType=gdal.GDT_Float32, attribute='K_factor')
    ds= None
    
    return rasterio.open(os.path.join(out_dir, 'K_factors.tif'))
      
        
def calculateRKLS(directory):
    '''Run all steps toCalculate the erosion potential raster for a given Watershed.'''
    dem_path=os.path.join(directory, 'H2O_shed_DEM.tif')  
    LS_raster=calculateLS(dem_path, directory)
    k_raster=extract_K_facs(LS_raster, directory)
    makeRKLS_raster(LS_raster, k_raster, directory)
    LS_raster.close()
    k_raster.close()

def makeRKLS_raster(LS_raster, k_raster, out_dir):
    '''Final step to make RKLS raster'''
    print(time.localtime())
    print('Making RKLS raster')
    profile=LS_raster.profile
    LS_data=LS_raster.read(1)
    k_data=k_raster.read(1)
    
    L_raster, k_raster=None, None
    
    RKLS_vals=vect_rkls(LS_data, k_data)
    
    with rasterio.open(os.path.join(out_dir, 'RKLS.tif'), 'w', **profile) as dst:
        dst.write(RKLS_vals.astype(rasterio.float32), 1)    

def calculateLS(dem_path, out_dir):
    '''Calculate the length-slope factor using routines from pygeoprocessing.'''
    
    stream_path=os.path.join(out_dir, 'stream.tif')
    pit_filled_dem_path=os.path.join(out_dir, 'pit_filled_dem.tif')
    slope_path=os.path.join(out_dir, 'slope_raster.tif')
    flow_direction_path=os.path.join(out_dir, 'flow_direction.tif')
    #avg_aspect_path=os.path.join(out_dir, 'avg_aspect.img')
    flow_accumulation_path=os.path.join(out_dir, 'flow_acc.tif')
    #ls_factor_path=os.path.join(out_dir, 'ls.img')
    #rkls_path=os.path.join(out_dir, 'rkls.img')
    threshold_flow_accumulation=1
    #aligned_drainage_path=False
    #stream_and_drainage_path=stream_path
    
    print('Making pit-filled DEM')
    #all these functions create a raster file at the location specified in the last 'path' arg
    pyg_routing.fill_pits((dem_path, 1), pit_filled_dem_path)
    print('Calculating Slope')
    pygeoprocessing.calculate_slope((pit_filled_dem_path, 1),
            slope_path)
    print('Routing Flow')
    pyg_routing.flow_dir_mfd(
            (pit_filled_dem_path, 1),
            flow_direction_path)
    
    print('Flow accumulation')
    pyg_routing.flow_accumulation_mfd((flow_direction_path, 1),
            flow_accumulation_path)
    try:
        pyg_routing.extract_streams_mfd(
            (flow_accumulation_path, 1),
            (flow_direction_path, 1),
            float(threshold_flow_accumulation),
            stream_path,)
    except:
        pass
    
    with rasterio.open(slope_path) as slope_raster:
        slope_array=slope_raster.read(1)
    with rasterio.open(flow_accumulation_path) as flow_acc_raster:
        flow_array=flow_acc_raster.read(1)
    
    slope_array=venforce_non_neg(slope_array)
    flow_array=venforce_non_neg(flow_array)
    
    print (time.localtime())
    print('calculating LS')
    
    LS=vgetLS(slope_array, flow_array)
    slope_array=None
    flow_array=None
    
    
    with rasterio.open(dem_path) as dem:
        profile=dem.profile
        
    profile.update({'nodata': 0})
    
    print (time.localtime())
    print("Writing LS raster")
    with rasterio.open(os.path.join(out_dir, 'LS.tif'), 'w', **profile) as dst:
        dst.write(LS.astype(rasterio.float32), 1)
    
    LS_raster=rasterio.open(os.path.join(out_dir, 'LS.tif'))
    
    return LS_raster


def resize_raster(filepath, factor):
    '''Decrease the size of a raster by a scaling factor.'''
    with rasterio.open(filepath) as dataset:

    # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height / factor),
                int(dataset.width / factor)
            ),
            resampling=Resampling.bilinear
        )
    
        # scale image transform
        out_transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
    out_meta=dataset.meta.copy()
    out_meta.update({"transform": out_transform,
                     'width': data.shape[-1],
                     'height': data.shape[-2],
                    }
                   )
    with rasterio.open(filepath, 'w', **out_meta) as dest:
        dest.write(data)


def watershed_raster(HUC12, out_dir):
    '''Make a DEM raster for a watershed. Save it as out_dir\dem '''
    
    #collect rasters to merge together
    src_to_merge=[]
    files=os.listdir(dems_dir)
    for code in H2Oshed_tiles[HUC12]:
        files=[f for f in files if code in f]
        if files:
            src_to_merge.append(rasterio.open(os.path.join(dems_dir, files[0])))
    
    if not src_to_merge:
        return 
    
    rast, out_transform=merge(src_to_merge)
    
    #set raster profile to be same as src but with different shape/xform:
    out_meta = src_to_merge[0].meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": rast.shape[1],
                    "width": rast.shape[2],
                     "transform": out_transform,
                    }
                   )
    
    
    out_fp=os.path.join(out_dir, 'H2O_shed_DEM.tif')
    
    for r in src_to_merge:
        r.close()
    
    plt.imshow(rast[0])
    plt.show()
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(rast)
    
    
    
    
    #resize_raster(out_fp, 2)
    








county_code='SO01'
aoi_path=os.path.join(os.getcwd(), 'intermediate_data', f'Geologic_{county_code}_poly.shp')
dems_dir=os.path.join(os.getcwd(), 'source_data', 'DEM_rasters')

main_out_dir=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')


watershed_path=os.path.join(os.getcwd(), 'source_data', 'VT_Subwatershed_Boundaries_-_HUC12-shp', 'VT_Subwatershed_Boundaries_-_HUC12.shp')

ref_maps=[]
for p in [f for f in os.listdir(dems_dir) if os.path.isdir(os.path.join(dems_dir, f))]:
    path=os.path.join(dems_dir, p, f'{p[1:]}.shp')
    print(path)
    m=gpd.read_file(path)
    m.plot()
    plt.show()
    ref_maps.append(m)
m=None


ref_map=ref_maps.pop()

while ref_maps:
    next_map=ref_maps.pop()
    next_map.to_crs(ref_map.crs, inplace=True)
    ref_map=merge_tiles(ref_map, next_map)
    
ref_map.plot()
plt.show()


#%%
aoi=gpd.read_file(aoi_path)
crs=aoi.crs
aoi['null']=0
aoi=aoi.dissolve(by='null')[['geometry', 'AREASYMBOL']]
ref_map.to_crs(crs, inplace=True)
aoi['geometry']=aoi.buffer(200)
ref_map2=gpd.overlay(aoi, ref_map, how='intersection')
ref_map2.plot()



h2Oshed=gpd.read_file(watershed_path)



h2Oshed.to_crs(ref_map2.crs)
ext=gpd.overlay(ref_map2, h2Oshed)


#%%
h2Oshed=None
ref_map=None
ref_map2=None

ext.plot(column='HUC12')


H2Oshed_tiles={}
for H2Oshed in ext['HUC12'].unique():
    df=ext[ext['HUC12']==H2Oshed]
    H2Oshed_tiles[H2Oshed]=[t for t in df['TILENUMBER'].unique() if type(t)==str]

df=None
ext=None
next_map=None
#%%

  

soil_shp= r"C:\Users\benja\VT_P_index\model\intermediate_data\Geologic_SO01_poly.shp"   


for HUC12_code in H2Oshed_tiles:    
    print(HUC12_code)
    out_dir=os.path.join(main_out_dir, HUC12_code)
    if (not os.path.exists(out_dir)) and H2Oshed_tiles[HUC12_code]:
        os.makedirs(out_dir)
        watershed_raster(HUC12_code, out_dir)
        with open(os.path.join(out_dir, f'tiles.txt'), 'w') as f:
            for line in H2Oshed_tiles[HUC12_code]:
                print(line, file=f)
        
        calculateRKLS(out_dir)
        

