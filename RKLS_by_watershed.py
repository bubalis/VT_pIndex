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
from scipy.spatial import cKDTree


#%%
def get_tileNum(row):
     for col in [c for c in row.index if 'TILENUM' in c]:
         if type(row[col])==str:
             return row[col]
         
def give_first_valid(iterable):
    '''Return the first non ''Falsy' value in an iterable'''
    
    for i in iterable:
        if i:
            return i

def merge_tiles(tiles_1, tiles_2):
    '''Merge two tile geometries'''
    
    tiles_2.to_crs(tiles_1.crs)
    merged=gpd.overlay(tiles_1, tiles_2, how='union')
    merged['TILENUMBER']=merged.apply(get_tileNum, axis=1)
    return merged[['TILENUMBER', 'geometry']]




@np.vectorize
def getLS(slope_val, flow_acc):
    '''LS value from a cell.
    From https://jblindsay.github.io/wbt_book/available_tools/geomorphometric_analysis.html#SedimentTransportIndex'''
    if flow_acc<0:
        return 0
    if slope_val<0:
        return 0
    slope_rads=math.tan(slope_val/100) #https://www.archtoolbox.com/representation/geometry/slope.html
    res= 1.4*((flow_acc/22.13)**.4)*(math.sin(slope_rads)/.0896)**1.3
    return res

@np.vectorize
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
    

@np.vectorize
def rkls(LS, k, r=83):
    '''Calculate rkls (potential erosion) from LS, k and r.
    R=82 is approximate value for Addison County: 
    Map from here: https://www.engr.colostate.edu/~pierre/ce_old/Projects/linkfiles/Cooper%20R-factor-Final.pdf
    on page 15, shows it as >80 and <100, with the line for 80 VERY close to county line.
    '''
    
    return LS*k*r

   
def extract_K_facs(LS_raster, soil_shp, out_dir):
    '''Extract K values from the soils shape layer, 
    to fit over  and match the Length-slope Raster.'''
    x_min,  y_min, x_max, y_max = LS_raster.bounds
    xRes, yRes=LS_raster.transform[0], -LS_raster.transform[4]
    ds = gdal.Rasterize(os.path.join(out_dir, 'K_factors.tif'), soil_shp, xRes=xRes, yRes=yRes, 
                        outputBounds=[x_min, y_min, x_max, y_max], 
                        outputType=gdal.GDT_Float32, attribute='K_factor')
    ds= None
    
    return rasterio.open(os.path.join(out_dir, 'K_factors.tif'))
      
        
def calculateRKLS(directory, soil_shp):
    '''Run all steps toCalculate the erosion potential raster for a given Watershed.'''
    dem_path=os.path.join(directory, 'H2O_shed_DEM.img')  
    LS_raster=calculateLS(dem_path, directory)
    k_raster=extract_K_facs(LS_raster, soil_shp, directory)
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
    
    L_raster, k_raster=None, None #clear out for memory useage. 
    
    RKLS_vals=rkls(LS_data, k_data)
    out_path=os.path.join(out_dir, 'RKLS.tif')
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(RKLS_vals.astype(rasterio.float32), 1)    

def calculateLS(dem_path, out_dir):
    '''Calculate the length-slope factor using routines from pygeoprocessing.
    Return the opened LS raster. '''
    
    #set out_paths of raster to be generated. 
    stream_path=os.path.join(out_dir, 'stream.img')
    pit_filled_dem_path=os.path.join(out_dir, 'pit_filled_dem.img')
    slope_path=os.path.join(out_dir, 'slope_raster.img')
    flow_direction_path=os.path.join(out_dir, 'flow_direction.img')
    flow_accumulation_path=os.path.join(out_dir, 'flow_acc.img')
    
    #avg_aspect_path=os.path.join(out_dir, 'avg_aspect.img')
    
    
    
    
    
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
        print('Routing Error')
        pass
    
    #load in data for calculating LS:
        
    with rasterio.open(slope_path) as slope_raster:
        slope_array=slope_raster.read(1)
    with rasterio.open(flow_accumulation_path) as flow_acc_raster:
        flow_array=flow_acc_raster.read(1)
    
    slope_array=enforce_non_neg(slope_array)
    flow_array=enforce_non_neg(flow_array)
    
    print (time.localtime())
    print('calculating LS')
    
    LS=getLS(slope_array, flow_array)
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

#%%
def merge_tiles_raster(tile_codes, dems_dir):
    '''Merge a set of rasters, given a list of tile_codes 
    and a directory that they are located in.
    Return the raster, its out-transoform and out_metadata.'''
    
    src_to_merge=[]
    file_list=os.listdir(dems_dir)
    for code in tile_codes:
        files=[f for f in file_list if code in f]
        if files:
            src_to_merge.append(rasterio.open(os.path.join(dems_dir, files[0])))
    
    if not src_to_merge:
        return
    
    rast, out_transform=merge(src_to_merge)
    
    out_meta = src_to_merge[0].meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": rast.shape[1],
                    "width": rast.shape[2],
                     "transform": out_transform,
                    }
                   )
    
    for r in src_to_merge:
        r.close()
        
    return rast, out_transform, out_meta
#%%   
def aoi_raster(tiles, dems_dir, out_dir):
    '''Make a raster of the DEM of an aoi defined by a set of tiles. '''
    
    rast, out_transform, out_meta=merge_tiles_raster(tiles, dems_dir)
    out_path=os.path.join(out_dir, 'DEM.tif')
    #plt.imshow(rast[0])
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(rast)



def watershed_raster(HUC12, H2Oshed_tiles, dems_dir, out_dir):
    '''Make a DEM raster for a watershed. Save it as out_dir\dem. '''
    
    #collect rasters to merge together
    
    tile_codes=H2Oshed_tiles[HUC12]
    print(tile_codes)
    rast, out_transform, out_meta=merge_tiles_raster(tile_codes, dems_dir)
    
    out_path=os.path.join(out_dir, 'H2O_shed_DEM.img')
    
    
    plt.imshow(rast[0])
    plt.show()
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(rast)
    
    
    #resize the raster
    resize_raster(out_path, 4)
    











#%%
def make_H2Oshed_tiles(aoi_path, ref_map, watershed_path):
    '''Make a dictionary:
        key- HUC12 number.
        value: list of all DEM tiles needed to perform calculatons. 
    '''
    
    
    aoi=gpd.read_file(aoi_path)
    crs=aoi.crs
    aoi['null']=0
    aoi=aoi.dissolve(by='null')[['geometry', 'AREASYMBOL']]
    ref_map.to_crs(crs, inplace=True)
    aoi['geometry']=aoi.buffer(800)
    ref_map2=gpd.overlay(aoi, ref_map, how='intersection')
    ref_map2.plot()
    out_path=os.path.join(os.getcwd(), 'intermediate_data', 'cells.shp')
    
    
    
    h2Oshed=gpd.read_file(watershed_path)
    h2Oshed.to_crs(ref_map2.crs)
    ext=gpd.overlay(ref_map2, h2Oshed)
    out_path=os.path.join(os.getcwd(), 'intermediate_data', 'cells.shp')
    ext.to_file(out_path)
    
    
    
    h2Oshed=None
    ref_map=None
    ref_map2=None
    
    ext.plot(column='HUC12')
    plt.show()
    
    
    H2Oshed_tiles={}
    for H2Oshed in ext['HUC12'].unique():
        df=ext[ext['HUC12']==H2Oshed]
        H2Oshed_tiles[H2Oshed]=[t for t in df['TILENUMBER'].unique() if (type(t)==str or type==int) ]
    
    df=None
    
    next_map=None
    
    return H2Oshed_tiles, ext
#%%
def make_aoi_tiles(aoi_path, ref_map):
    aoi=gpd.read_file(aoi_path)
    crs=aoi.crs
    aoi['null']=0
    aoi=aoi.dissolve(by='null')[['geometry', 'AREASYMBOL']]
    ref_map.to_crs(crs, inplace=True)
    aoi['geometry']=aoi.buffer(200)
    ref_map2=gpd.overlay(aoi, ref_map, how='intersection')
    ref_map2.plot()
    out_path=os.path.join(os.getcwd(), 'intermediate_data', 'cells.shp')
    
    ref_map2.to_file(out_path)
    ref_map2.plot()
    
    
    ref_map=None
    
    
    plt.show()
    tiles= [t for t in ref_map2['TILENUMBER'].unique() if type(t)==str]
    print(tiles)
    return tiles

#%%
def main_full_aoi(county_code):
    aoi_path=os.path.join(os.getcwd(), 'intermediate_data', f'Geologic_{county_code}_poly.shp')
    
    dems_dir=os.path.join(os.getcwd(), 'source_data', 'DEM_rasters')
    
    main_out_dir=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')
    


    #read in reference maps of the DEM raster grid into one shapefile
    ref_maps=[]
    for p in [f for f in os.listdir(dems_dir) if os.path.isdir(os.path.join(dems_dir, f))]:
        path=os.path.join(dems_dir, p, f'{p[1:]}.shp')
        print(path)
        m=gpd.read_file(path)
        ref_maps.append(m)
   
    
    #combine into one ref map
    ref_map=ref_maps.pop()
    while ref_maps:
        next_map=ref_maps.pop()
        next_map.to_crs(ref_map.crs, inplace=True)
        ref_map=merge_tiles(ref_map, next_map)
    m=None    
    ref_map.plot()
    
    
    
    soil_shp= os.path.join(os.getcwd(), 'intermediate_data', f'Geologic_{county_code}_poly.shp') 
    
    out_dir=os.path.join(os.getcwd(), 'intermediate_data', 'USLE', county_code)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    soil_shp= os.path.join(os.getcwd(), 'intermediate_data', f'Geologic_{county_code}_poly.shp') 
    tiles=make_aoi_tiles(aoi_path, ref_map)
    aoi_raster(tiles, dems_dir, out_dir)
    calculateRKLS(out_dir, soil_shp)
    return tiles
    


def make_ref_map(dems_dir):
    '''Create a Reference map of available tiles'''
    ref_maps=[]
    for p in [f for f in os.listdir(dems_dir) if os.path.isdir(os.path.join(dems_dir, f))]:
        path=os.path.join(dems_dir, p, f'{p[1:]}.shp')
        print(path)
        m=gpd.read_file(path)
        ref_maps.append(m)
   
    
    #combine into one ref map
    ref_map=ref_maps.pop()
    while ref_maps:
        next_map=ref_maps.pop()
        next_map.to_crs(ref_map.crs, inplace=True)
        ref_map=merge_tiles(ref_map, next_map)
    m=None    
    ref_map.plot()
    plt.show()
    return ref_map


def main_by_watershed(county_code):
    '''Run RKLS and intermediate steps for all HUC12 watersheds in the county.
    All rasters for a given watershed are saved to a directory of that name. '''
    
    aoi_path=os.path.join(os.getcwd(), 'intermediate_data', f'Geologic_{county_code}_poly.shp')
    
    dems_dir=os.path.join(os.getcwd(), 'source_data', 'DEM_rasters')
    
    main_out_dir=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')
    
    watershed_path=os.path.join(os.getcwd(), 'source_data', 'VT_Subwatershed_Boundaries_-_HUC12-shp', 'VT_Subwatershed_Boundaries_-_HUC12.shp')


    #read in reference maps of the DEM raster grid into one shapefile
    
    ref_map=make_ref_map(dems_dir)
    
    soil_shp= os.path.join(os.getcwd(), 'intermediate_data', f'Geologic_{county_code}_poly.shp') 
  
    H2Oshed_tiles, ext=make_H2Oshed_tiles(aoi_path, ref_map, watershed_path)
    for HUC12_code in H2Oshed_tiles:    
        print(HUC12_code)
        out_dir=os.path.join(main_out_dir, HUC12_code)
        if all([(not os.path.exists(out_dir)),
                HUC12_code in ext['HUC12'].unique(),
                H2Oshed_tiles[HUC12_code]
                ]):
            os.makedirs(out_dir)
            
            watershed_raster(HUC12_code, H2Oshed_tiles, dems_dir, out_dir)
            calculateRKLS(out_dir, soil_shp)
            with open(os.path.join(out_dir, f'tiles.txt'), 'w') as f:
                for line in H2Oshed_tiles[HUC12_code]:
                    print(line, file=f)
            
            #save a list of tiles in the watershed
            
            
            
            
if __name__=='__main__':
    county_code='SO01'
    main_by_watershed(county_code)
    

#%%
