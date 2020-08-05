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
import re 
import math
import pygeoprocessing
import pygeoprocessing.routing as pyg_routing
import gdal
import time
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
from rasterio import MemoryFile
from rasterio.plot import show
from rasterio.mask import mask

from load_in_funcs import load_counties, load_aoi

#%%
global crs

def try_open_raster(raster_path):
    try:
        return rasterio.open(raster_path)
    except rasterio.RasterioIOError:
        print(f"Raster     {raster_path}    is corrupted")

def get_tileNum(row):
     for col in [c for c in row.index if 'TILENUM' in c]:
         if type(row[col])==str and row[col]:
             return row[col]
         
def give_first_valid(iterable):
    '''Return the first non ''Falsy' value in an iterable'''
    for i in iterable:
        if i:
            return i





@np.vectorize
def getLS(slope_val, flow_acc):
    '''LS value from a cell.
    From https://jblindsay.github.io/wbt_book/available_tools/geomorphometric_analysis.html#SedimentTransportIndex'''
    if flow_acc<0:
        return 0
    if slope_val<0:
        return 0
    slope_rads=math.tan(slope_val/100) #https://www.archtoolbox.com/representation/geometry/slope.html
    return  1.4*((flow_acc/22.13)**.4)*(math.sin(slope_rads)/.0896)**1.3
    

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

   
def extract_K_facs(soil_shp, out_dir):
    '''Extract K values from the soils shape layer, 
    to fit over  and match the Length-slope Raster.'''
    with rasterio.open(os.path.join(out_dir, 'LS.tif')) as LS:
        x_min,  y_min, x_max, y_max = LS.bounds
        xRes, yRes=LS.transform[0], -LS.transform[4]
    path_to_save=os.path.join(out_dir, 'K_factors.tif')
    
    
    ds = gdal.Rasterize(path_to_save, soil_shp, xRes=xRes, yRes=yRes, 
                        outputBounds=[x_min, y_min, x_max, y_max], 
                        outputType=gdal.GDT_Float32, attribute='K_factor')
    

    return #rasterio.open(path_to_save)


      
        
def calculateRKLS(directory, soil_shp):
    '''Run all steps toCalculate the erosion potential raster for a given Watershed.'''
    dem_path=os.path.join(directory, 'H2O_shed_DEM.tif')  
    calculateLS(dem_path, directory)
    extract_K_facs(soil_shp, directory)
    makeRKLS_raster(directory)

def makeRKLS_raster(out_dir):
    '''Final step to make RKLS raster'''
    print(time.localtime())
    print('Making RKLS raster')
    
    LS_raster=rasterio.open(os.path.join(out_dir, 'LS.tif'))
    k_raster=rasterio.open(os.path.join(out_dir, 'K_factors.tif'))
    
    profile=LS_raster.profile.copy()
    
    LS_data=LS_raster.read(1)
    k_data=k_raster.read(1)
    
    LS_raster, k_raster=None, None #clear out for memory useage. 
    
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
    
    #
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
    
    
    #make sure values are valid
    slope_array=enforce_non_neg(slope_array)
    flow_array=enforce_non_neg(flow_array)
    
    
    print (time.localtime())
    print('calculating LS')
    
    LS=getLS(slope_array, flow_array)
    
    #clear memory
    slope_array=None
    flow_array=None
    
    
    with rasterio.open(dem_path) as dem:
        profile=dem.profile
        
    profile.update({'nodata': 0})
    
    print (time.localtime())
    print("Writing LS raster")
    with rasterio.open(os.path.join(out_dir, 'LS.tif'), 'w', **profile) as dst:
        dst.write(LS.astype(rasterio.float32), 1)
    
    #LS_raster=rasterio.open(os.path.join(out_dir, 'LS.tif'))
    
    return #LS_raster


def resize_raster(filepath, factor, out_path):
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
                     "driver": "GTiff",}
                   )
    with rasterio.open(out_path, 'w+', **out_meta) as dest:
        dest.write(data)

#%%
def list_tiles_to_merge(tile_codes, dems_dir):
    src_to_merge=[]
    file_list=os.listdir(dems_dir)
    
    
    for code in tile_codes:    
        files=[f for f in file_list if code in f]
        if files:
            src_to_merge.append(os.path.join(dems_dir, files[0]))
    
    if not src_to_merge:
        print("No rasters to merge")
    return src_to_merge


def merge_tiles_raster(tile_codes, dems_dir, watershed_geom):
    '''Merge a set of rasters, given a list of tile_codes 
    and a directory that they are located in.
    Return the raster, its out-transoform and out_metadata.'''
    global corrupted_rasters
    na_val=-9999
    
    for file in os.listdir('scratch'):
        os.remove(os.path.join('scratch', file))
        
    masked_files=[]
    files_to_merge=list_tiles_to_merge(tile_codes, dems_dir)
    for raster_path in files_to_merge:
        out_path=os.path.join('scratch', raster_path.split('\\')[-1])
        try:
            mask_raster(raster_path, watershed_geom, out_path)
            masked_files.append(out_path)
        except ValueError:
            try:
                ax=show(rasterio.open(raster_path))
                watershed_geom.plot(ax=ax)
                plt.show()
                continue
            except:
                print('File Corrupted:' )
                print(raster_path)
                corrupted_rasters.append(raster_path)
    
    src_to_merge=[try_open_raster(src) for src in masked_files]
    src_to_merge=[src for src in src_to_merge if src]
        
    rast, out_transform=merge(src_to_merge)
    out_meta = src_to_merge[0].meta.copy()
    out_meta.update({
                     "height": rast.shape[1],
                    "width": rast.shape[2],
                     "transform": out_transform,
                    }
                   )
    for src in src_to_merge:
        src.close()
    for file in os.listdir('scratch'):
        os.remove(os.path.join('scratch', file))
    return rast, out_transform, out_meta


 

    
def mask_raster(raster_path, shapes, out_path=None):
    '''Mask a raster with the geometries given in shapes.
    Save to out_path. If out_path is not specified, save to original path.'''
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
    
    if not out_path:
        out_path=raster_path
        
    with rasterio.open(out_path, "w+", **out_meta) as dest:
        dest.write(out_image)


def reproject_rast(raster_path, dst_crs, out_path=None):
    if not out_path:
        out_path=raster_path
        
    with rasterio.open(raster_path, 'r') as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
        })
        transform=src.transform
        src_crs=src.crs
        reproj_kwargs={
        'source': src.read(1),
        'src_transform':transform,
        'src_crs':src_crs,
        'dst_transform':transform,
         'dst_crs':dst_crs,
         'resampling':Resampling.nearest}
            
    with rasterio.open(out_path, 'w+', **kwargs) as dst:
        reproject(**reproj_kwargs, destination=np.zeros((height, width)))  
    assert os.path.exists(out_path)
#%%   
def aoi_raster(tiles, dems_dir, out_dir):
    '''Make a raster of the DEM of an aoi defined by a set of tiles. '''
    
    rast, out_transform, out_meta=merge_tiles_raster(tiles, dems_dir)
    out_path=os.path.join(out_dir, 'DEM.tif')
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(rast)



def watershed_raster(HUC12, H2Oshed_tiles, dems_dir, out_dir, buffer_geometry,crs, watershed_geom):
    '''Make a DEM raster for a watershed. Save it as out_dir\dem. '''
    #collect rasters to merge together
    
    tile_codes=H2Oshed_tiles[HUC12]
    print(tile_codes)
    rast, out_transform, out_meta=merge_tiles_raster(tile_codes, dems_dir, watershed_geom)
    
    try:
        plt.imshow(rast[0])
        plt.show()
    except:
        pass
    
    dst_crs={'init':crs.to_string()}

    out_path=os.path.join(out_dir, 'H2O_shed_DEM.tif')
    
    
    with rasterio.open(out_path, 'w+', **out_meta) as dst:
        dst.write(rast)
    
    del rast
    
    mask_raster(out_path, buffer_geometry, out_path)
    with rasterio.open(out_path) as r:
        show(r)
    

    










#%%

def make_ext_gdf(aoi, ref_map, watersheds, crs):
    '''Make a geo-dataframe defining the extent of all operations.'''
    
    
    aoi['null']=0
    aoi=aoi.dissolve(by='null')[['geometry', 'CNTY']]
    aoi['geometry']=aoi.buffer(50)
    ref_map2=gpd.overlay(aoi, ref_map, how='intersection')
    ref_map2.plot()
    out_path=os.path.join(os.getcwd(), 'intermediate_data', 'cells.shp')
    
    
    ext=gpd.overlay(ref_map2, watersheds)
    ext.plot(column='HUC12')
    plt.show()
    if not os.path.exists(out_path):
        ext.to_file(out_path)
    return ext
    

def make_H2Oshed_tiles(aoi, ref_map, watersheds):
    '''Make a dictionary:
        key- HUC12 number.
        value: list of all DEM tiles needed to perform calculatons. 
    '''
    ext=make_ext_gdf(aoi, ref_map, watersheds, aoi.crs)

    
    #memory clearance
    ref_map=None
    
    H2Oshed_tiles={}
    for H2Oshed in ext['HUC12'].unique():
        df=ext[ext['HUC12']==H2Oshed]
        li=[]
        for col in [col for col in ext.columns if 'Orig' in col]:
            li+=[t for t in df[col].unique() if t ]
        H2Oshed_tiles[H2Oshed]=li
    
    #memory clearance
    df=None
    next_map=None
    
    return H2Oshed_tiles, ext

#%%
def merge_tiles(tiles_1, tiles_2, crs):
    '''Merge two shapefiles delineating tile geometries'''
    print('Merging Tiles')
    tiles_2.to_crs(crs, inplace=True)
    merged=gpd.overlay(tiles_1, tiles_2, how='union')
    merged['TILE_NUMBER']=merged.apply(get_tileNum, axis=1)
    
    #renaming columns
    cols_to_rename=[col for col in merged.columns if 'TILENUM' in col]
    n=max([int(col[-1]) for col in merged.columns if 'Orig_tilenum' in col]+[0])
    merged.rename(columns={col: f'Orig_tilenum{i+n+1}'  for i, col in enumerate(cols_to_rename)}, inplace=True )

    cols_to_keep=['TILE_NUMBER', 'geometry']+[c for c in merged.columns if 'Orig_tile' in c]
    return merged[cols_to_keep]


def first_non_missing(*args, na_val=0):
    for arg in args:
        if arg!= na_val:
            return arg
    return na_val


def make_ref_map(dems_dir, crs):
    '''Create a Reference map of available tiles'''
    ref_maps=[]
    ref_pathes= [f for f in os.listdir(dems_dir) if os.path.isdir(os.path.join(dems_dir, f))]
    print(ref_pathes)
    #sort by year, latest first.
    for p in sorted(ref_pathes, key=lambda x: int (re.search('\d\d\d\d', x).group()), reverse=True):
        path=os.path.join(dems_dir, p, f'{p[1:]}.shp')
        print(path)
        m=gpd.read_file(path)
        ref_maps.append(m)
   
    
    #combine into one ref map
    ref_map=ref_maps.pop()
    ref_map.to_crs(crs, inplace=True)
    while ref_maps:
        next_map=ref_maps.pop()
        ref_map=merge_tiles(ref_map, next_map, crs)
   
    m=None    
    ref_map.plot()
    plt.show()
    return ref_map.fillna('')



#%%
def main(county_codes):
    '''Run RKLS and intermediate steps for all HUC12 watersheds in the county.
    All rasters for a given watershed are saved to a directory of that name. '''

    
    dems_dir=os.path.join(os.getcwd(), 'source_data', 'DEM_rasters')
    
    main_out_dir=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')
    
    if not os.path.exists(main_out_dir):
        os.makedirs(main_out_dir)
    
    watershed_path=os.path.join(os.getcwd(), 'source_data', 'VT_Subwatershed_Boundaries_-_HUC12-shp', 'VT_Subwatershed_Boundaries_-_HUC12.shp')
    
    
    
    watersheds=gpd.read_file(watershed_path)
    aoi=load_counties(county_codes)
    
    crs=aoi.crs
    
    watersheds.to_crs(crs, inplace=True)
    
    #read in reference maps of the DEM raster grid into one shapefile   
    ref_map=make_ref_map(dems_dir, crs)
    
    soil_shp= os.path.join(os.getcwd(), 'source_data', 'GeologicSoils_SO', 'GeologicSoils_SO.shp' ) 
  
    
    crop_fields=gpd.read_file(r"C:\Users\benja\VT_P_index\model\P_Index_LandCoverCrops\P_Index_LandCoverCrops\Crop_DomSoil.shp")
    
    crop_fields.to_crs(crs, inplace=True)
    
    buffers=gpd.GeoDataFrame(geometry=crop_fields.buffer(30))
    del crop_fields
    

    H2Oshed_tiles, ext=make_H2Oshed_tiles(aoi, ref_map, watersheds)
    
    for HUC12_code in H2Oshed_tiles:
        buffer_geometry=gpd.overlay(ext[ext['HUC12']==HUC12_code], buffers)
        buffer_geometry=buffer_geometry['geometry']
        print(HUC12_code)
        out_dir=os.path.join(main_out_dir, HUC12_code)
        tiles=list_tiles_to_merge(H2Oshed_tiles[HUC12_code], dems_dir)
        if all([(not os.path.exists(out_dir)),
                HUC12_code in ext['HUC12'].unique(),
                H2Oshed_tiles[HUC12_code],
                not buffer_geometry.empty,
                tiles
                ]):
            os.makedirs(out_dir)
            watershed_geom=watersheds[watersheds['HUC12']==HUC12_code]['geometry'].buffer(100)
            watershed_raster(HUC12_code, H2Oshed_tiles, dems_dir, out_dir, buffer_geometry, crs, watershed_geom)
            del buffer_geometry
            calculateRKLS(out_dir, soil_shp)
            
            
            with open(os.path.join(out_dir, f'tiles.txt'), 'w') as f:
                for line in H2Oshed_tiles[HUC12_code]:
                    print(line, file=f)
            
            #save a list of tiles in the watershed
            
            
def dissolve_to_single_shape(gdf):
    gdf['null']=0
    return gdf.dissolve(by='null')            


if __name__=='__main__':
    corrupted_rasters=[]
    county_codes=[1, 7, 11, 15]
    main(county_codes)
    
#%%
