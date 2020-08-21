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
import ogr


import pygeoprocessing
import pygeoprocessing.routing as pyg_routing
from whitebox import WhiteboxTools
import gdal
import time
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.mask import mask
from rasterio import features
from load_in_funcs import load_counties, load_aoi


#%%
global crs
main_dir=os.getcwd()

def ensure_dir():
    '''Reset directory to main directory.'''
    if os.getcwd()!=main_dir:
        os.chdir(main_dir)


#%%
def Feature_to_Raster(input_shp, output_tiff, snap_raster,
                      field_name=False, NoData_value=-9999, data_type=gdal.GDT_Int16):
    """
    Converts a shapefile into a raster
    From:
    https://www.programcreek.com/python/example/101827/gdal.RasterizeLayer
    #6
    """

    # Input
    inp_driver = ogr.GetDriverByName('ESRI Shapefile')
    inp_source = inp_driver.Open(input_shp, 0)
    inp_lyr = inp_source.GetLayer()
    inp_srs = inp_lyr.GetSpatialRef()

    # Extent
    with rasterio.open(snap_raster) as example:
        cellsize=example.transform[0]
        x_min=example.transform[2]
        y_max=example.transform[5]
        y_ncells, x_ncells = example.shape
        

    # Output
    out_driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_tiff):
        out_driver.Delete(output_tiff)
    out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,
                                   1, data_type)

    out_source.SetGeoTransform((x_min, cellsize, 0, y_max, 0, -cellsize))
    out_source.SetProjection(inp_srs.ExportToWkt())
    out_lyr = out_source.GetRasterBand(1)
    out_lyr.SetNoDataValue(NoData_value)

    # Rasterize
    if field_name:
        gdal.RasterizeLayer(out_source, [1], inp_lyr,
                            options=["ATTRIBUTE={0}".format(field_name)])
    else:
        gdal.RasterizeLayer(out_source, [1], inp_lyr, burn_values=[1])

    # Save and/or close the data sources
    inp_source = None
    out_source = None

    # Return
    return output_tiff 



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


def LS_calculator(sca, slope, aspect, cell_size=2.8):
    '''Calculate LS factor for cell.
    Inputs: sca--> source contributing area in meters^2.
    cell_size: dimensions in meters.
    aspect: aspect of cell in radians.
    slope: slope of cell in radians.
    Adapted from doi:10.3390/geosciences5020117
    
    https://www-jswconline-org.ezproxy.uvm.edu/content/jswc/51/5/427.full.pdf
    '''
    aspect=np.radians(aspect)
    slope=np.radians(slope)
    X=np.abs(np.sin(aspect))+np.abs(np.cos(aspect))
    m=calculate_m(slope)
    numerator=np.power(sca+(cell_size**2),(m+1))-np.power(sca, (m+1))
    
    denominator=np.power(cell_size,(m+2))*np.power(X, m)*(22.13**m)
    L=numerator/denominator
    
    S=S_factor(slope)
    
    LS_array= L*S
    return LS_array, L, m, X



def S_factor(slope):
    '''Calculates slope_factor array for all cells.'''
    slope_grad=np.tan(slope)
    cond_1=10.8*np.sin(slope)+0.3
    cond_2=16.8*np.sin(slope)-.5
    s_array=np.where(slope_grad>=.09, cond_2, cond_1)
    return s_array


def calculate_m(slope):
    '''Make array of m-exponent for USLE LS. 
    Beta is the ratio of rill erosion to inter-rill erosion. '''
    sine=np.sin(slope)
    beta= (sine/.0896)/(.56+3*np.power(sine, .8))
    return beta/(beta+1)

def make_LS_raster(fps):
    '''Make the LS_factor raster.'''
    
    
    
  
    with rasterio.open(fps['pointer']) as src:
        aspect=src.read(1)
        cell_size=src.transform[0]
        out_meta=src.meta.copy()
        
    arrays=[rasterio.open(fps[f]).read(1) for f in ['sca', 'slope']]+[aspect]
    LS, L, m, X=LS_calculator(*arrays, cell_size=cell_size )
    
    for string in ['LS', 'L', 'm', 'X' ]:
        with rasterio.open(fps[string], 'w+', **out_meta) as dst:
            dst.write(np.array([locals()[string]]))
            
        
    
    
    






    


    


def rkls(LS, k, r=83):
    '''Calculate rkls (potential erosion) from LS, k and r.
    R=82 is approximate value for Addison County: 
    Map from here: 
    https://www.engr.colostate.edu/~pierre/ce_old/Projects/linkfiles/Cooper%20R-factor-Final.pdf
    on page 15, shows it as >80 and <100, 
    with the line for 80 VERY close to county line.
    '''
    out_val= LS*k*r
    return np.where(out_val>0, out_val, 0)
   
def extract_K_facs(fps):
    '''Extract K values from the soils shape layer, 
    to fit over  and match the Length-slope Raster.'''
    snap_raster=fps['dem']
    ensure_dir()
    Feature_to_Raster(fps['soils'], fps['K_factors'], snap_raster,
                      field_name='K_factor', NoData_value=-9999, 
                      data_type=gdal.GDT_Float32 )
    
    
    

      
 
def make_local_fps(fps, HUC12_code):
    out_dir=os.path.join(fps['main_out_dir'], HUC12_code)
    for raster in ['LS', 'K_factors','RKLS', 'L', 'm', 'X' , 
                'pit_filled', 'slope', 'sca', 
                   'pointer', 'dem']:
        fps[raster]=os.path.join(out_dir, f'{raster}.tif')
    fps['buffers']=os.path.join(out_dir, 'buffers.shp')
    fps['out_dir']=out_dir    
    return fps
    
 
def calculateRKLS(fps, buffers):
    '''Run all steps toCalculate the erosion potential raster for a given Watershed.'''
    
    all_LS_steps(fps, buffers )
    extract_K_facs(fps)
    makeRKLS_raster(fps)

def makeRKLS_raster(fps):
    '''Final step to make RKLS raster'''
    print(time.localtime())
    print('Making RKLS raster')
    
    LS_raster=rasterio.open(fps["LS"])
    k_raster=rasterio.open(fps['K_factors'])
    
    profile=LS_raster.profile.copy()
    
    LS_data=LS_raster.read(1)
    k_data=k_raster.read(1)
    
    RKLS_vals=rkls(LS_data, k_data)
    
    with rasterio.open(fps['RKLS'], 'w', **profile) as dst:
        dst.write(RKLS_vals.astype(rasterio.float64), 1)    



def all_LS_steps(fps, buffers):
    '''Calculate the length-slope factor using routines from Whitebox Tools.
    Return the opened LS raster. '''
    print(f'Working in {fps["out_dir"]}')
    #set out_paths of raster to be generated.
    
    
    
    wbt=WhiteboxTools()
    
        
        
    
    buffers.to_file(fps['buffers'])
    wbt.breach_depressions(fps['dem'], fps['pit_filled'])
    wbt.slope(fps['pit_filled'], fps['slope'])
    wbt.d_inf_pointer(fps['pit_filled'], fps['pointer'])
    wbt.clip_raster_to_polygon(fps['pointer'], fps['buffers'], fps['pointer'], maintain_dimensions=True)
    wbt.d_inf_flow_accumulation(fps['pointer'], fps['sca'])
    make_LS_raster(fps)
    
    ensure_dir()
    

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
        except (ValueError, rasterio.RasterioIOError):
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



#%%   



def watershed_raster(HUC12, H2Oshed_tiles, 
                         fps, watershed_geom, crs):
    
    
    '''Make a DEM raster for a watershed. Save it as out_dir\dem. '''
    #collect rasters to merge together
    
    tile_codes=H2Oshed_tiles[HUC12]
    print(tile_codes)
    rast, out_transform, out_meta=merge_tiles_raster(
                                    tile_codes, fps['dem_dir'], watershed_geom)
    
    try:
        plt.imshow(rast[0])
        plt.show()
    except:
        pass
    
    dst_crs={'init':crs.to_string()}

    
    
    with rasterio.open(fps['dem'], 'w+', **out_meta) as dst:
        dst.write(rast)
    
    del rast
    
    with rasterio.open(fps['dem']) as r:
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
def merge_tiles_shp(tiles_1, tiles_2, crs):
    '''Merge two shapefiles delineating tile geometries'''
    print('Merging Tiles')
    tiles_2.to_crs(crs, inplace=True)
    merged=gpd.overlay(tiles_1, tiles_2, how='union')
    merged['TILE_NUMBER']=merged.apply(get_tileNum, axis=1)
    
    #renaming columns
    cols_to_rename=[col for col in merged.columns if 'TILENUM' in col]
    n=max([int(col[-1]) for col in merged.columns if 'Orig_tilenum' in col]+[0])
    merged.rename(columns={col: f'Orig_tilenum{i+n+1}'  
                           for i, col in enumerate(cols_to_rename)}, inplace=True )
    
    tile_cols=[c for c in merged.columns if 'Orig_tile' in c]
    cols_to_keep=['TILE_NUMBER', 'geometry']+tile_cols
    return merged[cols_to_keep]



def first_non_missing(*args, na_val=0):
    for arg in args:
        if arg!= na_val:
            return arg
    return na_val


def make_ref_map(dems_dir, crs):
    '''Create a Reference map of available tiles'''
    ref_maps=[]
    ref_pathes= [f for f in os.listdir(dems_dir) if 
                 os.path.isdir(os.path.join(dems_dir, f))]
    
    print(ref_pathes)
    #sort by year, latest first.
    year_finder=lambda x: int (re.search('\d\d\d\d', x).group())
    
    for p in sorted(ref_pathes, key=year_finder, reverse=True):
        path=os.path.join(dems_dir, p, f'{p[1:]}.shp')
        print(path)
        m=gpd.read_file(path)
        ref_maps.append(m)
   
    
    #combine into one ref map
    ref_map=ref_maps.pop()
    ref_map.to_crs(crs, inplace=True)
    while ref_maps:
        next_map=ref_maps.pop()
        ref_map=merge_tiles_shp(ref_map, next_map, crs)
   
    m=None    
    ref_map.plot()
    plt.show()
    return ref_map.fillna('')

#%%
def setup_globals(county_codes):
    '''Run RKLS and intermediate steps for all HUC12 watersheds in the county.
    All rasters for a given watershed are saved to a directory of that name. '''
    
    dems_dir=os.path.join( 'source_data', 'DEM_rasters')
    
    main_out_dir=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')
    
    if not os.path.exists(main_out_dir):
        os.makedirs(main_out_dir)
    
    
    watershed_path=os.path.join( 'source_data', 
                                'VT_Subwatershed_Boundaries_-_HUC12-shp', 
                                'VT_Subwatershed_Boundaries_-_HUC12.shp')
    
    
    
    watersheds=gpd.read_file(watershed_path)
    aoi=load_counties(county_codes)
    
    crs=aoi.crs
    
    watersheds.to_crs(crs, inplace=True)
    
    #read in reference maps of the DEM raster grid into one shapefile   
    ref_map=make_ref_map(dems_dir, crs)
    
    soil_shp= os.path.join('source_data', 
                           'GeologicSoils_SO', 'GeologicSoils_SO.shp' ) 
    
    cf_path=os.path.join('P_Index_LandCoverCrops',
                         'P_Index_LandCoverCrops', 'Crop_DomSoil.shp')
    
    crop_fields=gpd.read_file(cf_path)
    
    crop_fields.to_crs(crs, inplace=True)
    
    all_buffers=gpd.GeoDataFrame({'idnum':crop_fields.index}, geometry=crop_fields.buffer(5))
    

    H2Oshed_tiles, ext=make_H2Oshed_tiles(aoi, ref_map, watersheds)
    fps={'soils': soil_shp, 'dem_dir':dems_dir, 'main_out_dir': main_out_dir}
    
    return locals()
    
            
def run_watershed(HUC12_code, ext,  H2Oshed_tiles, watersheds, all_buffers, fps
                  ,crs):
    '''Complete all operations for a given watershed.'''
    
    
    
    buffers=gpd.overlay(ext[ext['HUC12']==HUC12_code], all_buffers)
    print(HUC12_code)
    fps=make_local_fps(fps, HUC12_code)
    tiles=list_tiles_to_merge(H2Oshed_tiles[HUC12_code], dems_dir)
    if all([(not os.path.exists(fps['out_dir'])),
            HUC12_code in ext['HUC12'].unique(),
            H2Oshed_tiles[HUC12_code],
            not buffers.geometry.empty,
            tiles
            ]):
        
        os.makedirs(fps['out_dir'])
        h2o_subset=watersheds[watersheds['HUC12']==HUC12_code]
        watershed_geom=h2o_subset['geometry'].buffer(100)
        
        watershed_raster(HUC12_code, H2Oshed_tiles, 
                         fps, watershed_geom, crs)
        
        calculateRKLS(fps, buffers)
        
        #save a list of tiles in the watershed
        with open(os.path.join(fps['out_dir'], 'tiles.txt'), 'w') as f:
            for line in H2Oshed_tiles[HUC12_code]:
                print(line, file=f)
           
        
            
def dissolve_to_single_shape(gdf):
    gdf['null']=0
    return gdf.dissolve(by='null')            

#%%
if __name__=='__main__':
    corrupted_rasters=[]
    county_codes=[1, 
                  #7, 
                  #11, 
                  #15
                  ]
    
    globals().update(setup_globals(county_codes))
    for HUC12_code in list(H2Oshed_tiles.keys()):
        run_watershed(HUC12_code, ext,  H2Oshed_tiles, 
                      watersheds, all_buffers, fps
                  ,crs)

    
#%%
