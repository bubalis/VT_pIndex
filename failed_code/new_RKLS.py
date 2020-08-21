# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:39:32 2020

@author: benja
"""


import os
import geopandas as gpd

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import re 
import math
from shapely.geometry import LineString, MultiLineString




from whitebox import WhiteboxTools
import gdal
import ogr
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.mask import mask
from rasterio import features
from load_in_funcs import load_counties, load_aoi
from RKLS_by_watershed2 import *
from rasterio.features import shapes
from gdalconst import GA_ReadOnly
from rasterstats import zonal_stats

main_dir=os.getcwd()
main_out_dir=os.path.join(os.getcwd(), 'intermediate_data', "USLE")
scratch_dir=os.path.join(main_out_dir, 'scratch')
globals().update(setup_globals([1]))
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)
stream_path=os.path.join("intermediate_data","waterways.shp")
streams=gpd.read_file(stream_path)


HUC12='041504020502'
#%%


def LS_factors(length, slope):
    S1=10.8*math.sin(slope)+0.03
    S2=16.8*math.sin(slope)-.05
    is_steep=slope>=.09
    S=S1*(not is_steep)+S2*is_steep
    
    
    return (length/22.13)**m

def mask_raster(raster_path, shapes, out_path=None, **mask_kwargs):
    '''Mask a raster with the geometries given in shapes.
    Save to out_path. If out_path is not specified, save to original path.'''
    with rasterio.open(raster_path) as src:
        print(src.shape)
        out_image, out_transform = mask(src, shapes)
        out_meta = src.meta.copy()
        print(out_meta)
        print(out_image.shape)
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform}, 
               )
    
    if not out_path:
        out_path=raster_path
        
    with rasterio.open(out_path, "w+", **out_meta) as dest:
        dest.write(out_image)

def merge_tiles_raster(tile_codes, dems_dir):
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
        masked_files.append(raster_path)
    
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


def watershed_raster(HUC12, H2Oshed_tiles, dems_dir, out_dir, mask_geometry,crs, watershed_geom):
    '''Make a DEM raster for a watershed. Save it as out_dir\dem. '''
    #collect rasters to merge together
    
    tile_codes=H2Oshed_tiles[HUC12]
    print(tile_codes)
    rast, out_transform, out_meta=merge_tiles_raster(
                                    tile_codes, dems_dir, watershed_geom)
    
    try:
        plt.imshow(rast[0])
        plt.show()
    except:
        pass
    
    
    
    out_path=os.path.join(out_dir, 'H2O_shed_DEM.tif')
    
    
    with rasterio.open(out_path, 'w+', **out_meta) as dst:
        dst.write(rast)
    
    del rast
    
    mask_raster(out_path, mask_geometry, out_path)
    reproject_rast(out_path, {'epsg': mask_geometry.crs.to_epsg()}, out_path)
    with rasterio.open(out_path) as r:
        show(r)
    return out_meta


def watershed_streams(shed_shp, streams, out_dir):
    pointer=os.path.join(out_dir, 'pointer.tif')
    with rasterio.open(pointer) as example:
        x_min,  y_min, x_max, y_max = example.bounds
        xRes, yRes=example.transform[0], -example.transform[4]
        
    stream_shpfile=os.path.join(scratch_dir, 'streams.shp')
    stream_rast=os.path.join(scratch_dir, 'streams_partial.tif')
    sec_streams=os.path.join(scratch_dir, 'streams_partial2.tif')
    partial_streams=gpd.clip(streams, shed_shp)
    #partial_streams['geometry']=partial_streams.geometry.buffer(10)
    partial_streams.to_file(stream_shpfile)
    Feature_to_Raster(stream_shpfile, stream_rast, pointer, field_name='Water_ID')
    #clip_rast_to_rast(sec_streams, pointer, sec_streams)
    #clip_rast_to_rast(pointer, sec_streams, pointer)
    assert same_raster_extents(pointer, stream_rast)
    
    
def Feature_to_Raster(input_shp, output_tiff, snap_raster,
                      field_name=False, NoData_value=-9999):
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
                                   1, gdal.GDT_Int16)

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

def ensure_dir():
    if os.getcwd()!=main_dir:
        os.chdir(main_dir)
        
def get_IDNum(row):
     for col in [c for c in row.index if 'TILENUM' in c]:
         if type(row[col])==str and row[col]:
             return row[col]
         
def give_first_valid(iterable):
    '''Return the first non ''Falsy' value in an iterable'''
    for i in iterable:
        if i:
            return i

def run_watershed(ext, HUC12, H2Oshed_tiles, watersheds, soil_shp, all_buffers, crop_fields, 
                  crs, main_out_dir, dems_dir):
    print(HUC12)
    out_dir=os.path.join(main_out_dir, HUC12)
    tiles=list_tiles_to_merge(H2Oshed_tiles[HUC12], dems_dir)
    if all([(not os.path.exists(out_dir)),
            HUC12 in ext['HUC12'].unique(),
            H2Oshed_tiles[HUC12],
            not buffer_geometry.empty,
            tiles
            ]):
        os.makedirs(out_dir)
        buffers=gpd.overlay(ext[ext['HUC12']==HUC12], buffers)
        buffers['geometry'].to_file(os.path.join(out_dir, 'buffers.shp'))
        
        shed_shp=watersheds[watersheds['HUC12']==HUC12]
        
        watershed_dem(HUC12, H2Oshed_tiles, 
                         dems_dir, out_dir, 
                          shed_shp)
        hillslope_gdf=hillslopes(shed_shp, streams,out_dir)
        
def watershed_dem(HUC12, H2Oshed_tiles, dems_dir, out_dir, shed_shp):
    tile_codes=H2Oshed_tiles[HUC12]
    print(tile_codes)
    rast, out_transform, out_meta=merge_tiles_raster(
                                    tile_codes, dems_dir)
    
    try:
        plt.imshow(rast[0])
        plt.show()
    except:
        pass
    
    

    out_path=os.path.join(out_dir, 'H2O_shed_DEM.tif')
    
    
    with rasterio.open(out_path, 'w+', **out_meta) as dst:
        dst.write(rast)
    
    # del rast     
    crs={'init': f'epsg: {shed_shp.crs.to_epsg()}'}
    reproject_rast(out_path, crs, out_path)
    mask_raster(out_path, shed_shp.geometry, out_path, crop=True)
    
    with rasterio.open(out_path) as r:
        show(r)
    return out_meta        
        
#%%
    
    
 
    
def polygonize(raster_path, mask=None):
    
    with rasterio.open(raster_path) as src:
        array = src.read(1) # first band
        results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(array, transform=src.transform)))
    geoms = list(results)
    gdf= gpd.GeoDataFrame.from_features(geoms)
    return gdf

def hillslopes(shed_shp, streams, out_dir):
    wbt=WhiteboxTools()
    ensure_dir()
    dem=os.path.join(out_dir, 'H2O_shed_DEM.tif')
    pit_filled=os.path.join(out_dir, 'pit_filled_dem.tif')
    pointer=os.path.join(out_dir, 'pointer.tif')
    streams_path=os.path.join(scratch_dir, 'streams_partial.tif')
    hillslope_path=os.path.join(out_dir, 'hillslopes.tif')
    ensure_dir()
    wbt.breach_depressions(dem, pit_filled)
    ensure_dir()
    wbt.d8_pointer(pit_filled, pointer)
    watershed_streams(shed_shp, streams, out_dir)
    ensure_dir()
    hiilslope_shp=os.path.join(out_dir, 'hillslopes.shp')
    
    
    assert same_raster_extents(pointer, streams_path)
    
    
    wbt.hillslopes(pointer, streams_path, hillslope_path)
    hillslope_gdf=polygonize(hillslope_path, mask=None)
    hillslope_gdf=hillslope[hillslope_gdf.geometry.area>20]
    hillslope_gdf.to_file(hillslope_path)
    
    d_inf_pointer=os.path.join(out_dir, 'd_inf.tif')
    slope_path=os.path.join(out_dir, 'slope.tif')
    wbt.d_inf_pointer(pit_filled,  d_inf_pointer)
    wbt.slope(pit_filled, output)
    wbt.clip(hillslope_shp, os.path.join(out_dir, 'buffers.shp',), hillslope_shp )
    hillslope_gdf=gpd.read_file(hillslope_shp)
    
    for raster in [pit_filled, slope_path, d_inf_pointer]:
        wbt.clip_raster_to_polygon(i, polygons, output)
    

def RKLS_from_hillslopes(hillslope_gdf, out_dir):
    angles=zonal_stats(hillslope_gdf, os.path.join(out_dir, 'd_inf.tif'), stats=['mean', 'median'])
    slopes=zonal_stats(hillslope_gdf, os.path.join(out_dir, 'slope.tif'), stats=['mean'])
    hillslope_gdf['flow_angle']=[a['mean'] for a in angles]
    hillslope_gdf['slope']=[s['mean'] for s in slopes]
    
    slope_lengths=[]
    for i, row in hillslope_gdf.iterrows():
        shape=row['geometry']
        rot=shapely.affinity.rotate(shape, row['flow_angle'])
        slope_lengths.append(height_at_vertical(rot))
    
    hillslop_gdf['slope']=slopes
    hillslope_gdf['LS_factor']=hillslope_gdf['slope']
    

LS = (As
/22.13)m × (sinβ/0.0896)n

def height_at_vertical(shape):
    coords=sorted(list(shape.boundary.coords))
    low_point=coords[0]
    while True:
        high_point= coords.pop()
        height_line=LineString([(high_point[0], high_point[1]), (low_point[0], high_point[1])])
        bottom_point=list(height_line.intersection(shape).coords)[1]
        if bottom_point:
            return high_point[0]-bottom_point[0]
    
    

def same_raster_extents(path1, path2):
    with rasterio.open(path1) as r1:
        with rasterio.open(path2) as r2:
            print (r1.crs, r2.crs)
            if  not (r1.crs==r2.crs):
                return False
            print(r1.transform[0], r2.transform[0])
            print(r1.read(1).shape, 'n', r2.read(1).shape)
            print (r1.bounds, '\n', r2.bounds)
            if not (r1.read(1).shape==r2.read(1).shape and
                    r1.bounds==r2.bounds):
                return False
            return True


def raster_to_shpfile(src_path, band_num, dst_layername):
    src_ds = gdal.Open( src_path)
    assert src_ds
    srcband = src_ds.GetRasterBand(band_num)
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp" )
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )
    gdal.Polygonize( srcband, None, dst_layer, -1, [], callback=None )
    return dst_layer

def make_streams(watersheds, ext):
    '''Make a shapefile of all bodies of water in the ext.
    assumes streams are width= 0, while rivers/creeks are actually mapped in their boundaries. '''
    
    
    stream_path=os.path.join('source_data', 'NHD_H_Vermont_State_Shape', 
                             "Shape", 'NHDFlowline.shp')
    
    
    
    #rivers:    
    river_path=os.path.join( 'source_data', 'NHD_H_Vermont_State_Shape', 
                            "Shape", 'NHDArea.shp')
    
    #ponds, lakes:
    bodies_path=os.path.join('source_data', 'NHD_H_Vermont_State_Shape',
                             "Shape", 'NHDWaterbody.shp')
    #other areas?
    area_path=os.path.join('source_data', 'NHD_H_Vermont_State_Shape', 
                           "Shape", 'NHDArea.shp')    
    
    paths=[stream_path, river_path, bodies_path, river_path]
    streams=gpd.geopandas.pd.concat([
            gpd.read_file(path) for path in paths])
    
    streams.to_crs(ext.crs, inplace=True)
    
    streams.reset_index( inplace=True)
    
    streams=gpd.clip(streams, ext)
    streams['geometry']=streams.geometry.buffer(1.5)
    
    streams['Water_ID']=streams.index
    save_path=os.path.join(os.getcwd(), 'intermediate_data', 'waterways.shp')
    #streams=streams.drop(columns=['index_right'])
    streams['geometry']=streams.geometry.buffer(5)
    streams.to_file(save_path)
    
    return streams

def geom_to_line(geom):
    if type (geom)==LineString or type (geom)==MultiLineString:
        return geom
    elif type(geom)==tuple:
        return MultiLineString(geom)
    elif type(geom)==shapely.geometry.Polygon:
        return geom.boundary
    return LineString(geom)



def pad_raster_to_template(r, template):
    add_to_left=r.bounds[0]-template.bounds[0]
    add_to_right=r.bounds[2]-template.bounds[2]
    add_to_top=template.bounds[3]-r.bounds[3]
    add_to_bottom=r.bounds[1]-templat.bounds[1]

#%%

def clip_rast_to_rast(rast_to_mask, masking_raster, out_path):
    with rasterio.open(masking_raster) as mask_rast:
        b=mask_rast.bounds
        mask_shp=Polygon([(b[0], b[1]), 
                          (b[0], b[3]),  
                          (b[2], b[3]), 
                          (b[2], b[1]) ])
        mask_shp=gpd.GeoDataFrame(geometry=[mask_shp])
        out_meta=mask_rast.meta.copy()
    with rasterio.open(rast_to_mask) as src:
        
        mask_shp.plot()
        
        plt.show()
        array, out_transform=mask(src, mask_shp.geometry, crop=True)
        out_meta.update({"transform": out_transform, 
                         'dtype': src.meta['dtype']})
    
    with rasterio.open(out_path, "w+", **out_meta) as dest:
        dest.write(array)
        

def mask_rast_to_rast(rast_to_mask, masking_raster, output):
    maskDs = gdal.Open(masking_raster, GA_ReadOnly)# your mask raster
    projection=maskDs.GetProjectionRef()
    geoTransform = maskDs.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * maskDs.RasterXSize
    miny = maxy + geoTransform[5] * maskDs.RasterYSize
    
    
    data=gdal.Open(rast_to_mask, GA_ReadOnly) #Your data the one you want to clip
    gdal.Translate(output,data,format='GTiff',projWin=[minx,maxy,maxx,miny],outputSRS=projection)     
    

def clean_pointer_to_streams(pointer, stream_rast):
    with rasterio.open(pointer) as src:
        p_array=src.read(1)
        with rasterio.open(stream_rast) as stream:
           
            s_array=stream.read(1)
            s_array=np.where(s_array==-9999, 1, 0)
            p_array=p_array*s_array
            out_meta=src.meta.copy()
            plt.imshow(p_array)
            
    with rasterio.open(pointer, 'w+', **out_meta) as dst:
        dst.write(np.array([p_array]))
        