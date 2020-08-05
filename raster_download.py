# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:43:36 2020

@author: benja


Script to download all DEM tiles in aoi for a given county.
Downloads tiles from VCGI.
From these DEM maps:
'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2015/DEMHE/',
'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2017/DEMHF/',
'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2014/DEMHE/',
'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2013/DEMHE/',


"""
import os
import geopandas as gpd
import numpy as np
import time 
import requests
import zipfile
import rasterio.mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import RasterioIOError
from bs4 import BeautifulSoup
from load_in_funcs import load_counties, load_aoi
from RKLS_by_watershed import make_ext_gdf, resize_raster


#%%
dst_crs=CRS.from_epsg(32145)


file_names={'VT_Data_-_County_Boundaries-shp.zip': 'https://opendata.arcgis.com/datasets/2f289dbae90347c58cd1765db84bd09e_29.zip?outSR=%7B%22latestWkid%22%3A32145%2C%22wkid%22%3A32145%7D',
            'VT_Subwatershed_Boundaries_-_HUC12-shp.zip': 'https://opendata.arcgis.com/datasets/3efbd587cc9a4f078be04fa23db5097a_9.zip?outSR=%7B%22latestWkid%22%3A32145%2C%22wkid%22%3A32145%7D',
            'GeologicSoils_SO.zip': 'http://maps.vcgi.vermont.gov/gisdata/vcgi/packaged_zips/GeologicSoils_SO/GeologicSoils_SO.zip',
            "NHD_H_Vermont_State_Shape.zip": 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/HighResolution/Shape/NHD_H_Vermont_State_Shape.zip'}

def make_directories():
    source_dir='Source_data'
    if not os.path.exists('Source_data'):
        os.makedirs('Source_data')

def download_shapefiles():
    for url, file_name in file_names.items():
        download_file(url, 'Source_data', file_name=file_name)
        if file_name:
            print(f'File_name:{file_name}')
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(file_name[:-4])
            os.remove(os.path.join('Source_data'), file_name)
#%%
def reproj_raster(file_path, out_path, crs):
    with rasterio.open(file_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

            
def download_file(url, directory, start_url='', file_name=None): 
    '''Download a file and save it to the given directory.'''
    if not file_name:
        file_name=url.split('/')[-1]
    file_path=os.path.join(directory, file_name)
    if file_name in  os.listdir(directory):
        print('File already in Directory')
        return file_path, False
    
    time.sleep(10)
    r=requests.get(start_url+url)
    print(f'Downloading File:  {url}')
    
    with open(file_path, 'wb') as f:
        f.write(r.content)
    return file_path, True

    
def list_links(b_url):
    '''Get urls of all links from a webpage.
    Args: a url. '''
    page_data=requests.get(b_url)
    soup=BeautifulSoup(page_data.text, features='lxml')
    return [link.get('href') for link in soup.find_all('a')]




def retrieve(county_codes):
    '''Get all DEM raster tiles for a given county code. '''
    
    base_urls=['https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2015/DEMHE/',
               'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2017/DEMHF/',
               'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2014/DEMHE/',
               'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2013/DEMHE/',
              ] 
    
    
    #make aoi_polygon
    

    watershed_path=os.path.join(os.getcwd(), 'source_data', 'VT_Subwatershed_Boundaries_-_HUC12-shp')   
    h2oshed=gpd.read_file(watershed_path)
    
    aoi=load_aoi(county_codes, h2oshed)
    aoi.drop(columns=['HUC12'], inplace=True)

    directory=os.path.join('source_data', 'DEM_rasters')
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    joined_ref_map=gpd.GeoDataFrame()
    for b_url in base_urls:
        links=list_links(b_url)
        
        zip_file_url=[l for l in links if l[-4:]=='.zip'][0]
        file_name, _=download_file(zip_file_url, directory, 'https://maps.vcgi.vermont.gov')
        
        
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(file_name[:-4])
                
            #load the reference map 
        ref_map_file=[f for f in os.listdir(file_name[:-4]) if f[-4:]=='.shp'][0]
        ref_map=gpd.read_file(os.path.join(file_name[:-4], ref_map_file))
        
#crop the ref_map to the area of interest
        ref_map.plot()
        ref_map=ref_map.to_crs(aoi.crs)
        ref_map=gpd.overlay(ref_map, aoi)
        joined_ref_map=joined_ref_map.append(ref_map)
                
    ext=make_ext_gdf(aoi, joined_ref_map, h2oshed, aoi.crs)        
            #download all files in aoi        
    for file_url in ext['DOWNLOAD_P'].unique():
        raster_fp, was_downloaded=download_file(file_url, directory)
        if was_downloaded:
            try:
                resize_raster(raster_fp, 4, raster_fp) #resize raster to size that will be used in USLE Calculations
            except RasterioIOError:
                print(f'Corrrupted Raster:  {raster_fp}')
                continue

#%%
def main(county_codes):
    retrieve(county_codes)

if __name__=='__main__':
    county_codes=[1, 7, 11, 15]
    main(county_codes)