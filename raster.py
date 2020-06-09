# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:43:36 2020

@author: benja


Script to download and merge DEM rasters for an aoi.
"""

import gdal
import rasterio 
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import re
import requests
import os
import zipfile
import subprocess
import rasterio.mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from bs4 import BeautifulSoup


#%%
dst_crs=CRS.from_epsg(32145)



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

        
def check_file_download(end_url, aoi_buf):
    '''Check if file falls within aoi_buf'''
    if end_url[-4:]!='img':
        return False
    match_y=y_search.search(end_url)
    match_x=x_search.search(end_url)
    if match_x and match_y:
        x, y=int(match_x.group()), int(match_y.group())
        return all([aoi_buf.bounds['minx']<x*100,
                    aoi_buf.bounds['maxx']>x*100,
                    aoi_buf.bounds['miny']<y*100,
                    aoi_buf.bounds['maxy']>y*100,
                    ] )
            
def download_file(url, directory, start_url=''): 
    file_name=os.path.join(directory, url.split('/')[-1])
    r=requests.get(start_url+url)
    with open(file_name, 'wb') as f:
        f.write(r.content)
    return file_name

    
def list_links(b_url):
    '''Get urls of all links from a webpage.
    Args: a url. '''
    page_data=requests.get(b_url)
    soup=BeautifulSoup(page_data.text)
    return [link.get('href') for link in soup.find_all('a')]

def retreive(county_code):
    '''Get'''
    
    base_urls=['https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2015/DEMHE/',
               'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2017/DEMHF/',
               'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2014/DEMHE/',
               'https://maps.vcgi.vermont.gov/gisdata/vcgi/lidar/0_7M/2013/DEMHE/',
              ] 
    
    
    #make aoi_polygon
    
    
    aoi_path=os.path.join(os.getcwd(), 'intermediate_data', f'Geologic_{county_code}_poly.shp')
    aoi=gpd.read_file(aoi_path)
    aoi['null']=1
    aoi=aoi.dissolve(by='null')

    directory=os.path.join('source_data', county_code, 'DEM_rasters')
    
    for b_url in base_urls:
        links=list_links(b_url)
        
        zip_file_url=[l for l in links if l[-4:]=='.zip'][0]
        file_name=download_file(zip_file_url, directory, 'https://maps.vcgi.vermont.gov')
        
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(file_name[:-4])
                
            #load the reference map 
            ref_map_file=[f for f in os.listdir(file_name[:-4]) if f[-4:]=='.shp'][0]
            ref_map=gpd.read_file(os.path.join(file_name[:-4], ref_map_file))
            
            #crop the ref_map to the area of interest
            ref_map=ref_map.to_crs(aoi.crs)
            ref_map=gpd.overlay(ref_map, aoi)
                
            
    #download all files in aoi        
    for file_url in ref_map['DOWNLOAD_P'].to_list():
        download_file(file_url, directory)
    return directory
"""
def merge_rasters(directory, out_name='merged.img'):
    '''Merge all rasters in a directory. 
    Arguments: Directory 
    out_name: default is "merged.img" '''
    #write list of raster filenames to text file
    with open(os.path.join(directory, 'raster_list.txt'), 'w') as f:
        for filename in os.listdir(directory):
            if filename[-4:]=='.img':
                print(os.path.join(os.getcwd(), directory, filename), file=f)
    
    cmd = f'python C:\\Users\\benja\\anaconda3\\Scripts\\gdal_merge.py -o {os.path.join(os.getcwd(), directory, out_name)} -q -v --optfile {os.path.join(os.getcwd(), directory, "raster_list.txt")}'
    subprocess.call(cmd, shell=True)
  """   
    
#%%
   

#if __name__=='__main__':
 #   retrieve('SO01')