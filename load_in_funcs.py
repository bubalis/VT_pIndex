# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:00:51 2020

@author: benja
"""

import geopandas as gpd
import os

def load_aoi(county_codes, watersheds):
    '''Return an area of interes that is the boundary of all sub-watersheds in the 
    Lake Champlain Watershed that intersect with the counties of interest.'''
    counties=load_counties(county_codes)
    
    watersheds=watersheds[watersheds['HUC12'].str.contains('0415040')] #All watersheds that flow into lake champlain
    overlay=gpd.overlay(watersheds, counties, how='intersection')
    watersheds=watersheds[watersheds['HUC12'].isin(overlay['HUC12'].unique())]
    
    return gpd.overlay(counties, watersheds, how='union')

def load_counties(county_codes):
    '''Return a gdf of county boundaries for each county code.'''
    county_path=os.path.join(os.getcwd(), 'source_data', 'VT_Data_-_County_Boundaries-shp')
    counties=gpd.read_file(county_path)
    return counties[counties['CNTY'].isin(county_codes)]


