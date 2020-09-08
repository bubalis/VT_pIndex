# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:18:21 2020

@author: benja
"""

import geopandas as gpd
import os
import pandas as pd
import numpy as np
from load_in_funcs import load_counties, load_aoi


crop_fields=gpd.read_file(r"C:\Users\benja\VT_P_index\model\P_Index_LandCoverCrops\P_Index_LandCoverCrops\Crop_DomSoil.shp")
crs=crop_fields.crs

zip_code_path=os.path.join('Source_data', 'VT_ZIP_Code_Areas-shp')
soil_tests_path=os.path.join('Source_data', 'soil_test_results.csv')

zips=gpd.read_file(zip_code_path)
zips.to_crs(crs, inplace=True)


counties=gpd.read_file(r"C:\Users\benja\VT_P_index\model\Source_data\VT_Data_-_County_Boundaries-shp\VT_Data_-_County_Boundaries.shp")
counties.to_crs(zips.crs, inplace=True)
zc=gpd.overlay(zips, counties)

zc['Area']=zc.geometry.area
zc.sort_values(by='Area', inplace=True, ascending=False)
zc=zc.dissolve(by='NAME')






#bool_array=((df['FieldUOM']=='ACRES') * (df['FieldArea']>=5) | df['Commercial'])
#df=df[bool_array]


field_codes='''125
133
126
134
128
135
129
136
91
109
94
99
253
88
100
219
101
90
138
83
195
84
230
87
231
86
232
85
137
234'''.split('\n')

df=pd.read_csv(soil_tests_path)
df['County']=df['County'].apply(lambda x: str(x).upper())

field_codes=[int(c) for c in field_codes]
df=df[df['Crop ID'].isin(field_codes)]


df['Zip']=df['Zip'].astype(str)
df['Zip']='0'+df['Zip']
df['Zip']=df['Zip'].apply(lambda x: x.split('-')[0])


gdf=gpd.GeoDataFrame(pd.merge(df, zc, right_on='ZCTA', left_on='Zip', 
                              how='left', validate='many_to_one'))
gdf.drop(columns=['CNTYGEOID', 'ShapeSTAre', 'ShapeSTLen', 
         'Area', 'LSAD', 'LSAD_TRANS', 'SHAPESTAre', 'SHAPESTLen', 'OBJECTID_2',
       'CNTY', 'OBJECTID_1',], inplace=True)

gdf['County']=gdf['CNTYNAME']
gdf=gdf[gdf['geometry'].isna()==False]

gb=gdf[['Zip','P','Al', 'geometry']]

gb1=gb.dissolve('Zip', aggfunc='mean')

gb2=gdf[['County','P','Al', 'geometry']]
gb4=gb2.dissolve('County', aggfunc='mean')

gb3=gb2.dissolve('County', aggfunc='count')


gb5=gb2.dissolve('County', aggfunc='median')


ovly=gpd.overlay(crop_fields, zips)

gb6=gb[gb['P']>100].dissolve('Zip', aggfunc='count')

from scipy import stats
