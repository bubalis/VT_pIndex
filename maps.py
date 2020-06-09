# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:21:05 2020

@author: benja
"""

import json
import os
import natcap.invest
import shapefile
import re
import requests
import geopandas as gpd
import np

# Values for RUSLE k, tillage and crop factors based on data from: http://www.omafra.gov.on.ca/english/engineer/facts/12-051.htm#t2
with open(r'C:\Users\benja\VT_P_index\model\Source_data\RUSLE_factors.txt') as f:
    dic=json.loads(f.read())


k_factors=dic['k_factors']
k_factors={k.lower(): v for k, v in k_factors.items()}


gdf=gpd.read_file(os.path.join('Source_data', 'soils', 'GeologicSoils_SO01', 'Geologic_SO01_poly.shp'))

soil_texts=[key.lower() for key in k_factors.keys()]
soil_texts.sort(key=len, reverse=True)


#%%
soil_text_reg=re.compile('(?<=<B>TYPICAL PEDON:</B> ).+?(?=,)')

def get_soil_tex(name):
    '''Get the soil name with texture from the USDA for a given string.
    e.g.: get_soil_tex('Marlow')
    returns 'Marlow Fine Sandy loam' '''
    name=name.upper()
    response=requests.get(f'https://soilseries.sc.egov.usda.gov/OSD_Docs/{name[0]}/{name}.html')
    try:
        return soil_text_reg.search(response.text).group()
    except:
        return None


def get_kFactor(MUNAME):
    
    for tex in soil_texts:
        if tex in MUNAME.lower():
            return k_factors[tex]['Avg']
    else:
        parts=MUNAME.split()[0].split('-')
        parts=[get_soil_tex(part) for part in parts]
        return np.mean([get_kFactor(part) for part in parts if part])

K_factor_dict={MUNAME: get_kFactor(MUNAME) for MUNAME in gdf['MUNAME'].unique()}
#%%
gdf['K_factor']=gdf.apply(get_kFactor, axis=1)


def find_unknown_textures(gdf):
    results={{[]}}
    for i, row in gdf.iterrows():
        string=row['MUNAME'].split()[0]
        if '-' in string:
            key=re.split('\s|-', row['MUNAME'])[0]
        value=row['K_factor']
        if not value:
           for name in string.split('-'):
               if name not in results:
                   results[name]=get_soil_tex(name)
            


missing=list(set(missing))

missing_dict





    