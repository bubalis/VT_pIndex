# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:21:05 2020

@author: benja

This script retrieves k values and clay booleans for all soil units
"""

import json
import os
import re
import requests
import geopandas as gpd
import numpy as np

# Values for RUSLE k, tillage and crop factors based on data from: http://www.omafra.gov.on.ca/english/engineer/facts/12-051.htm#t2
with open(r'C:\Users\benja\VT_P_index\model\Source_data\RUSLE_factors.txt') as f:
    dic=json.loads(f.read())


k_factors=dic['k_factors']
k_factors={k.lower(): v for k, v in k_factors.items()}


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


def get_kFactor_clay(MUNAME):
    '''Lookup the K_factor for a soil type.'''
    for tex in soil_texts:
        if tex in MUNAME.lower():
            return k_factors[tex]['Avg'], 'clay' in MUNAME.lower()
    else:
        parts=MUNAME.split()[0].split('-') #split for multi-type soil names
        parts=[get_soil_tex(part) for part in parts] #get soil textures
        return np.mean([get_kFactor_clay(part)[0] for part in parts if part]), any(['clay' in part.lower() for part in parts if part]) #return their mean k-factor if there are multiple



#%%
counties=['SO01']
if __name__=='__main__':
    for code in counties:
        gdf=gpd.read_file(os.path.join('Source_data', 'soils', f'GeologicSoils_{code}', f'Geologic_{code}_poly.shp'))
        soil_dict={MUNAME: get_kFactor_clay(MUNAME) for MUNAME in gdf['MUNAME'].unique()}
        clay_dict={key: value[1] for key, value in soil_dict.items()}
        K_factor_dict={key: value[0] for key, value in soil_dict.items()}
        gdf['K_factor']=gdf['MUNAME'].map(K_factor_dict)
        gdf['is_clay']=gdf['MUNAME'].map(clay_dict)
        
        gdf.to_file(os.path.join('intermediate_data',  f'Geologic_{code}_poly.shp'))





    