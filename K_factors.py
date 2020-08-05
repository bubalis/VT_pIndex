# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 07:39:51 2020
retrieve_Kvals.py
@author: benja
"""

import json
import os
import re
import requests
import geopandas as gpd
import numpy as np





#%%





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
        print(MUNAME)
        parts=MUNAME.split()[0].split('-') #split for multi-type soil names
        parts=[get_soil_tex(part) for part in parts] #get soil textures
        parts=[part for part in parts if part]
        if ' '.join(parts)==MUNAME:
            print('Stuck')
            return 0, False
        k_fac= np.mean([get_kFactor_clay(part)[0] for part in parts if part]) #return their mean k-factor if there are multiple
        is_clay=any(['clay' in part.lower() for part in parts if part]) 
        return k_fac, is_clay

def load_kfactor_data():
    with open(r'C:\Users\benja\VT_P_index\model\Source_data\RUSLE_factors.txt') as f:
        dic=json.loads(f.read())
        k_factors=dic['k_factors']
        k_factors={k.lower(): v for k, v in k_factors.items()}
    
    
    return [key.lower() for key in k_factors.keys()]


def setK_facs_and_clay(file_path):
    soil_text_reg=re.compile('(?<=<B>TYPICAL PEDON:</B> ).+?(?=,)')
    gdf=gpd.read_file(file_path)
    if 'is_clay' in gdf.columns and 'K_factor' in gdf.columns:
        return
    soil_texts=load_kfactor_data()
    
    li=list(gdf['MUNAME'].unique())
    li.sort(key=len, reverse=True)
    clay_dict, K_factor_dict={}, {}
    for MUNAME in li:
        
        value=get_kFactor_clay(MUNAME)
        clay_dict[MUNAME]=value[1]
        K_factor_dict[MUNAME]=value[0]
    
    gdf['is_clay']=gdf['MUNAME'].apply(lambda x: clay_dict.get(x))
    gdf['K_factor']=gdf['MUNAME'].apply(lambda x: K_factor_dict.get(x))
    
    gdf.to_file(os.path.join('Source_data', 'GeologicSoils_SO'))

#%%
if __name__=='__main__':
    file_path=os.path.join('Source_data', 'GeologicSoils_SO')
    setK_facs_and_clay(file_path)
    

#%%
