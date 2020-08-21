# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:23:33 2020

@author: benja
"""
import datetime
import csv
import string
import math



dic={'Bennington': {'growing season': [1.7, 1.54, 1.66], 'snowmelt': [.75, 1, 1.81] },  
 'Rutland': {'growing season': [1.7, 1.54, 1.66], 'snowmelt': [.75, 1, 1.81] },
 'Windham': {'growing season': [1.53, 1.62, 1.78], 'snowmelt': [.94, 1.25, 1.94] },
 'Windsor': {'growing season': [1.53, 1.62, 1.78], 'snowmelt': [.94, 1.25, 1.94] },
 'Addison': {'growing season': [1.54, 1.49, 1.75], 'snowmelt': [.81, 1.13, 1.81] },
 'Chittenden': {'growing season': [1.54, 1.49, 1.75], 'snowmelt': [.81, 1.13, 1.81] },
 'Franklin': {'growing season': [1.54, 1.49, 1.75], 'snowmelt': [.81, 1.13, 1.81] },
 'Grand Isle': {'growing season': [1.54, 1.49, 1.75], 'snowmelt': [.81, 1.13, 1.81] },
 'Washington': {'growing season': [1.52, 1.55, 1.58], 'snowmelt': [.88, 1.19, 1.75] },
 'Lamoille': {'growing season': [1.52, 1.55, 1.58], 'snowmelt': [.88, 1.19, 1.75] },
 'Orange': {'growing season': [1.47, 1.45, 1.84], 'snowmelt': [.94, 1.31, 2.06] },
 'Orleans': {'growing season': [1.47, 1.45, 1.84], 'snowmelt': [.94, 1.31, 2.06] },
 'Essex': {'growing season': [1.47, 1.45, 1.84], 'snowmelt': [.94, 1.31, 2.06] },
 'Caledonia': {'growing season': [1.47, 1.45, 1.84], 'snowmelt': [.94, 1.31, 2.06] },
 }








dic2={}
for county, values in dic.items():
    dic2[county]=[(values['growing season'][i]+values['snowmelt'][i])*.22651 for i in range(0,3)]
    
def lookupBRV(county, elevation):
    '''Lookup the Base Runoff Volume for a field, based on county and elevation.
    Based on table 5 on page 8 of VTPI docs.
    '''
    values=dic2[county]
    if 0<elevation<600:
        return values[0]
    elif 600<=elevation<=1000:
        return values[1]
    elif 1000<elevation<6000:
        return values[2]
    elif elevation<=0 or math.isnan(elevation):
        print(f'Eleveation of {elevation} not valid assigning to 0')
        return values[0]
    
    else:
        print(f'{elevation}  not valid')
        raise ValueError
        
runoff_adj_dict={'Corn & other row crops': [0.42, 0.25, 1.00, 0.75, 1.96, 1.48, 2.65, 1.98],
'Row crop + successful cover crop': [0.22, 0.15, 0.67, 0.55, 1.48, 1.23, 2.17, 1.82], 
'Vegetable crop - clean cultivated': [0.42, 0.25, 1.00, 0.75, 1.96, 1.48, 2.65, 1.98], 
'Vegetable crop - mulched, living row cover': [0.22, 0.15, 0.67, 0.55, 1.48, 1.23, 2.17, 1.82],
'Vegetable crop - vining or high-canopy': [0.22, 0.15, 0.67, 0.55, 1.48, 1.23, 2.17, 1.82] ,
'Small grains': [0.22, 0.15, 0.67, 0.55, 1.48, 1.23, 2.17, 1.82],
 'Alfalfa & other hay crops': [0.12, 0.12, 0.55, 0.55, 1.37, 1.37, 1.98, 1.98], 
'Pasture': [.02, 0.02, .30, 0.30, 1.08, 1.08, 1.82, 1.82,] ,
'CRP, other ungrazed, perm. veg.': [.01, 0.01, 0.12, 0.12, 0.5, 0.50, 1.00, 1.00], 
'Woodland': [.01, 0.01, 0.15, 0.15, 0.62, 0.62, 1.10, 1.10],
'Hay': [0.12, 0.12, .55, 0.55, 1.37, 1.37, 1.98, 1.98]}

def runoff_parser(crop_name, cover_crop):
    if crop_name in runoff_adj_dict:
        return crop_name
    elif crop_name in ['Corn', 'Soy']:
        if cover_crop==True:
            return 'Row crop + successful cover crop'
        else:
            return 'Corn & other row crops'
    elif crop_name=='Small grain':
        return 'Small Grains'
    elif crop_name=='Hay':
        return 'Alfalfa & other hay crops'
    elif crop_name=='Fallow':
        return 'CRP, other ungrazed, perm. veg.'
    else:
        raise ValueError
    
    






hydro_groups=['A',
              'B',
              'C',
              'D']

new_RAF_dict={
        key:{hydro_groups[i]:value[i*2:(i*2+2)] for i in range(0,4)}
for key, value in runoff_adj_dict.items()}

tile_drain_adj_dict={'A':'A', #page 11
                 'B': 'A',
                 'C': 'B',
                 'D': 'C'}



def lookup_RAF(hydro_group, veg_type, cover_perc, tile_drain):
    '''Get the runoff adjustment factor based on Soil Hydrologic Group, Cover% and Vegetation Type.
    Based on page 9, table 6. '''
    if tile_drain:
        hydro_group=tile_drain_adj_dict[hydro_group]
    subset=new_RAF_dict[veg_type][hydro_group]
    if 0<=cover_perc<=.20:
        out= subset[0]
    elif .20<cover_perc<=1:
        out= subset[1]
    else:
        out=0
    if out==0:
        print(cover_perc)
        print('RAF assigned to Zero.')
    return out

soil_hydro_factors={'A': 0.5,
'B': 1.0,
'C': 1.6,
'D': 2.0}


def get_soil_hyd_factor(string, tile_drain):
    '''retrieve hydrologic factor. 
    Pass string (letter or two letters separated by slash)
    and a boolean for tile_drain'''
    if '/' in string: #adjustments for dual hydrologic groups
        if tile_drain: 
            string=string.split('/')[0] #tile drain gives higher hydro rating
        else:
            string=string.split('/')[1]
    else: #Hydrologic groups can't be upgraded twice
        if tile_drain: 
            string=tile_drain_adj_dict[string] #Tile drainage upgrades the hydrologic group
    return soil_hydro_factors[string]



fertilizer_app_dict={
('Surface Applied', 'May-September'): 0.5,
('Surface Applied', 'October-December 15'): 1.0,
('Surface Applied', 'December 15-March'): 1.3,
('Surface Applied', 'April bare'): 0.8,
('Surface Applied', 'April vegetated'): 0.6,
('Surface Applied', 'April -  bare'): 0.8,
('Surface Applied', 'April - vegetated'): 0.6,
'Incorp. / moldboard': 0.05,
'Incorp. / chisel': 0.25,
'Incorp. / disk': 0.40,
'Inj. or subsurf. banded': 0
}

def getFertFactor(method, date):
    if method=='Surface Applied':
        return fertilizer_app_dict[(method, date)]
    elif method=='not incorporated':
        return 1
    elif method:
        return fertilizer_app_dict[method]
    else:
        return 0
    
manure_method_dict={
        'Inject': 0,
        'subsurf. band': 0,
        'Inject / subsurf. band':0,
'Moldboard': 0.05,
'Chisel': 0.25,
'Disk': 0.4,
'Not incorporated': 1.0,
'None applied': 0,
'moldboard':.05,
'chisel':.25,
'not_incorporated':1,
'inject':0


}


'''
def manure_timing(date, vegetation):
    day_of_year=date.timetuple().tm_yday
    if 350>=day_of_year or day_of_year<92:
        return 1.3
    elif 92>=day_of_year<122:
        if vegetation:
            return .6
        else: 
            return .8
    elif 122>=day_of_year<275:
        return .5
    elif 275>=day_of_year<350:
        return 1
'''

manure_timing_dic={'May - Sept': 0.5,
'Oct - Dec 15': 1.0,
'Dec 15 - March': 1.3,
'April - bare': 0.8,
'April - vegetated': 0.6,
'None applied': 0, 
'summer': .5,
'fall': 1.0,
'winter': 1.3,
'spring': .8}

def manure_timing(manure_date):
    '''Get manure timing factor.'''
    return manure_timing_dic[manure_date]


def incorp_timing(days, method):
    '''Return the risk factor based on how long between manure application and incorporation.'''
    if method=='Not incorporated':
        return 1
    if method=='Inject / subsurf. band':
        return 0
    if not days:
        return 0
    if days==0:
        return 0
    elif 0<days<1:
        return .07
    elif 1<=days<2:
        return .13
    elif 2<=days<4:
        return .24
    elif 4<=days<7:
        return .38
    elif 7<=days<21:
        return .74
    else:
        return 1
    
uptake_dict={'Corn & other row crops': 100,
 'Row crop + successful cover crop': 50,
 'Small grains': 50,
 'Alfalfa & other hay crops': 50,
 'Pasture': 0,
 'Vegetable crop - clean cultivated': 50,
 'Vegetable crop - mulch or living row cover': 50,
 'Vegetable crop - vining or high canopy': 50}



def manure_factor(manure_date, method, time_to_incorp):
    '''Calculate the manure factor for surface runoff.
    Described pages 3-4 in the docs.
    date: datetime obj of application, 
    method: string
    time_to_incorp: numeric,
    vegetation: boolean.'''
    
    method_factor= manure_method_dict[method]
    risk=incorp_timing(time_to_incorp, method)
    seasonal_factor=manure_timing(manure_date)
    return method_factor+(seasonal_factor-method_factor)*risk

#%%
def str_to_int(string):
    return float(string.replace(',', ''))

def load_LS():
    with open('LSdata.csv') as file:
        reader=csv.reader(file)
        values=[line for line in reader]
        lengths=[str_to_int(x) for x in values[0][1:]]
        dic={}
        for line in values[1:]:
            dic[str_to_int(line[0])]={length:str_to_int(factor) for length, factor in zip(lengths, line[1:])}
    return dic


def lookupRange(value, dic):
    '''Get value for the key in a dictionary based on range.
    value: a number.
    dic: a dictionary with numeric keys. 
    Returns the  values for the 1st key that is greater than value/'''
    dic_keys=list(dic.keys())
    if value in dic_keys: #If the value is in the dictionary keys
        return dic[value] #Just return that key
    else:
        dic_keys.append(value)
        dic_keys.sort()
        i=dic_keys.index(value)
        if i==len(dic_keys)-1: #If the value is the highest value
            return dic[dic_keys[-2]] # return the value for the highest key in dict
        else:
            return dic[dic_keys[i+1]] #return the next value from dictionary
