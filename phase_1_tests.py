# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:59:36 2020

@author: benja



Testing based on the spreadsheet model. 


"""


import xlrd
from field import manureApplication, CropFieldFromDic, fertApplication
import re
import numpy as np
import os


#%%
def parse_manure(n, dic):
    
    name=f'Manure {n}'
    values={k:v for k,v in dic.items() if name in k}
    parameters={'rate': values[f'{name} rate, lb P2O5/ac'],
                'date': values[f'{name} application time of year'],
                'incorp_method': values[f'{name} application method'],
                'time_to_incorp': parse_manure_time(values[f'{name} time to incorporation']),
                'Type':values[f'{name} type'],}
    return parameters

def manure_date(string):
    if string.lower()=='none applied':
        return None
    else:
        string=string.split('-')[0]
        
def parse_manure_time(string):
    if string.lower()=='none applied':
        return 1
    elif string.lower()=='immediate':
        return 0
    elif string.lower()=='not incorporated':
        return 99
    elif '<' in string:
        return int(re.search('\d+', string).group())-.5
    else:
        return int(re.search('\d+', string).group())+.5
    

def parse_fert_date_method(string):
    
    if string.lower()=='none applied':
        return 'not incorporated', None
    elif string.split()[0] in ['Inj.', 'Incorp.']:
        return string.split(r'(')[0].strip(), None
    elif string.split()[0]=='Surf.':
        if 'April' in string:
            veg=re.search(r'(?<=\()\w+', string).group()
            return 'Surface Applied', f'April {veg}' 
            
        else:
            return 'Surface Applied', string.split()[2]
    


def parse_fertilizer(dic):
    incorp_method, date=parse_fert_date_method(dic['Fertilizer method/timing'])
    parameters={'rate': dic['Fertilizer rate, lb P2O5/ac'],
                'field': None,
                'date': date,
                'incorp_method': incorp_method,
                
                }
    return parameters

def fert_from_dict(dic):
    params=parse_fertilizer(dic)
    return fertApplication(**params)
    
#%%

def y_n_to_bool(string):
    '''Convert a string of yes or no to a boolean'''
    if string.lower()=='yes':
        return True
    elif string.lower()=='no':
        return False
    else:
        raise ValueError

def coverParser(string):
    '''Return a float value for cover percent based on a string from the sheet'''
    string=str(string).strip()
    if string=='2.0':
        return .05
    elif string=='3.0':
        return .40
    else:
        print(string)
        raise ValueError

def elevParser(string):    
    '''Return a numeric value from the sheet's string value for elevation'''
    if string[0]=='<':
        return 300
    elif string[0]=='6':
        return 800
    elif string[0]=='>':
        return 1200
    else:
        raise ValueError
        
def countyParser(string):
    counties=string.split(r"(" )[1]
    return counties.split(',')[0]

def get_hydro_grp(string):
    if string[0:5]=='Other':
        return re.search(r'(?<=HydrGrp )\w', string).group()
    return string.split(r'(')[1][0]
    
def is_clay(string):
    return 'Clay'==string

def parse_field_params(dic):
    params={
    'county': countyParser(dic['Location (Vermont county)']), 
    'elevation': elevParser(dic['Elevation zone, feet']), 
    'soil_test_phos': dic["Soil test P, ppm (Mod. Morgan's)"],
    'Al_level': dic['Reactive soil aluminum, ppm'],
    'erosion_rate': dic['Erosion rate (RUSLE or WEPP, tons/ac/yr)'],
     'hydro_group' :get_hydro_grp(dic['Soil type or series (& HydrGrp)']),
     'soil_is_clay': is_clay(dic['Texture group']),
     'cover_perc': coverParser(dic['Surface cover %']), 
     'veg_type':dic['Crop / Vegetation type'], 
     'distance_to_water' : dic['TOTAL distance FROM field edge TO any water conveyance, ft' ],
     'buffer_width': dic['Vegetated buffer width BETWEEN field edge & conveyance, ft' ],
     'manure_setback':dic['Manure spreading setback dist WITHIN field, ft'],
    'tile_drain': y_n_to_bool(dic['Presence of Pattern Tile Drainage']),
    #'sed_control_struc':dic[ 'Sediment trap structure or other erosion control'],
    'sed_cntrl_structure_fact': None,
   'manure_applications': [manure_from_dict(n, dic) for n in [1,2,3,]],
    'fertilizer_applications':[fert_from_dict(dic) ],
    #'tile_drain':False,
    #'cover_perc': .05
    
    }
    return params

#%%
def manure_from_dict(n, dic):
    d=parse_manure(n, dic)
    d['field']=0
    return manureApplication(**d)    

def load_example_data(data_col=5):
    wb = xlrd.open_workbook('Vermont-Phosporus-Index (1).xlsx') 
    sheet = wb.sheet_by_index(2)

    values={}
    for i,row in enumerate(sheet.get_rows()):
        values[row[4].value]=row[data_col].value
        if i>192:
            return values


def check_correct(test_field, values):
    results=[test_field.results[name] for name in [ 'total p loss',
     'surface particulate loss' ,
     'surface dissolved loss',
     'subsurface_loss']]
    
    test_values=[values[name] for name in [ 'P Index:',
                    'Pathway I:  Sediment-bound P',
                    'Pathway II:  Dissolved P in surface runoff',
                    'Pathway III: Subsurface loss of diss. & sed.-bound P',]]
    assert np.isclose(results, test_values, rtol=.03).all(), (test_values, results)
    
    
        
    
    
    
if __name__=='__main__':
    for i in range(5, 30):
        values=load_example_data(i)
        dic=parse_field_params(values)
        
        test_field=CropFieldFromDic(dic)
        test_field.calcPindex()
        with open(os.path.join('test_results', f'test{i}.txt'), 'w') as f:
            for key, value in values.items():
                print (f'{key}:   {value}', file=f)
        check_correct(test_field, values)
        
                
                

            
    #%%
    
