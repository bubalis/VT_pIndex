# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:28:21 2020

@author: benja
"""
from field import CropFieldFromDic, manureApplication
from SALib.sample import saltelli
import pandas as pd


class CategoricalSampler(object):
    '''Class to turn Numerical Values drawn for sensitivity analysis into categorical variables.'''
    def __init__(self, *param_values):
        self.param_values=param_values
    
    def draw(self, i):
        '''Convert value i into a category for this variable.
        Return that category'''
        i=round(i)
        return self.param_values[i]
    
    def reverse_lookup(self, val):
        return self.param_values.index(val)
    
H_group_sampler=CategoricalSampler('A', 'B', 'C', 'D')
Bool_Sampler=CategoricalSampler('False', 'True', )
vegSampler=CategoricalSampler('Corn & other row crops', 'Row crop + successful cover crop', 
                              'Alfalfa & other hay crops', 'Pasture')

class CropFieldSA(CropFieldFromDic):
    
    def __init__(self, **params):
        self.params={k:v for k,v in params.items() if k[1]!='_'}
        self.params['hydro_group']=H_group_sampler.draw(params['hydro_group'])
        self.params['soil_is_clay']=Bool_Sampler.draw(params['soil_is_clay'])
        self.params['veg_type']=vegSampler.draw(params['veg_type'])
        self.params['tile_drain']=Bool_Sampler.draw(params['tile_drain'])
        #self.params['manure_applications']=[manureApplication(**{k:v for k, v in params.items() 
         #                                                        if k[0:2]=='m_'})]
        self.params['manure_applications']=[]
        self.params['fertilizer_applications']=[]
        
        self.params['county']='Addison'
        self.params['elevation']=300
        self.params['sed_cntrl_structure_fact']=None
        CropFieldFromDic.__init__(self, self.params)
        
def run_simulation(parameters):
    field=CropFieldSA(**parameters)
    results =field.calcPindex()
    results= pd.Series(results)
    
        

    return results

28329
problem = {
      'num_vars': 11,
      'names': [
 'soil_test_phos',
 'Al_level',
 'erosion_rate',
 'hydro_group',
 'soil_is_clay',
 'cover_perc',
 'veg_type',
 'distance_to_water',
 'buffer_width',
 'manure_setback',
 'tile_drain',
               ],
                
      'bounds': [[0.1, 150],
                 [0.1, 20],
                 [0.1, 10],
                 [0, 3],
                 [0, 1],
                 [0, 1],
                 [0, 3],
                 [10, 200],
                 [10, 100],
                 [10, 50], 
                 [0,1], 
                 ]
    }
    
n = 2000
param_values = saltelli.sample(problem, n, calc_second_order=True)
    
    
experiments = pd.DataFrame(param_values,
                               columns=problem['names'])
    
dic=experiments.to_dict('records')
results = [CropFieldSA(**parameters).calcPindex() for parameters in dic]