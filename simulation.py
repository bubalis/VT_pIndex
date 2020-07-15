# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:22:04 2020
Script for simulating p index from shapefiles.
@author: benja
"""
import os
import sim_variables
import geopandas as gpd
import numpy as np
import pandas as pd
from field import CropField, manureApplication, fertApplication
from C_factors import Rotation, crop_seqs
from field_geodata import save_shape_w_cols, load_shape_w_cols

#globablly defined dictionaries.
manure_method_conversion_dic={**{
    'inject':'Inject / subsurf. band',
    'not_incorporated': 'Not incorporated',
    None: "Not incorporated"},
    **{s:s.title() for s in ['moldboard', 'disk', 'chisel']}}

fert_method_conversion_dic={**{'not_incorporated': 'Surface Applied',
                            'inject': 'Inj. or subsurf. banded',
                            None: 'Surface Applied'},
                            **{s:f'Incorp. / {s}' for s in ['moldboard', 'disk', 'chisel']}}

veg_type_dict={'Hay': 'Alfalfa & other hay crops',
               'Fallow': 'Pasture',
               'Small_Grain': 'Small grains', 
               'Other Crop': 'Small grains'}

m_date_conversion_dic={'summer': 'May - Sept',
                            'fall': 'Oct - Dec 15',
                            'winter': 'Dec 15 - March'}

f_date_conversion_dic={'summer' :'May-September',
 'fall': 'October-December 15',
 'winter': 'December 15-March' }

  
 

 
class CropFieldFromShp(CropField):
    '''Initate Crop Field from a single series of a shapefile.'''
    def __init__(self, series, params_to_sim):
        if series['crop_type']=='Other Crop':
            series['crop_type']='Small_Grain'
        
        self.params_to_sim=params_to_sim
        self.known_params=series.to_dict()
        
        self.known_params['acres']=series['geometry'].area*0.000247105 #sq meters to acres
        self.idnum=self.known_params['IDNUM']
        self.set_rotation()
        self.combined_results=[]
        self.known_params['soil_is_clay']=bool( self.known_params['soil_is_clay'])
        
    def calcPindex(self):
        CropField.calcPindex(self)
        self.combined_results.append(self.results)
        
    def initialize_sim(self, variable_objs):
        '''Set up all needed inputs to run the P Index on Field'''
        self.simulate_params(variable_objs)
        self.setup_data()
        self.USLE()
        
        
    def set_rotation(self):
        '''Assign the crop rotation to the field.'''
        
        crop_params={param.split('_')[1]: self.known_params[param]  for param in 
                                  [p for p in self.known_params if 'Years_' in p]}
      
        self.rotation=Rotation(**crop_params)
    
    
    def sim_rotation(self):
        '''Set variables related to crop sequence.'''
        crop_seq= self.rotation.draw_crops(self.known_params['crop_type'])
        is_establishment_year=(crop_seq[0]!=crop_seq[1])
        return crop_seq, is_establishment_year
    
    def simulate_params(self, variable_objs):
        '''Simulate Parameters for the field. 
        pass a dictionary of variable objects. '''
        crop_seq, is_establishment_year=self.sim_rotation()
        
        self.sim_params={'crop_seq': crop_seq,
                         'is_establishment_year': is_establishment_year}
        

        #parameters simulated directly from variable objects
        variables=self.params_to_sim
        for name in variables:
            self.sim_params[name]=variable_objs[name].draw(
                                            **{**self.known_params, **self.sim_params})
            
            
        
        self.sim_params['veg_type']=self.get_veg_type()
        
        #simulate manure/fertilizer applications
        self.gen_Manure(variable_objs)
        self.gen_Fert(variable_objs)
        
    
          
    def summarize_results(self):
        sum_results={component:np.mean([r[component] for r in self.combined_results])
                                              for component in self.results.keys() }
        sum_results['total_p_lost_lbs']=self.params['acres']*sum_results['total p index']/80
        return sum_results
        
        
    def get_veg_type(self):
        if self.known_params['crop_type']=='Corn':
            if self.sim_params['cover_crop']:
                return 'Row crop + successful cover crop'
            else:
                return 'Corn & other row crops'
        else:
            return veg_type_dict[self.known_params['crop_type']]


    def gen_Manure(self, variable_objs):
        '''Generate Manure Applications for this field with simulated data. '''
        
        self.sim_params['manure_applications']=[]
        for i in range(int(self.sim_params['num_manure_applications'])):
            manure_params={'manure_Type': 'Cow',
                           'manure_rate': self.sim_params['total_p_applied']/self.sim_params['num_manure_applications']}
            
            
            for m_var in ['incorp_method', 'time_to_incorp', 'date']: #simulated manure parameters
                params={**manure_params, **self.known_params, **self.sim_params}
                manure_params[f'manure_{m_var}']=variable_objs[f'manure_{m_var}'].draw(**params)
            
            manure_params={key[7:]: value for key, value in manure_params.items()}
            
            #initate a manure application object
            self.sim_params['manure_applications'].append(SimManure(field=self, **manure_params))
   
        
   
    def gen_Fert(self, variable_objs):
         '''Generate Fertilizer Applications for this field, using simulated data. '''
         self.sim_params['fertilizer_applications']=[]
         
         for i in range(int(self.sim_params['num_fert_applications'])):
            fertilizer_params={'fert_rate':5}
            
            for p_var in ['fert_incorp_method',
                'fert_date']:
                params={**self.known_params,**self.sim_params}
                
                fertilizer_params[f'{p_var}']=variable_objs[f'{p_var}'].draw(**params)
            fertilizer_params={key[5:]:value for key, value in fertilizer_params.items()}                      
            self.sim_params['fertilizer_applications'].append(SimFert(field=self, **fertilizer_params))
         self.params={**self.sim_params, **self.known_params}          
     
    def USLE(self):
        '''Set the erosion rate for the field.'''
        
         #to do: set param for USLE
        
        self.params['C']=self.getC_fac()
        self.params['P']=.8 #P will actually have to be simulated!
        self.params['erosion_rate']=np.product([self.params[n] for n in ['RKLS', "C", 'P']])
     
        
    def getC_fac(self):
        '''Retreive the C factor from USLE.'''
        for seq in crop_seqs:
            dic= seq.respond(self)
            if dic:
                break
        c= dic[self.params['tillage_timing']][self.params['tillage_method']]
        return c
        
        
class SimFert(fertApplication):
    '''Fertilizer application with simulated parameters. '''
    
    
    def __init__(self, field, rate, incorp_method, date):
        incorp_method=fert_method_conversion_dic[incorp_method]
        date=self.get_timing(date, field)
        fertApplication.__init__(self, field, rate, incorp_method, date)

    def get_timing(self, date, field):
        '''Convert p timing into the form the model wants.
        NOTE: TODO--> Deal with the possibility of previous-year cover crops. 
        Need to simulate PREVIOUS YEAR cropping to do this.
        
        '''
        if date=='spring':
            if field.sim_params['veg_type'] in [ 'Alfalfa & other hay crops', 
                                                 'CRP, other ungrazed, perm. veg.',
                                                 'Small grains']:
                return 'April vegetated'
            else:
                return 'April bare'
        else:
            return f_date_conversion_dic[date]      


class SimManure(manureApplication):
    '''Manure application with simulated parameters.'''
    
    def __init__(self, field, rate, date, time_to_incorp, incorp_method, Type):
        incorp_method=manure_method_conversion_dic[incorp_method]
        date=self.get_timing(date, field)
        manureApplication.__init__(self, field, rate, date, time_to_incorp, incorp_method, Type)
    
    
    
    def get_timing(self, date, field):
        '''Convert p timing into the form the model wants.
        NOTE: TODO--> Deal with the possibility of previous-year cover crops. 
        Need to simulate PREVIOUS YEAR cropping to do this.
        
        '''
        if date=='spring':
            if field.sim_params['cover_crop'] or field.sim_params['veg_type'] in [ 'Alfalfa & other hay crops', 
                                                 'CRP, other ungrazed, perm. veg.',
                                                 'Small grains']:
                return 'April - vegetated'
                
            else:
                return 'April - bare'
        else:
            return m_date_conversion_dic[date]      

def fix_hydro_group(string):
    '''Arbitrarily assigns unrated soils to hydo_group_B'''
    
    if string=='not rated':
        return 'B'
    else:
        return string


class simulation():
    def __init__(self, directory, params_to_sim, variables_path):
        self.directory=directory
        self.fields=[]
        self.params_to_sim=params_to_sim
        self.variable_objs=sim_variables.load_vars_csv(variables_path)
        
    def load_data(self):
        '''Load in data from the shapefile in the simulation directory.'''
        gdf=load_shape_w_cols(self.directory)
        gdf=self.fix_data(gdf)  
        
        for i, row in gdf.iterrows():
            self.fields.append(CropFieldFromShp(row, self.params_to_sim))
        
        return self.fields, gdf

    def fix_data(self, gdf):
        '''Make minor fixes to how data is represented. '''
        gdf['hydro_group']=gdf['hydro_group'].apply(lambda x: x.split('/')[-1])
        gdf['hydro_group']=gdf['hydro_group'].apply(fix_hydro_group)
        gdf['buffer_width']=gdf['distance_to_water']
        return gdf


    def simPindex(self, n_times=1):
        '''Simulate P Index for all of the fields.'''
        
        self.records=[]
        for n in range(n_times):
            print(f'Running Simulation #{n} out of {n_times}')
            for field in self.fields:
                field.initialize_sim(self.variable_objs)
                field.calcPindex()
                self.records.append({**{'Sim Number': n}, **field.params, **field.results})
                
    def load_test(self):
        '''Load a small subset for a test.'''
        
        gdf=load_shape_w_cols(self.directory)
        gdf=self.fix_data(gdf)  
        
        for i, row in gdf.head(100).iterrows():
            self.fields.append(CropFieldFromShp(row, self.params_to_sim))
        
        return self.fields, gdf
    
    def summarize_results(self):
        sum_results=[{**field.known_params,**field.summarize_results()} 
                     for field in self.fields]
        return pd.DataFrame(sum_results)



#%%
if __name__=='__main__':
    params_to_sim=['soil_test_phos',
                     'Al_level',
                     'tile_drain',
                     'num_manure_applications',
                     'total_p_applied',
                     'num_fert_applications',
                     'manure_setback',
                     'cover_crop',
                     'cover_perc',
                     'tillage_method',
                     'tillage_timing',
                     'sed_cntrl_structure_fact']
    
    variables_path=r"C:\Users\benja\VT_P_index\model\sim_variables.txt"
    shapes_path=os.path.join(os.getcwd(), 'intermediate_data', 'SO01_fields')
    
    sim=simulation(shapes_path, params_to_sim, variables_path)
    
    fields, gdf=sim.load_test()
    sim.simPindex(20)
    gdf=gpd.GeoDataFrame(sim.summarize_results())
    dir_path=os.path.join(os.getcwd(), 'results', 'scratch')
    save_shape_w_cols(gdf, dir_path)


