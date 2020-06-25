# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:22:04 2020
Script for simulating p index from shapefiles.
@author: benja
"""
import os
import sim_variables
from field_geodata import main as field_geodata
from field import CropField, manureApplication, fertApplication
import geopandas as gpd
import numpy as np



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
    '''Initate class from a series from a shapefile.'''
    def __init__(self, series, variable_objs):
        if series['crop_type']=='Other Crop':
            series['crop_type']='Small_Grain'
        self.known_params=series.to_dict()
        self.idnum=self.known_params['IDNUM']
        self.simulate_params(variable_objs)
        self.setup_data()
        self.USLE()
        
        
        
    def simulate_params(self, variable_objs):
        '''Simulate Parameters for the field. 
        pass a dictionary of variable objects. '''
        
        self.sim_params={}
        
        #parameters simulated directly from variable objects
        for name in ['soil_test_phos',
                     'Al_level',
                     'tile_drain',
                     'num_manure_applications',
                     'total_p_applied',
                     'num_fert_applications',
                     'manure_setback',
                     'cover_crop',
                     'cover_perc',
                     'tillage_method',
                     'sed_cntrl_structure_fact']:
            self.sim_params[name]=variable_objs[name].draw(
                                            **{**self.known_params, **self.sim_params})
            
            
        
        self.sim_params['veg_type']=self.get_veg_type()
        
        #simulate manure/fertilizer applications
        self.gen_Manure()
        self.gen_Fert()
        
        
          
        
        
        
        
    def get_veg_type(self):
        if self.known_params['crop_type']=='Corn':
            if self.sim_params['cover_crop']:
                return 'Row crop + successful cover crop'
            else:
                return 'Corn & other row crops'
        else:
            return veg_type_dict[self.known_params['crop_type']]


    def gen_Manure(self):
        '''Generate Manure Applications for this field with simulated data. '''
        
        self.sim_params['manure_applications']=[]
        for i in range(int(self.sim_params['num_manure_applications'])):
            manure_params={'m_Type': 'Cow',
                           'm_rate': self.sim_params['total_p_applied']/self.sim_params['num_manure_applications']}
            
            
            for m_var in ['incorp_method', 'time_to_incorp', 'date']: #simulated manure parameters
                params={**manure_params, **self.known_params, **self.sim_params}
                manure_params[f'm_{m_var}']=variable_objs[f'm_{m_var}'].draw(**params)
            manure_params={key[2:]: value for key, value in manure_params.items()}
            
            #initate a manure application object
            self.sim_params['manure_applications'].append(SimManure(field=self, **manure_params))
   
        
   
     def gen_Fert(self):
         '''Generate Fertilizer Applications for this field with simulated data. '''
        self.sim_params['fertilizer_applications']=[]
        for i in range(int(self.sim_params['num_fert_applications'])):
            fertilizer_params={'p_rate':5}
            
            for p_var in ['p_incorp_method',
                'p_date']:
                params={**self.known_params,**self.sim_params}
                
                fertilizer_params[f'{p_var}']=variable_objs[f'{p_var}'].draw(**params)
            fertilizer_params={key[2:]:value for key, value in fertilizer_params.items()}                      
            self.sim_params['fertilizer_applications'].append(SimFert(field=self, **fertilizer_params))
        self.params={**self.sim_params, **self.known_params}          
     
     def USLE(self):
         #to do: set C and P params for USLE
        self.params['C']=.5
        self.params['P']=1
        
        self.params[]
        self.params['erosion_rate']=np.product([self.params[n] for n in ['RKLS', "C", 'P']])
        
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
            if field.sim_params['veg_type'] in [ 'Alfalfa & other hay crops', 
                                                 'CRP, other ungrazed, perm. veg.',
                                                 'Small grains']:
                return 'April - vegetated'
            else:
                return 'April - bare'
        else:
            return m_date_conversion_dic[date]      



class simulation():
    def __init__(self, directory):
        self.directory=directory
        self.fields=[]
        
    def load_data(self):
        shape_file=[f for f in os.listdir(self.directory) if 'shp' in f][0]
        path=os.path.join(self.directory, shape_file)
        gdf=gpd.read_file(path)
        
        with open(os.path.join(self.directory, 'column_names.txt')) as f:
            gdf.columns=[line for line in f.read().split('\n') if line]
        gdf['hydro_group']=gdf['hydro_group'].apply(lambda x: x.split('/')[-1])
        gdf['buffer_width']=gdf['distance_to_water']
        variables_path=r"C:\Users\benja\VT_P_index\model\sim_variables.txt"
        variable_objs=sim_variables.load_vars_csv(variables_path)
        
        for i, row in gdf.iterrows():
            self.fields.append(CropFieldFromShp(row, variable_objs))
            
        
        return self.fields, gdf
    
    def simPindex(self):
        for field in self.fields:
            field.calcPindex()



field_geodata()
sim=simulation(os.path.join(os.getcwd(), 'intermediate_data', 'SO01_fields'))
fields, gdf=sim.load_data()
sim.simPindex()




