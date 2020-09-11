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
import matplotlib.pyplot as plt
import seaborn as sns
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

hydro_int_dict={letter: i for i, letter in enumerate(['A', 'B', 'C', 'D'])}  

 
class CropFieldFromShp(CropField):
    '''Initate Crop Field from a single series of a shapefile.'''
    def __init__(self, series, params_to_sim):
        if series['crop_type']=='Other Crop':
            series['crop_type']='Small_Grain'
        
        self.params_to_sim=params_to_sim
        series['soil_is_clay']=bool(series['soil_is_clay'])
        self.known_params=series.to_dict()
        
        #sq meters to acres
        self.known_params['acres']=series['acres']
        self.idnum=self.known_params['IDNUM']
        self.set_rotation()
        self.combined_results=[]
        self.run_params=[]
        
    def calcPindex(self):
        CropField.calcPindex(self)
        self.results['erosion_rate']=self.params['erosion_rate']
        self.combined_results.append(self.results)
        self.run_params.append(self.sim_params)
        
    def initialize_sim(self, var_objs):
        '''Set up all needed inputs to run the P Index on Field'''
        self.simulate_params2(var_objs)
        self.setup_data()
        self.USLE()
        
    
    def all_data(self):
        '''Return a list of dictionaries, each representing the results and parameters
        for a different model run.'''
        return [{**self.known_params, **run_params, **results} 
                for run_params, results in 
                zip(self.run_params, self.combined_results)]
    
    def rotation_sanity_check(self):
        '''Reassign Highly Erodeable fields with implausible crop rotations.
        Land with potential erosion>100 tons/ac/yr 
        that is in Hay is considered continuous hay.
        Land with potential erosion >30 
        that isn't already assigned to continuous hay/fallow 
        is assigned to 2 years corn, 6 years hay. 
        '''
        crop_names=['Other', 'Fallow', 'Corn', 'Hay']
        results=()
        dic=self.known_params
        if dic['crop_type'] in ['Hay', 'Fallow'] and dic['RKLS']>=60:
              results= (0, 0, 0, 10)
        
        elif dic['RKLS']>=60 and dic['crop_type']=='Corn':
            self.known_params['crop_type']='Hay'
            results=(0,0, 0, 8)
              
        elif dic['RKLS']>=30 and dic['Years_Corn']>0:
            results=(0,0, 2, 6)
            
        if results and results!=tuple([dic[f'Years_{crop}'] for crop in crop_names]):
            print('Crop Rotation failed sanity check. Reassigning...')
            for i, crop in enumerate(crop_names):
                self.known_params[f'Years_{crop}']=results[i]
    
    
    
    def set_rotation(self):
        '''Assign the crop rotation to the field.'''
        self.rotation_sanity_check()
        crop_params={param.split('_')[1]: self.known_params[param]  
                     for param in self.known_params if 'Years_' in param}
      
        self.rotation=Rotation(**crop_params)
    
    
    def sim_rotation(self):
        '''Set variables related to crop sequence.'''
        crop_seq= self.rotation.draw_crops(self.known_params['crop_type'])
        is_establishment_year=(crop_seq[0]!=crop_seq[1])
        return crop_seq, is_establishment_year
    
    
    
    def simulate_params2(self, var_objs):
        '''Simulate Parameters for the field. 
        pass a dictionary of variable objects. '''
        global index_counter
        crop_seq, is_establishment_year=self.sim_rotation()
        
        self.sim_params={'crop_seq': crop_seq,
                         'is_establishment_year': is_establishment_year}
        

        #parameters simulated directly from variable objects
        variables=self.params_to_sim
        i=index_counter
        for name in variables:
            
            self.sim_params[name]=var_objs[name].drawFrom(i=i,
                                                **{**self.known_params, **self.sim_params})
            
            
        self.sim_params['veg_type']=self.get_veg_type()
        
        #simulate manure/fertilizer applications
        self.gen_Manure(var_objs, i=i)
        self.gen_Fert(var_objs, i=i)
        
        index_counter+=1
    
    def simulate_params(self, var_objs):
        '''Simulate Parameters for the field. 
        pass a dictionary of variable objects. '''
        crop_seq, is_establishment_year=self.sim_rotation()
        
        self.sim_params={'crop_seq': crop_seq,
                         'is_establishment_year': is_establishment_year}
        

        #parameters simulated directly from variable objects
        variables=self.params_to_sim
        
        
        for name in variables:
            
            self.sim_params[name]=var_objs[name].draw(size=1,
                                                **{**self.known_params, **self.sim_params})[0]
            
        self.sim_params['veg_type']=self.get_veg_type()
        
        #simulate manure/fertilizer applications
        self.gen_Manure(var_objs)
        self.gen_Fert(var_objs)
        
    
          
    def ensemble_results(self):
        sum_results={component:np.mean([r[component] for r in self.combined_results])
                                              for component in self.results.keys() }
        sum_results['adj_p_lost_lbs']=np.product([self.params['acres'],
                                                  sum_results['total p index'],
                                                  1/80])
        return sum_results
        
        
    def get_veg_type(self):
        '''Corrections for converting crop_type into veg_type'''
        
        if self.known_params['crop_type']=='Corn':
            if self.sim_params['cover_crop']:
                return 'Row crop + successful cover crop'
            else:
                return 'Corn & other row crops'
        else:
            return veg_type_dict[self.known_params['crop_type']]


    def gen_Manure(self, var_objs, i):
        '''Generate Manure Applications for this field with simulated data. '''
        
        self.sim_params['manure_applications']=[]
        for n in range(int(self.sim_params['num_manure_applications'])):
            manure_params=self.setup_manure_params(var_objs, i)
            m_app=SimManure(field=self, **manure_params)
            self.sim_params['manure_applications'].append(m_app)
   
    def setup_manure_params(self, var_objs, i):
        '''Gather all paramters to initialize a simulated manure application.'''
        
        manure_params={'manure_Type': 'Cow',
                           'manure_rate': 
                            (self.sim_params['total_p_applied']
                            /self.sim_params['num_manure_applications'])}
            
        #simulated manure parameters
        
        for m_var in ['incorp_method', 'time_to_incorp', 'date']: 
        
            params={**manure_params, **self.known_params, **self.sim_params}
            var_obj=var_objs[f'manure_{m_var}']
            manure_params[f'manure_{m_var}']=var_obj.drawFrom(i=i, **params)
            i-=50
        manure_params={key[7:]: value for key, value in manure_params.items()}
        return manure_params
   
    
    def gen_Fert(self, var_objs, i):
         '''Generate Fertilizer Applications for this field, using simulated data. '''
         self.sim_params['fertilizer_applications']=[]
         
         for n in range(int(self.sim_params['num_fert_applications'])):
            fertilizer_params={'fert_rate':5}
            
            for p_var in ['fert_incorp_method','fert_date']:
                
                params={**self.known_params,**self.sim_params}
                
                fertilizer_params[f'{p_var}']=var_objs[f'{p_var}'].drawFrom(i=i, **params)
            
            fertilizer_params={key[5:]:value for
                               key, value in fertilizer_params.items()}   
            
            fert_app=SimFert(field=self, **fertilizer_params)                
            self.sim_params['fertilizer_applications'].append(fert_app)
         self.params={**self.sim_params, **self.known_params}          
     
        
    def USLE(self):
        '''Set the erosion rate for the field.'''
        
         #to do: set param for USLE
        self.sim_params['C']=self.getC_fac()
        self.sim_params['P']=.8 #P will actually have to be simulated!
        self.sim_params['erosion_rate']=np.product([self.sim_params[n] for n in ["C", 'P']])*self.known_params['RKLS']
        self.params['erosion_rate']=self.sim_params['erosion_rate']
        
        
    def getC_fac(self):
        '''Retreive the C factor from USLE.'''
        if self.params['veg_type']=='Row crop + successful cover crop': 
             #modify crop sequence if its corn following cover crop
            crops=['Corn', 'Small_Grain']
            tillage_timing='spring'
        else:
            crops=self.params['crop_seq']
            tillage_timing=self.params['tillage_timing']
        
        for seq in crop_seqs:
            dic= seq.respond(crops)
            if dic:
                break
        try:
            c= dic[tillage_timing][self.params['tillage_method']]
        except KeyError:
            print(dic)
            print(tillage_timing)
            print(self.params['tillage_method'])
            print(self.params)
            raise
            
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
        manureApplication.__init__(
            self, field, rate, date, 
            time_to_incorp, incorp_method, Type)
    
    def get_timing(self, date, field):
        '''Convert p timing into the form the model wants.
        NOTE: TODO--> Deal with the possibility of previous-year cover crops. 
        Need to simulate PREVIOUS YEAR cropping to do this.
        
        '''
        if date=='spring':
            if (field.sim_params['cover_crop'] or 
                field.sim_params['veg_type'] in [ 'Alfalfa & other hay crops', 
                                                 'CRP, other ungrazed, perm. veg.',
                                                 'Small grains']):
                return 'April - vegetated'
                
            else:
                return 'April - bare'
        else:
            return m_date_conversion_dic[date]      

def fix_hydro_group(string):
    '''Arbitrarily assigns unrated soils to hydo_group_C'''
    if string in ['not rated', 'water', 'None']:
        return 'C'
    else:
        return string



class simulation():
    def __init__(self, data_directory, params_to_sim, variables_path, run_name):
        self.data_directory=data_directory
        self.fields=[]
        self.params_to_sim=params_to_sim
        self.var_objs=sim_variables.load_vars_csv(variables_path)
        self.run_name=run_name
        
    def load_data(self, cols_to_swap={}):
        '''Load in data from the shapefile in the simulation data_directory.'''
        gdf=load_shape_w_cols(self.data_directory)
        gdf=self.fix_data(gdf)  
        self.geometries=gdf['geometry']
        if cols_to_swap:
            gdf=swap_cols(gdf, cols_to_swap)
        
        for i, row in gdf.iterrows():
            self.fields.append(CropFieldFromShp(row, self.params_to_sim))
        
        return self.fields, gdf
    
    def load_subset(self, key, values):
        gdf=load_shape_w_cols(self.data_directory)
        gdf=gdf[gdf[key].isin(values)]
        self.geometries=gdf['geometry']
        gdf=self.fix_data(gdf)  
        for i, row in gdf.iterrows():
            self.fields.append(CropFieldFromShp(row, self.params_to_sim))
        
        return self.fields, gdf

    def fix_data(self, gdf):
        '''Make minor fixes to how data is represented. '''
        gdf['hydro_group']=gdf['hydro_group'].apply(lambda x: x.split('/')[-1])
        gdf['hydro_group']=gdf['hydro_group'].apply(fix_hydro_group)
        gdf['acres']=gdf['geometry'].area*0.000247105
        #this may call for a more complex method. 
        gdf['buffer_width']=gdf['distance_to_water']
        return gdf


    def simPindex(self, n_times=1):
        '''Simulate P Index for all of the fields.'''
        
        self.records=[]
        global index_counter
        
        for v in self.var_objs.values():
            print(f'Drawing up values for {v}')
            v.setup_for_draws(len(self.fields))
            
        for n in range(n_times):
            index_counter=0
            for v in self.var_objs.values():
                v.shuffle()
                
            print(f'Running Simulation #{n+1} out of {n_times}')
            for field in self.fields:
                field.initialize_sim(self.var_objs)
                field.calcPindex()
                self.records.append({**{'Sim Number': n}, 
                                     **field.params, **field.results})
           
                
    def load_test(self, size=100):
        '''Load a small subset for a test.'''
        
        gdf=load_shape_w_cols(self.data_directory)
        gdf=self.fix_data(gdf)
        self.geometries=gdf['geometry']
        
        for i, row in gdf.head(100).iterrows():
            self.fields.append(CropFieldFromShp(row, self.params_to_sim))
        
        return self.fields, gdf
    
    def ensemble_results(self):
        '''Return a dataframe of known parameters and ensemble mean for each field.'''
        
        sum_results=[{**field.known_params,**field.ensemble_results()} 
                     for field in self.fields]
        return pd.DataFrame(sum_results)

    
    
    def summary_charts(self, df, name_modifier):
        '''Create several charts to summarize the outputs of the simulation.'''
        
        charts_dir=os.path.join(os.getcwd(), 'results', self.run_name, 'charts')
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
            
        results_cols=['surface particulate loss',
        'surface dissolved loss',
        'subsurface loss', 'total p index']
        
        fig=sns.pairplot(data=df[results_cols+['crop_type']], 
                         hue='crop_type', plot_kws={'alpha':.3})
        plt.title(f'{self.run_name}   {name_modifier}')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f'results_{name_modifier}.png'))
        
        plt.show()
        fig=plt.hist(df['total p index'], bins=100)
        plt.title(f'{self.run_name}   {name_modifier}')
        plt.savefig(os.path.join(charts_dir, f'p_hist_{name_modifier}.png'))
        
        
        df['hydro_int']=df['hydro_group'].apply(lambda x: hydro_int_dict.get(x))
        input_cols=['RKLS',
        'erosion_rate',
         'soil_is_clay',
         'buffer_width',
         'hydro_int', 
         ]
         
        fig=sns.pairplot(data=df[input_cols+['total p index', 'crop_type']], 
                         hue='crop_type', plot_kws={'alpha':.3})
        plt.title(f'{self.run_name}   {name_modifier}')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f'component_{name_modifier}.png'))
        
    def save_results(self, charts=True):
        gdf=gpd.GeoDataFrame(self.ensemble_results(), geometry=self.geometries)
        gdf['hydro_int']=gdf['hydro_group'].apply(lambda x: hydro_int_dict.get(x))
        dir_path=os.path.join(os.getcwd(), 'results', self.run_name)
        df=pd.DataFrame(self.records)
        df['hydro_int']=df['hydro_group'].apply(lambda x: hydro_int_dict.get(x))
        
        if charts:
            self.summary_charts(gdf, 'ensemble_avgs')
            #self.summary_charts(df, 'all_runs')
        save_shape_w_cols(gdf, dir_path)
        df.to_csv(os.path.join(dir_path, 'all_runs.csv'))
        return gdf, df
    
    

def swap_cols(df, mapping):
    for key, value in mapping.items():
        df[key]=df[value]
    return df    
        
def total_manure_application(df):
    df['total_manure_applied']=df['total_p_applied']*df['acres']
    return np.mean( 
        [
        df[df['Sim Number']==n]['total_manure_applied'].sum() 
         for n in df['Sim Number'].unique()
         
         ])
    
def estimate_cow_number(df, county):
    '''Estimate the number of cows required to produce 
    this quantity of manure P
     70 lbs P2O5 per cow per year based on an average of values 
     from table 1 of this doc:
    https://ag.umass.edu/sites/ag.umass.edu/files/fact-sheets/pdf/EstimatingManureInventory%2811-30%29.pdf'''
    
    subdf=df[df['county']==county]
    subdf['p_applied']=subdf['total_p_applied']*subdf['acres']
    lbs_p=subdf.groupby('IDNUM')['p_applied'].mean().sum()
    return lbs_p/70
    
#%%
if __name__=='__main__':
    import time
    start=time.perf_counter()
    shapes_path=os.path.join(os.getcwd(), 'intermediate_data', 'aoi_fields')
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
    
    variable_sim_files=[f for f in os.listdir('variable_simulators') if 'experimental' not in f]
    for file in variable_sim_files:
        index_counter=0
        var_fp=os.path.join('variable_simulators', file)
        run_name=file.split('.')[0]
        sim=simulation(shapes_path, params_to_sim, var_fp, run_name)
        fields, gdf=sim.load_data()
        sim.simPindex(250)
        sim.save_results()
        
    end=time.perf_counter()
    print('Total Time to run:')
    print(end-start)

