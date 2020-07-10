# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 08:02:04 2020

@author: benja

Functions for simulating and retrieving values for the RUSLE crop-cover ('C') factor.

"""
import random 
import csv
import math




class cropSeq(object):
    '''Class for a crop sequences to calculate C_factor'''
    
    def __init__(self, crops, tillage_dict):
        self.crops=[crop for crop in crops if crop]
        self.tillage_dict=tillage_dict
    
    def check_match(self, field):
        '''Check if field's crop sequence matches crop sequence.
        '''
        return (len(field.params['crop_seq'])>=len(self.crops) and 
        [all([x==y for x,y in zip(field.params['crop_seq'], self.crops)])])
            
    def respond(self, field):
        '''If field and sequence match, return the tillage dict.'''
        if self.check_match(field):
            return self.tillage_dict
    
    def __repr__(self):
        return 'Crop sequence:  ' +'\n'.join(self.crops)
    
#%%
def tryfloat(string):
    try:
        return float(string)
    except ValueError:
        return math.nan

def load_sequences(path=r"C:\Users\benja\VT_P_index\model\Source_data\C_factors.csv"):
    with open(path) as csv_file:
        reader=csv.reader(csv_file)
        lines=[line for line in reader]
    
    keys=lines[0]
    keys[0]='Crop'
    
    tillage_dicts={}
    
    for line in lines[1:]:
        crops=tuple([cell for cell in line[0:5]])
        dic={line[5]:{key: tryfloat(cell) for key, cell in zip(keys[6:10], line[6:10])}}
        if crops in tillage_dicts:
            tillage_dicts[crops]={**tillage_dicts[crops], **dic}
        else:
            tillage_dicts[crops]=dic
    
    crop_seqs=[cropSeq(crops, tillage_dict) for crops, tillage_dict in tillage_dicts.items()]
    
    return sorted(crop_seqs, key=lambda x: len(x.crops), reverse=True)


crop_seqs=load_sequences()
#%%    
    
def search_crop_seq(crops):
    '''Return the C factor sub-dictionary for a given crop sequence.'''
    
    dic=crop_seqs
    
    for crop in crops:
        if 'Tillage' in dic:
            return dic
        else:
            return dic
            if crop in dic:
                dic=dic[crop]
            else:
                return dic['Other']
            
def get_cFactor(crops, tillage):
    '''Return the C factor for a crop sequence and tillage pattern. '''
    return search_crop_seq(crops)[tillage]
    
    
class Rotation(object):
    '''A crop rotation. Used for drawing a multi-year crop sequence.'''
    
    def __init__(self, **crops):
        self.years=[]
        for key, value in crops.items():
            self.years+=[key]*value
        self.years
    
    def set_year(self, crop_name):
        '''Choose a random start year for a crop sequence,
        where the starting crop is == to crop_name.'''
        while True:
            i=random.randint(0, len(self.years)-1)
            if self.years[i]==crop_name:
                return i
        
        
    def draw_crops(self, crop_name):
        '''Return a list of crops for this field. 
        The list represents previous crops in the sequence.
        list[0]: current crop/
        list[1]: last year's crop. 
        etc...'''
        if crop_name in self.years:
            i=self.set_year(crop_name)
            return [self.years[(n+i) % len(self.years)] 
                        for n in range(len(self.years), 0,  -1)]
        elif crop_name=='Fallow':
            return ['Fallow', 'Fallow']
        else:
            return [crop_name]+self.years
    
    
        


class continuous(Rotation):
    def draw_crops(self, crop_name):
        return [crop_name]*10




r=Rotation(Corn= 2, 
            Hay=3)

r.draw_crops('Corn')