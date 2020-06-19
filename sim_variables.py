# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:29:05 2020

@author: benja
"""
import random
from scipy import stats

class variable():
    
    def __init__(self, dist, name, **kwargs):
        self.dist_func=dist(**kwargs)
        self.name=name
        
    def draw(self, **kwargs):
        return self.dist_func.rvs(**kwargs)
    
    def __repr__(self):
        print(f'Variable Generator: {self.name}')
        print(self.dist_func)
        
        
class categorical():
    '''Class for a random categorical Variable.
    Pass a probability dictionary:
    keys: categories, 
    values: their relative probabilities.
    Probabiltie'''
    def __init__(self, probabilities):
        self.rvs=setup_categorical_variable(probabilties)
    
        
    
def setup_categorical_variable(probabilities):
    ''' 
    Create a function 
    '''
    sum_prob=sum(probabilties.values())
    if sum_prob!=1:
        print('Warning: Probabilites do not sum to 1. Calculating based on relative probabilities')
    
    i=0
    out_vals=[]
    for cat, prob in probabilites.items():
        i+=prob/sum_probs
        out_vals.append((i, cat))
        
    def draw1():
        p=random.random()
        for thresh, cat in out_vals:
            if p<thresh:
                return cat
            
    def rvs(size=1):
        if size=1:
            return draw1()
        else:
            return [draw1() for i in range(size)]
        
    return rvs


    

        