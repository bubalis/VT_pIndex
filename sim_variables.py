# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:29:05 2020

@author: benja


Script for generating probability distributions and conditional logic 
for simulated variables from a csv file. 
"""


import random
from scipy import stats
import csv
import string 


def cell_parser(s):
    '''Process a cell from a csv.'''
    
    if not s:
        return ''
    
    #if cell is a list, return as a list
    if ',' in s:
        return [n.strip() for n in s.split(',')]
    
    #if the cell is a number
    elif all([char in string.digits+'.' for char in s]):
        return float(s)
    
    else:
        return s

def load_vars_csv(path):
    '''Load in Variables from a csv file. 
    Return a dictionary of variables. '''
    
    
    with  open (path) as csv_file:
        reader=csv.reader(csv_file, delimiter='\t')
        rows=[row for row in reader]
        keys=rows[0]
        variables={}
        for row in rows[1:]:
            dic={key:cell_parser(value) for key, value in zip(keys, row)}
            if dic['Name']:
                variables[dic['Name']]=Variable_from_dict(dic)
        return variables




class Variable():
    '''Class for any variable being simulated.'''
    
    def __init__(self, name, dist, **kwargs):
        self.dist_func=dist(**kwargs)
        self.name=name
        
    def draw(self, **kwargs):
        '''Draw a single random or conditional value.'''
        
        if self.type=='User_defined': #only user-defined variables can be passed arbitrary kwargs
            return self.dist_func.rvs(size=1, **kwargs)[0]
        
        
        else:
            return self.dist_func.rvs(size=1)[0]
    
    def draw_no_replacement(self, size, **kwargs):
        '''Make multiple draws without replacement.'''
        results=[]
        for i in range(size):
            while True:
                rv=self.draw(**kwargs)
                if rv not in results:
                    results.append(rv)
                    break
        return results
    
    def __repr__(self):
        return f'Variable Generator: {self.name} \n {self.dist_func}'
    
class Variable_from_dict(Variable):
    '''Initialize a variable from a dictionary: 
    keys: Name - variable name.
    Distribution: name of their probability distribution.
    param {i} name: name of parameter.
    param {i} value: value of parameter. 
    Up to i==5. '''
    
    def __init__(self, dic):
        self.name=dic['Name']
        kwargs={}
        for i in range(1,6):
            kwargs[dic[f'param {str(i)} name']]=dic[f'param {str(i)} value']
        kwargs={k:v for k,v in kwargs.items() if k}
        dist=dic['Distribution']
        
        
        
        d=globals().get(dist)
        if  d: #  if the variable is defined in this script
            self.dist_func=d(**kwargs, name=self.name)
            self.type='User_defined'
        else: #if the variable is defined in an imported package
            
            module=globals().get(dist.split('.')[0])
            d=getattr(module, dist.split('.')[1])
            self.dist_func=d(**kwargs)
            self.type='From_package'
        
class prob_dist(): 
    '''Metaclass for self-designed probability distributions.'''
    
    def rvs(self, size=1, **kwargs):
        if size==1:
            try:
                return [self.draw1(**kwargs)]
            except:    
                print(self)
                raise ValueError
        else:
            return [self.draw1(**kwargs) for i in range(size)]


    
class categorical(prob_dist):
    '''Class for a random categorical Variable.
    Pass a probability dictionary:
    keys: categories, 
    values: their relative probabilities.
     '''
    def __init__(self, name, **kwargs):
        self.draw1=setup_categorical_variable(kwargs)
    


class constant(prob_dist):
    '''A variable set to a constant.'''
    def __init__(self, n, **kwargs):
        self.n=n
    
    def draw1(self, **kwargs):
        return self.n
    

class echo(prob_dist):
    '''Returns the value from a given key in the dictionary that is passed.'''
    
    def __init__(self, echo_field, **kwargs):
        self.echo_field=echo_field
    
    def draw1(self, **kwargs):
        return kwargs[self.echo_field]
    
class based_on_function(prob_dist):
    '''Return a value based on a function. '''
    def __init__(self, func, function_args):
        def draw1(self, function_args):
            return func(function_args)


    
def setup_categorical_variable(probabilities):
    ''' 
    Create a function to draw a categorical variable from. 
    '''
    sum_prob=sum(probabilities.values())
    if sum_prob!=1:
        print('Warning: Probabilites do not sum to 1. Calculating based on relative probabilities')
    
    i=0
    out_vals=[]
    
    for cat, prob in probabilities.items():
        i+=prob/sum_prob
        out_vals.append((i, cat))
        
    def draw1(**kwargs):
        p=random.random()
        for thresh, cat in out_vals:
            if p<thresh:
                return cat
            
        
    return draw1

class conditional(prob_dist):
    '''A probability object that draws from one of several other probability objects,
    depending on the parameters passed.
    name: a str
    cond_variable: the variable name that the result is conditional on.
    sub_variables: the options for the condition value that have their own unique distributions.
    else_name: names of variables that will return for else values. '''
    
    def __init__(self, name, cond_variable, sub_variables,  else_names=[], **kwargs):
        self.name=name
        self.cond_variable=cond_variable
        self.sub_variables=sub_variables
        self.else_names=else_names
        assert self.sub_variables, self.name
        
    def draw1(self, **kwargs):
        category=kwargs[self.cond_variable]
        try:
            if category in self.sub_variables:
                function=globals()['variables'][f'{self.name}__{category}']
                return function.draw(**kwargs)
            elif category in self.else_names:
                return globals()['variables'][f'{self.name}__else'].draw(**kwargs)
            else:
                raise ValueError
        except:
            print('ERROR:')
            print(self.else_names, self.sub_variables)
            print(category)
            raise ValueError

    

variables=load_vars_csv('sim_variables.txt')

                  
                     
                        
     
     
   

    



        