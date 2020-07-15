# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:29:05 2020

@author: benja


Script for generating probability distributions and conditional logic 
for simulated variables from a tsv file.

 
"""


import random
from scipy import stats
import csv
import string 
import math


def str_to_bool(string):
    if string.lower()=='true':
        return True
    elif string.lower()=='false':
        return False
    

def cell_parser(s):
    '''Process a cell from the tsv file.'''
    s=s.strip()
    if s in ['True', 'False']:
        return str_to_bool(s)
    
    #if cell is a list, return as a list
    elif ',' in s:
        return [cell_parser(n) for n in s.split(',')]
    
    #if the cell is a number
    elif not s:
        return ''
    elif all([char in string.digits+'.' for char in s]):
        return float(s)
        
    else:
        return s.strip()
    


def ensure_list(item):
    '''If item is a list, return it. 
    Else, return item as a 1-element list'''
    if type(item)==list:
        return item
    else:
        return [item]
#%%   

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
        for var in variables.values():
            var.connect_to_all(variables)
        return variables




class Variable():
    '''Class for any variable being simulated.'''
    
    def __init__(self, name, dist, **kwargs):
        self.dist_func=dist(**kwargs)
        self.name=name
        
    def draw(self, **kwargs):
        '''Draw a single random or conditional value.'''
        
        if self.type=='User_defined': #only user-defined variables can be passed arbitrary kwargs
            try:    
                return self.dist_func.rvs(size=1, **kwargs)[0]
            except TypeError:
                print(self)
                raise TypeError
        
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
    
    def connect_to_all(self, variables):
        self.dist_func.all_variables=variables
    
class Variable_from_dict(Variable):
    '''Initialize a variable from a dictionary: 
    keys: Name - variable name.
    Distribution: name of their probability distribution.
    param {i} name: name of parameter.
    param {i} value: value of parameter. 
    Up to i==5. '''
    
    def __init__(self, dic):
        
        self.name=dic['Name']
        self.description=dic['Description']
        kwargs={}
        for i in range(1,6):
            kwargs[dic[f'param {str(i)} name']]=dic[f'param {str(i)} value']
        kwargs={k:v for k,v in kwargs.items() if k}
        dist=dic['Distribution']
        
        self.dist_func, self.type=create_dist(dist, self.name, **kwargs)
        '''d=globals().get(dist)
        if  d: #  if the variable is defined in this script
            self.dist_func=d(**kwargs, name=self.name)
            self.type='User_defined'
            
        elif '.' in dist: #if the variable is defined in an imported package
            
            module=globals().get(dist.split('.')[0])
            d=getattr(module, dist.split('.')[1])
            self.dist_func=d(**kwargs)
            self.type='From_package'
            
        
            
        else:
            raise ValueError(f'"{dist}" not defined in script or in imported package')'''
        

def create_dist(dist_name, var_name, **kwargs):
    '''Retrieve distribution from globals and initialize it. 
    Return a probability distribution obj and 
    a string indicating whether the dist is User-defined or from a package.
    '''
    d=globals().get(dist_name)
    if  d: #  if the variable is defined in this script
        return d(**kwargs, name=var_name), 'User_defined'
    elif '.' in dist_name: #if the variable is defined in an imported package:
        module=globals().get(dist_name.split('.')[0])
        d=getattr(module, dist_name.split('.')[1])
        return d(**kwargs), 'From_package'   
     
    else:
        raise ValueError(f'"{dist_name}" not defined in script or in imported package')
    
class prob_dist(): 
    '''Metaclass for self-designed probability distributions.'''
    
    def rvs(self, size=1, **kwargs):
        
        if size==1:
            return [self.draw1(**kwargs)]
                
        else:
            return [self.draw1(**kwargs) for i in range(size)]
        

    #def __repr__(self):
     #   return f'Distribution object: {self.name}'
    
    
class categorical(prob_dist):
    '''Class for a random categorical Variable.
    Pass a probability dictionary:
    keys: categories, 
    values: their relative probabilities.
     '''
    def __init__(self, name, **kwargs):
        self.draw1=setup_categorical_variable(kwargs)
    
def setup_categorical_variable(probabilities):
    ''' 
    Create a function to draw a categorical variable from. 
    '''
    sum_prob=round(sum(probabilities.values()), 8)
    if sum_prob !=1:
        print(f'Warning: Probabilites sum to {sum_prob} not 1. Calculating based on relative probabilities')

    
    i=0
    out_vals=[]
    
    for cat, prob in probabilities.items():
        i+=prob/sum_prob
        out_vals.append((i, cat))
        
    def draw1(**kwargs):
        '''Draw a random # 0-1.
        Return the lowest category whose threshold is higher than the
        category's threshold.'''
        p=random.random()
        for thresh, cat in out_vals:
            if p<thresh:
                return cat
            
        
    return draw1


class constant(prob_dist):
    '''A variable set to a constant.'''
    def __init__(self, n, **kwargs):
        def draw1(**kwargs):
            return n
        self.draw1=draw1
    

class echo(prob_dist):
    '''Returns the value from a given key in the dictionary that is passed.'''
    
    def __init__(self, echo_field, **kwargs):
        def draw1(**kwargs):
            try:
                return kwargs[echo_field]
            except:
                print (kwargs)
            assert False
        self.draw1=draw1
    
def kwarg_wrapper(func):
    def wrapper(**kwargs):
        return func()


class dist_plus_function(prob_dist):
    '''Create a variable simulated by a drawing from a distribution, then altering the result by some function.'''
    def __init__(self, dist_name, function, dist_kws, **kwargs):
        dist_kwargs={dist_kw: kwargs[dist_kw] for dist_kw in ensure_list(dist_kws)}
        dist, dist_type=create_dist(dist_name, var_name=None, **dist_kwargs)
        def draw1(**kwargs):
            if dist_type=='User_defined':
                return function(dist.rvs(**kwargs),  **kwargs)
            elif dist_type=='From_package':
                return function(dist.rvs(), **kwargs)
        self.draw1=draw1
        
    
class ceil_lognorm(prob_dist):
    '''Lognormal distribution rounded up to nearest integer.'''
    def __init__(self, **kwargs):
        def ceil(x, **kwargs):
            return math.ceil(x)
        dist_plus_function.__init__(self, 'stats.lognorm', ceil, dist_kws=['s'], **kwargs)
        
        

class dist_w_bool(dist_plus_function):
    def __init__(self, distribution, dist_kws, bool_statement, else_response, **kwargs):
        
        def function(x, **kwargs):
            for key, value in kwargs.items(): 
                locals()[key]=value
            response= x*eval(bool_statement)
            if not response:
                return ensure_list(else_response)[0]
            return ensure_list(response)[0]
            
        dist_plus_function.__init__(self, distribution, function, dist_kws, **kwargs)
        

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
        if category in self.sub_variables:
            function=self.all_variables[f'{self.name}__{str(category)}']
            return function.draw(**kwargs)
        
        elif category in self.else_names:
            return self.all_variables[f'{self.name}__else'].draw(**kwargs)
        
        
        else:
            raise ValueError(f'''{category} not in sub_variables or else_names: 
                             {self.sub_variables} 
                             {self.else_names}''' )

    
if __name__=='__main__':
    variables=load_vars_csv('sim_variables.txt')

                  
                     
                        
     
     
   

    



        