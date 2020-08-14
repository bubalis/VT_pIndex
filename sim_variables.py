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
    
    #if cell is string of a boolean, return as bool
    if s in ['True', 'False']:
        return str_to_bool(s)
    
    #if cell is a list, return as a list
    elif ',' in s:
        return [cell_parser(n) for n in s.split(',')]
    
    
    elif not s:
        return ''
    
    #if the cell is a number, parse as number
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
                variables[dic['Name']]=Variable_from_dict(dic, variables)
        
        return variables




class Variable():
    '''Class for any variable being simulated.'''
    
    def __init__(self, name, dist, **kwargs):
        self.dist_func=dist(**kwargs)
        self.var_name=name
        
    def draw(self, **kwargs):
        '''Draw a single random or conditional value.'''
        return self.dist_func.rvs(size=1, **kwargs)[0]
            
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
        return f'Variable Generator: {self.var_name} \n {self.dist_func}'
    
class Variable_from_dict(Variable):
    '''Initialize a variable from a dictionary: 
    keys are:
    Name : variable name.
    Distribution: name of their probability distribution.
    param {i} name: name of parameter.
    param {i} value: value of parameter. 
    Up to i==5. '''
    
    
    def __init__(self, dic, variable_dic):
        
        self.var_name=dic['Name']
        self.description=dic['Description']
        kwargs={}
        for i in range(1,6):
            kwargs[dic[f'param {str(i)} name']]=dic[f'param {str(i)} value']
        kwargs={k:v for k,v in kwargs.items() if k}
        dist=dic['Distribution']
        
        self.dist_func=ProbDist(
            dist, self.var_name, 
            variable_dic, **kwargs)

        
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
        raise ValueError(
            f'"{dist_name}" not defined in script or in imported package')
    
class ProbDist(): 
    '''Metaclass for probability distributions.'''
    def __init__(self, dist_name, var_name, variable_dic, **kwargs):
        self.var_name=var_name
        self.dist_name=dist_name
        self.all_variables=variable_dic
        
        #
        d=globals().get(dist_name)
        if  d: #  if the variable is defined in this script
           d.__init__(self, **kwargs, name=var_name)
             
        elif '.' in dist_name: #if dist is defined in an imported package:
            try:
                ProbDistFromMod.__init__(
                    self, dist_name=dist_name, 
                    var_name=var_name, **kwargs, )
            except TypeError:
                print(
                f'{var_name} received invalid KW argument for {dist_name}')
                raise
        else:
           raise ValueError(
               f'"{dist_name}" not defined in script or in imported package')
    
        
    
    def __repr__(self):
        return f'{self.var_name}, {self.dist_name}'
    
    def rvs(self, size=1, **kwargs):
        try:
            return [self.draw1(**kwargs) for i in range(size)]
        except:
            print(self)
            raise

class ProbDistFromMod(ProbDist):
    '''Probability Distribution from an imported module'''
    
    def __init__(self, dist_name, var_name, **kwargs):
        
        module=globals().get(dist_name.split('.')[0])
        dist=getattr(module, dist_name.split('.')[1])
        self.frozen_dist=dist(**kwargs)
        def draw1(**kwargs):
            return self.frozen_dist.rvs()
        self.draw1=draw1
        
        
    
    
class categorical(ProbDist):
    '''Class for a random categorical Variable.
    Pass a probability dictionary:
    keys: categories, 
    values: their relative probabilities.
     '''
    def __init__(self, name, **kwargs):
        draw1=setup_categorical_variable(kwargs)
        self.draw1=draw1
        
def setup_categorical_variable(probabilities):
    ''' Create a function to draw a categorical variable from. '''
    
    
    sum_prob=round(sum(probabilities.values()), 8)
    if sum_prob !=1:
        print(f'''Warning: Probabilites sum to {sum_prob} not 1. 
              Calculating based on relative probabilities''')    
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



class constant(ProbDist):
    '''A "distribution" that returns a constant.'''
    def __init__(self, n, **kwargs):
        def draw1(**kwargs):
            return n
        self.draw1=draw1
    

class echo(ProbDist):
    '''A "distribution that the value from a given key 
    in the dictionary that is passed to it.'''
    
    def __init__(self, echo_field, **kwargs):
        def draw1(**kwargs):
            try:
                return kwargs[echo_field]
            except KeyError:
                print (f'Dictionary did not include Echo Field.')
                print (kwargs)
                raise
        self.draw1=draw1
    


class dist_plus_function(ProbDist):
    '''Create a variable simulated by a drawing from a distribution, 
    then altering the result by some function.'''
    
    def __init__(self, dist_name, function, dist_kws, **kwargs):
        dist_kwargs={dist_kw: kwargs[dist_kw] 
                     for dist_kw in ensure_list(dist_kws)}
        dist=ProbDist(
            dist_name, var_name="Temporary", 
            variable_dic=None, **dist_kwargs)
        
        def draw1(**kwargs):
            return function(dist.draw1(**kwargs), **kwargs)
        
        self.draw1=draw1
        
    
class ceil_lognorm(ProbDist):
    '''Lognormal distribution rounded up to nearest integer.'''
    
    def __init__(self, **kwargs):
        def ceil(x, **kwargs):
            return math.ceil(x)
        
        dist_plus_function.__init__(self, 'stats.lognorm', 
                                    ceil, dist_kws=['s'], **kwargs)
        
        

class dist_w_bool(dist_plus_function):
    '''A distribtution which returns a function if 
    a boolean statement evaluates as true, otherwise returns 0'''
    
    def __init__(self, distribution, dist_kws, 
                 bool_statement, else_response, **kwargs):
        
        def function(x, **kwargs):
            for key, value in kwargs.items(): 
                locals()[key]=value
            response= x*eval(bool_statement)
            
            if not response:
                return ensure_list(else_response)[0]
            return ensure_list(response)[0]
            
        dist_plus_function.__init__(self, distribution, 
                                    function, dist_kws, **kwargs)
        

class conditional(ProbDist):
    '''A probability object that draws from one of several other probability objects,
    depending on the parameters passed.
    name: a str
    cond_variable: the variable name that the result is conditional on.
    sub_variables: the options for the condition value 
    that have their own unique distributions.
    else_name: names of variables that will return for else values. '''
    
    def __init__(self, name, cond_variable, 
                 sub_variables,  else_names=[], **kwargs):
        
        self.cond_variable=cond_variable
        self.sub_variables=sub_variables
        self.else_names=else_names
        
        def draw1(**kwargs):
            category=kwargs[self.cond_variable]
            if category in self.sub_variables:
                function=self.all_variables[f'{self.var_name}__{str(category)}']
                return function.draw(**kwargs)
            
            elif category in self.else_names:
                function=self.all_variables[f'{self.var_name}__else']
                return function.draw(**kwargs)
            
            else:
                raise ValueError(
                    f'''{category} not in sub_variables or else_names: 
                                 {self.sub_variables} 
                                 {self.else_names}''' )
        self.draw1=draw1
        
if __name__=='__main__':
    variables=load_vars_csv(r"C:\Users\benja\VT_P_index\model\variable_simulators\bad_news.txt")

                  
                     
                        
     
     
   

    



        