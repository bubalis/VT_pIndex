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
import os
import numpy as np


def flatten(li):
    while '[' not in str(li):
        li=[x for y in li for x in y]
    return li

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
    
    elif s=='inf':
        return math.inf
    
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
        
    def draw(self, size,  **kwargs):
        '''Draw a single random or conditional value.'''
        try:
            return self.dist_func.rvs(size=size, **kwargs)
        except:
            print(self)
            raise
            
    def draw1(self, **kwargs):
        return self.draw(size=1,**kwargs)[0]
            
    def draw_no_replacement(self, size, **kwargs):
        '''Make multiple draws without replacement.'''
        results=[]
        for i in range(size):
            while True:
                rv=self.draw(size=1, **kwargs)[0]
                if rv not in results:
                    results.append(rv)
                    break
        return results
    
    def __repr__(self):
        return f'Variable Generator: {self.var_name} \n {self.dist_func}'
    
    def drawFrom(self, i, **kwargs):
        return self.dist_func.drawFrom(i, **kwargs)
    
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
        try:
            self.description=dic['Description']
            kwargs={}
            for i in range(1,6):
                kwargs[dic[f'param {str(i)} name']]=dic[f'param {str(i)} value']
            kwargs={k:v for k,v in kwargs.items() if k}
            dist=dic['Distribution']
            
            self.dist_func=create_dist(
                dist, self.var_name, 
                variable_dic, **kwargs)
        except:
            print(self.var_name)
            print(kwargs)
            print(variable_dic)
            raise
            

        
def create_dist(dist_name, var_name,  variable_dic, **kwargs):
    '''Retrieve distribution from globals and initialize it. 
    Return a probability distribution obj and 
    a string indicating whether the dist is User-defined or from a package.
    '''
    d=globals().get(dist_name)
    
    if  d: #  if the variable is defined in this script
        return d(var_name=var_name, variable_dic=variable_dic, **kwargs)
    
    elif '.' in dist_name: #if the variable is defined in an imported package
        return ProbDistFromMod(dist_name, var_name, variable_dic, **kwargs) 
     
    else:
        raise ValueError(
            f'"{dist_name}" not defined in script or in imported package')
    
 
   
  
class ProbDist(): 
    '''Metaclass for probability distributions.'''
    def __init__(self, dist_name, var_name, variable_dic, **kwargs):
        self.var_name=var_name
        self.dist_name=dist_name
        self.all_variables=variable_dic
               
    
    def __repr__(self):
        return f'{self.var_name}, {self.dist_name}'
    
    def rvs(self, size, **kwargs):
        response= self.draw(size=size, **kwargs) 
        try:
           
            return response
        except:
            print('Error')
            
            print(self)
            print(response[0])
            globals()['response']=response
            print('Program should terminate here')
            raise
            assert False
    
    def setup_for_draws(self, n):
        self.array=self.rvs(n)
    
    def drawFrom(self, i, **kwargs):
        return self.array[i]
    
class customDist(ProbDist):
    def __init__(self,  dist_name, var_name, variable_dic, **kwargs):
        ProbDist.__init__(self, dist_name, var_name, variable_dic, **kwargs)

    def draw1(self, **kwargs):
        return self.draw(size=1, **kwargs)[0]
    
    
class Dist_w_conditional(customDist):
    
    def drawFrom(self, i, **kwargs):
        kwargs.update({'i': i})
        assert 'i' in kwargs
        return self.draw1(**kwargs)
    
    def setup_for_draws(self, n):
        pass
    
    
        
class ProbDistFromMod(ProbDist):
    '''Probability Distribution from an imported module.
    e.g. any distribution in the scipy.stats library:
        stats.nbinom, stats.normal etc'''
    
    def __init__(self, dist_name, var_name, variable_dic, **kwargs):
        ProbDist.__init__(self, dist_name, var_name, variable_dic, **kwargs)
        module=globals().get(dist_name.split('.')[0])
        dist=getattr(module, dist_name.split('.')[1])
        self.frozen_dist=dist(**kwargs)
    
    def draw(self, size, **kwargs):
        return self.frozen_dist.rvs(size)
        
        
    
    
class categorical(customDist):
    '''Class for a random categorical Variable.
    Pass a probability dictionary:
    keys: categories, 
    values: their relative probabilities.
     '''
    def __init__(self, var_name, variable_dic, **kwargs):
        ProbDist.__init__(self, 'categorical', var_name, variable_dic, **kwargs)
        draw=setup_categorical_variable(kwargs)
        self.draw=draw

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
        
    def draw(size, **kwargs):
        '''Draw a random # 0-1.
        Return the lowest category whose threshold is higher than the
        category's threshold.'''
        p_s=np.random.random(size)
        out=[]
        for p in p_s:
            for thresh, cat in out_vals:
                if p<thresh:
                    out.append(cat)
        return out
            
    return draw

class numericConditional(Dist_w_conditional):
    '''A distribution that draws from one of several underlying distributions 
    based on the value of a given field. 
    Pass a set of '''
    
    def __init__(self, cond_variable, var_name, variable_dic, **thresholds):
        ProbDist.__init__(self, 'numericConditional', 
                          var_name, variable_dic=variable_dic)
        sub_variables=thresholds.keys()
        
        def draw1(**kwargs):
            try:
                check_val=kwargs[cond_variable]
                for key, value in thresholds.items():
                    if check_val<=value:
                        category=key
                        break
                else:
                    raise ValueError(
                        "Could not return Proper Value for numericCategorical Distribution")
            
    
                if category in sub_variables:
                        function=self.all_variables[f'{self.var_name}__{str(category)}']
                        return function.drawFrom(**kwargs)
                        
                else:
                    raise ValueError(
                        f'''{category} not in sub_variables: 
                                     {sub_variables} 
                                     {self.else_names}''' )
            
            except:
                print(cond_variable)
                print(check_val)
                print(kwargs)
                raise
                
        self.draw1=draw1
        
        def draw(size, **kwargs):
            return [self.draw1(**kwargs) for n in range(size)]
            
            
        self.draw=draw
        
        
        
        
        

class constant(ProbDist):
    '''A "distribution" that returns a constant.'''
    def __init__(self, var_name, variable_dic, n, **kwargs):
        ProbDist.__init__(self, 'constant', var_name, variable_dic=variable_dic)
        
        def draw(size, **kwargs):
            assert size
            if size==1:
                return [n]
            else:
                return [n]*size
        self.draw=draw
    

class echo(Dist_w_conditional):
    '''A "distribution" that returns the value from a given key 
    in the dictionary that is passed to it.
    '''
    
    def __init__(self, echo_field, var_name, variable_dic, **kwargs):
        ProbDist.__init__(self, 'echo', var_name, variable_dic=variable_dic)
        def draw(size, **kwargs):
            results=[]
            for i in range(size):
                try:
                    results.append( kwargs[echo_field])
                except KeyError:
                    print (f'Dictionary did not include Echo Field.')
                    print (kwargs)
                    raise
            return results
        self.draw=draw
       
    


class dist_plus_function(customDist):
    '''Create a variable simulated by a drawing from a distribution, 
    then altering the result by some function.'''
    
    def __init__(self, dist_name, function, dist_kws, **kwargs):
        
        dist_kwargs={dist_kw: kwargs[dist_kw] 
                     for dist_kw in ensure_list(dist_kws)}
        dist=create_dist(
            dist_name, var_name="Temporary", 
            variable_dic=None, **dist_kwargs)
        print(dist)
        
        def draw(size, **kwargs):
            return [function(dist.draw(size, **kwargs), **kwargs) for index in range(size)]
        
        self.draw=draw
        
    
class ceil_lognorm (dist_plus_function):
    '''Lognormal distribution rounded up to nearest integer.'''
    
    def __init__(self,var_name, variable_dic, **kwargs):
        ProbDist.__init__(self, 'ceil_lognorm', var_name, variable_dic=variable_dic)
        
        def ceil(x, **kwargs):
            return np.ceil(x)
        
        dist_plus_function.__init__(self, 'stats.lognorm', 
                                    ceil, dist_kws=['s'], **kwargs)
        
class poissonMax (dist_plus_function):
     def __init__(self, max_val, var_name, variable_dic, **kwargs):
         ProbDist.__init__(self, 'poissonMax', var_name, variable_dic=variable_dic)
         def enforce_max(x, **kwargs):
             return np.where(x<max_val, x, max_val)
         dist_plus_function.__init__(self, 'stats.poisson', enforce_max, dist_kws=['mu'], **kwargs)

class dist_w_bool (dist_plus_function, Dist_w_conditional):
    '''A distribtution which returns a function if 
    a boolean statement evaluates as true, otherwise returns 0'''
    
    def __init__(self, var_name, variable_dic, distribution, dist_kws, 
                 bool_statement, else_response, **kwargs):
        ProbDist.__init__(self, 'dist_w_bool', var_name, variable_dic=variable_dic)
        def function(x, **kwargs):
            for key, value in kwargs.items(): 
                locals()[key]=value
            response= x*eval(bool_statement)
            
            if not response:
                return ensure_list(else_response)[0]
            return ensure_list(response)[0]
            
        dist_plus_function.__init__(self, distribution, 
                                    function, dist_kws, **kwargs)
        

class conditional(Dist_w_conditional):
    '''A probability object that draws from one of several other probability objects,
    depending on the parameters passed.
    name: a str
    cond_variable: the variable name that the result is conditional on.
    sub_variables: the options for the condition value 
    that have their own unique distributions.
    else_name: names of variables that will return for else values. '''
    
    def __init__(self, var_name, cond_variable, 
                 sub_variables,  else_names=[], **kwargs):
        ProbDist.__init__(self, 'conditional', var_name, **kwargs)
        
        self.cond_variable=cond_variable
        self.sub_variables=sub_variables
        self.else_names=else_names
        
        def draw1(**kwargs):
            category=kwargs[self.cond_variable]
            if category in self.sub_variables:
                function = self.all_variables[f'{self.var_name}__{str(category)}']
                return function.drawFrom(**kwargs)
            
            elif category in self.else_names:
                function = self.all_variables[f'{self.var_name}__else']
                return function.drawFrom(**kwargs)
            
            else:
                raise ValueError(
                    f'''{category} not in sub_variables or else_names: 
                                 {self.sub_variables} 
                                 {self.else_names}''' )
        
        def draw(size, **kwargs):
            draws= [draw1(**kwargs) for i in range(size)]
            
            return draws
        self.draw=draw
        self.draw1=draw1

        
def extract_lines(csv_file):
    with open(csv_file) as file:
        reader=csv.reader(file, delimiter='\t')
        return [r for r in reader]

def copy_entries(var_name, from_sheet, to_sheet):
    '''Copy an entry or set of entries from one sheet to another sheet.'''
    
    lines=extract_lines(from_sheet)
    lines_to_copy=[line for line in lines if var_name in line[0]]
    
    
    new_lines=extract_lines(to_sheet)
    
    for i, line in new_lines:
        if var_name in line[0]:
            break
    
    new_lines=[line for line in new_lines if var_name not in line[0]]
    
    new_lines=new_lines[:i]+lines_to_copy+[[], []]+new_lines[i:]
    
    with open(to_sheet) as f:
        for line in new_lines:
            print(line.split('\t'), file=f)



def copy_to_all(var_name, from_sheet):
    '''Copy an entry or set of entries to all other variable simulator sheets
    in the directory.'''
    
    to_sheets=[f for f in os.listdir('variable_simulators') if from_sheet!=f]
    for to_sheet in to_sheets:
        copy_entries(var_name, from_sheet, to_sheet)

if __name__=='__main__':
    variables=load_vars_csv(r"C:\Users\benja\VT_P_index\model\variable_simulators\experimental.txt")
    globals().update(variables)

                  
                     
                        
     
     
   

    



        