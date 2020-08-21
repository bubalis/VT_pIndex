# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:33:42 2020

@author: benja
"""


from subprocess import call

for script in ['RKLS_by_watershed2', 'field_geodata', 'simulation']:
    print(script+'.py')
    call(['python', f'{script}.py'])
