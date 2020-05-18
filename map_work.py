# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:48:44 2020

@author: benja
"""

import shapefile
import os

path=r"C:\Users\benja\VT_P_index\Source_data\soils\GeologicSoils_SO01\Geologic_SO01_poly."
files=[path+ext for ext in ['shp', 'shx', 'dbf']]
files
s=shapefile.Reader(*files)


def isClay(row):
    return 'clay' in row['MUNAME']