# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:04:32 2020

@author: benja
"""


import raster_download
import RKLS_by_watershed
import field_geodata
import simulation

def space():
    print('\n\n\n\n')
county_codes=[1, 7, 11, 15]

#print('Downloading DEM Files')
#space()
#raster_download.main(county_codes)
print("Calculating RKLS Values")
space()
RKLS_by_watershed.main(county_codes)
print('Merging Calculated Spatial Data')
space()
field_geodata.main(county_codes)
print('Running Simulation')
space()
simulation.main()