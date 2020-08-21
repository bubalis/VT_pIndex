# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 08:01:15 2020

@author: benja
"""
import math
import numpy as np

def LS_factor(slope_length, slope_angle):
    slope_angle=np.radians(slope_angle)
    slope_grad=np.tan(slope_angle)
    return S_factor(slope_angle, slope_grad)*L_factor(slope_length, slope_grad)
    
def S_factor(slope_angle, slope_grad):
    if slope_grad<.09:
        return 10.8*np.sin(slope_angle)
    else:
        return 16.8*np.sin(slope_angle)
    
def L_factor(slope_length, slope_grad):
    if slope_grad<.01:
        m=.2
    elif slope_grad>=.01 and slope_grad<.03:
        m=.3
    elif slope_grad>=.03 and slope_grad<.4:
        m=.4
    else:
        m=.5
    
    return np.power(slope_length/22.13, m)