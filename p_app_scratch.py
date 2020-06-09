# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:00:50 2020

@author: benja
"""
'''
manureTPincr=X98-(X16*$C$41*X97)-(X16*X93*X97*Manure_runoff_factor*X138*X99))*0.44/2*IF(X95=1,3,1)*IF(X$15<=10,0.2,MAX(0.03,1.1277*X$15^(-0.7639)))
Cell 100
'''

def manure_after_uptake(rate,total_p_added, crop_uptake):
    if total_p_added==0:
        total_p_added=1
        
    increment= rate-(crop_uptake/2*rate/total_p_added)
    
    if increment>0:
        return increment
    return 0


def manure_incr(self):
    '''Calculate P not eroded or taken up by crop from manure.'''
    total_p_added= sum([f for f in s])
    post_uptake=manure_after_uptake(rate,total_p_added, crop_uptake)
    loss1=(rate*.005*manure_factor)
    loss2=(rate*al_factor*manure_factor*runoff_factor*hyd_group_factor*avail_factor)
    incr=(post_uptake-loss1-loss2)
    if incr>0:
        return incr
    return 0
    
def manureSTPincr(incr, method_factor, al_level):
    '''Cell 100 in spreadsheet'''
    if method_factor==1:
        mf=3
    else:
        mf=1
    return  incr*0.44/2*mf*al_adj(al_level)
    

def manureTPincr(incr, method_factor):
    '''Cell 101 in spreadsheet'''
    if method_factor==1:
        mf=7.5
    else:
        mf=1
    return incr*.44/2*mf

def al_adj(al_level):
    if al_level<=10:
        return .2
    else:
        factor=1.277*al_level**(-0.7639)
    return max([factor, .03])
    

uptake_dict={'Corn & other row crops': 100,
 'Row crop + successful cover crop': 50,
 'Small grains': 50,
 'Alfalfa & other hay crops': 50,
 'Pasture': 0,
 'Vegetable crop - clean cultivated': 50,
 'Vegetable crop - mulch or living row cover': 50,
 'Vegetable crop - vining or high canopy': 50}

#%%
crops='''Corn & other row crops
Row crop + successful cover crop
Small grains
Alfalfa & other hay crops
Pasture
Vegetable crop - clean cultivated
Vegetable crop - mulch or living row cover
Vegetable crop - vining or high canopy'''.split('\n')

values='''100
50
50
50
0
50
50
50'''.split('\n')

values=[int(v) for v in values]

uptake_dict={k:v for k,v in zip(crops, values)}
