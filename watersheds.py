# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:32:37 2020
Code for splitting up the Data 
@author: benja
"""

end_pointHUC12s=['041504081604', # Lake Champlain as a Whole
                 '041504081204', #Lake Champlain/St Albans Bay: St. A Bay /Northeast Arm
                 '041504080602', #Hoosington Brook/Lake Champlain: 'Otter Creek' Lake Segment
                 '041504080304', # McKenzie Brook-Lake Champlain: Roughly Lower Lake A/Port Henry  
                 '041504080902', #Malletts Bay
                 '041504080802' #shelburne Bay	]

def walk_HUC12s(ToHUC, gdf):
    '''Trace watersheds to where their water flows until you reach an endpoint.
    Return that endpoint'''
    if type(ToHUC)==int:
        return None
    elif ToHUC in end_pointHUC12s:
        return ToHUC 
    elif ToHUC in gdf['HUC12'].tolist():
        return walk_HUC12s(gdf[gdf['HUC12']==ToHUC]['ToHUC'].iloc[0], gdf)
    else:
        return ToHUC
    

endpoints=[walk_HUC12s(ToHUC, gdf) for ToHUC in gdf['HUC12'].tolist()]