# -*- coding: utf-8 -*-
"""
Created on Tue May 26 07:22:19 2020

@author: benja
"""

import geopandas as gpd
import os
import matplotlib.pyplot as plt
import natcap.invest



gdf=gpd.read_file(os.path.join('intermediate_data', 'Geologic_SO01_poly.shp'))
#%%
cmap=plt.cm.viridis
fig=plt.figure()
ax=plt.gca()
Ks=gdf.plot(column='K_factor', ax=ax, cmap=cmap, legend=True)
#fig.colorbar(Ks, ax=ax)
plt.legend(labels=['erosion factor'])
plt.savefig("test.png")
plt.show()
#%%