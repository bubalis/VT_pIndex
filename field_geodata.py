# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:49:15 2020

@author: benja
"""

import fiona 
import geopandas as gpd
import os
import rasterio
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from rasterio.plot import show as show_rast

def get_cropFields():
    '''Load Crop Fields geodatabase'''
    gdb_file=os.path.join("P_Index_LandCoverCrops", "P_Index_LandCoverCrops","Crop_DomSoil.dbf")
    layers = fiona.listlayers(gdb_file)
    layer=layers[0]
    
    
    gdf = gpd.read_file(gdb_file,layer=layer)
    return gdf

crop_fields=get_cropFields()
#%%

def load_soils(soils_path):
    soils=gpd.read_file(soils_path)
    soils['null']=0
    aoi=soils.dissolve(by='null')[['geometry', 'AREASYMBOL']]
    return soils, aoi

def snip_to_aoi(gdf, aoi):
    '''Crop down a shapefile to an area of interest.'''
    crs=aoi.crs
    gdf.to_crs(crs, inplace=True)
    return gpd.overlay(aoi, gdf, how='intersection')


soils_path=r"C:\Users\benja\VT_P_index\model\intermediate_data\Geologic_SO01_poly.shp"

soils, aoi=load_soils(soils_path)
crs=soils.crs
cf=snip_to_aoi(crop_fields, aoi)
cf['IDNUM']=cf.index

h2O_path=r"C:\Users\benja\VT_P_index\model\Source_data\VT_Subwatershed_Boundaries_-_HUC12-shp\VT_Subwatershed_Boundaries_-_HUC12.shp"

h2Osheds=gpd.read_file(h2O_path)

h2Osheds=snip_to_aoi(h2Osheds, aoi)


cfh2=gpd.overlay(h2Osheds, cf, how='intersection')
cfh2['Area']=cf['geometry'].area
#%%
cfh2.sort_values('HUC12', inplace=True)


usle_path=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')

#%%
def retrieve_rastervals(name, cf):
    results=[]
    bad_HUC12s=[]
    for HUC12_code in cf['HUC12'].unique():
        raster_path=os.path.join(usle_path, HUC12_code, name)
        if not os.path.exists(raster_path):
            bad_HUC12s.append(HUC12_code)
            #print(crop_fields2[crop_fields2['HUC12']==HUC12_code])
            continue
        r=rasterio.open(raster_path)
        array=r.read(1)
        affine=r.transform
        stats=zonal_stats(cf[cf['HUC12']==HUC12_code], array, affine=affine, stats=['mean', 'median', 'max'])
        results+=stats
        r.close()
    return [r['mean'] for r in results]
#%%  


for fact in ['K_Factors.tif', 'LS.tif', "RKLS.tif", 'flow_acc.img' ]:
    cfh2[fact.split('.')[0]]=retrieve_rastervals(fact, cfh2)

cfh2['slope']=retrieve_rastervals('slope_raster.img', cfh2)

#%%





#%%






#%%



#%%



#merge 
def weighted_avg(df, avgcol, weight_col):
    df['weighted_totals']=df[avgcol]*df[weight_col]
    return df['weighted_totals'].sum()/df[weight_col].sum()

def weighted_majority(df, cat_col, weight_col):
    categories=df[cat_col].unique()
    cat_vals=[(cat, df[df[cat_col]==cat][weight_col].sum() )
              for cat in categories]
    return max(cat_vals, key=lambda x: x[1])[0]

def weighted_boolean_avg(df, avgcol, weight_col):
    
    return round(weighted_avg(df, avgcol, weight_col))

rkls_values=[]
clay_values=[]
hydrogroup_values=[]
HUC12s=[]

soils_overlay=gpd.overlay(soils, cf, how='intersection')
cfh2['area']=cfh2['geometry'].area

for idnum in cf['IDNUM'].tolist():
    subset=cfh2[cfh2['IDNUM']==idnum]
    rkls_values.append(weighted_avg(subset, 'RKLS', 'area'))
    HUC12s.append(subset['HUC12'].unique())
    
    subset=soils_overlay[soils_overlay['IDNUM']==idnum]
    clay_values.append(weighted_boolean_avg(subset,  'is_clay', 'area'))
    hydrogroup_values.append(weighted_majority(subset, 'HYDROGROUP', 'area'))
    

cf=cf[['IDNUM', 'CROP_COVER', 'geometry', ]]    
cf['rkls']=rkls_values
cf['is_clay']=clay_values
cf['hydrogroup']=hydrogroup_values
cf['HUC12']=HUC12s

save_path=os.path.join(os.getcwd(), 'intermediate_data', 'SO01_fields.shp')
cf.to_file(save_path)



#to do: model distance to  water. 
#
#also possible: vegetated buffer width (harder)
#%%

#extract streams :
#assumes streams are width= 0, while rivers/creeks are actually mapped in their boundaries. 

stream_path=os.path.join(os.getcwd(), 'source_data', 'NHD_H_Vermont_State_Shape', "Shape", 'NHDFlowline.shp')

streams=gpd.read_file(stream_path)

streams.to_crs(aoi.crs, inplace=True)
streams=gpd.sjoin(streams, aoi)

#%%
#make boundaries of larger waterways:

    

river_path=os.path.join(os.getcwd(), 'source_data', 'NHD_H_Vermont_State_Shape', "Shape", 'NHDArea.shp')

rivers=gpd.read_file(river_path)

rivers=snip_to_aoi(water, aoi)

rivers['geometry']=water['geometry'].boundary

bodies_path=os.path.join(os.getcwd(), 'source_data', 'NHD_H_Vermont_State_Shape', "Shape", 'NHDWaterbody.shp')

bodies=gpd.read_file(bodies_path)

bodies=snip_to_aoi(bodies, aoi)

bodies['geometry']=bodies['geometry'].boundary


streams=streams.append(rivers)
streams=streams.append(bodies)

save_path=os.path.join(os.getcwd(), 'intermediate_data', 'waterways.shp')

streams.to_file(save_path)
#%%


'''Code for exploring USLE results with rasters in Google Earth.'''
'''

import fiona


highest=cf.sort_values('RKLS').dropna(subset=['RKLS']).tail(10)

fiona.supported_drivers['KML'] = 'rw'
highest['geometry'].to_file('maxima.kml', driver='KML')

highest['buf']=highest.buffer(100)
i2=0
for i, row in highest.iterrows():
    print(i)
    code=row['TNMID']
    if not os.path.exists(str(i)):
        os.makedirs(str(i))
    shape=[row['buf']]
    for data in ['K_Factors.tif', 'LS.tif', "RKLS.tif", 'flow_acc.img', 'pit_filled_dem.img', 'slope_raster.img' ]:
        name=data.split('.')[0]    
        path=os.path.join(usle_path, row['HUC12'], data)    
        with rasterio.open(path) as s:
            out_image, out_transform = mask(s, shape, crop=True)
            out_meta = s.meta
            
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform, 
                             
                             })
        out_path=os.path.join(os.getcwd(), str(i), f"masked_{name}_{i}.img")
        
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)
        
        
        with rasterio.open(out_path, 'r') as r1:
            
            show_rast(r1, alpha=.7, 
                      contour=False,
                      title=f'{name}_{i}'
                      )
            plt.plot(row['geometry'], color='k', alpha=.5, ax=ax, )
            plt.show()
    i+=1
    
    
ax=mh.plot()
m.plot(color='r', ax=ax)
#%%
'''
def CDF(X, Xname,**kwargs):
    '''Plot the cumulative distribution function of a variable.'''
    if type(X)!=np.array:
        X=np.array(X)
    X=X.flatten()
    X.sort()
    N=X.size
    Y=[i/N for i in range(N)]
    fig= plt.plot(X, Y, '.-', **kwargs)
    plt.xlabel(Xname)
    plt.ylabel("$P_<(x)$")
    return fig, X, Y



