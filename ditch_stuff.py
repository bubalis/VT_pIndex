# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:42:15 2020

@author: benja
"""


from whitebox import WhiteboxTools
import rasterio
import numpy as np
import os
import geopandas as gpd

main_dir=os.getcwd()
def ensure_dir():
    if os.getcwd()!=main_dir:
        os.chdir(main_dir)


#%%
ensure_dir()
wrk_dir=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')

scratch_dir=os.path.join(wrk_dir, 'scratch')
#os.makedirs(os.path.join(wrk_dir, 'scratch'))



wbt=WhiteboxTools()
ensure_dir()

dem=os.path.join(wrk_dir, '041504080401', "dem.tif")

for n in range(3, 18, 2):
    ensure_dir()
    output=os.path.join(scratch_dir, f'diff_{n}.tif') 
    wbt.diff_from_mean_elev(
    dem, 
    output, 
    filterx=n, 
    filtery=n, 
    )
    print(os.getcwd())


    with rasterio.open(output) as r:
        meta=r.meta.copy()
        data=r.read(1)
    
    meta.update({'dtype': np.int32})

    data=np.where(-9999<data, data, 9999)
    for thresh in np.linspace(-.5,-.1, 6):
        new_data=np.where(data<thresh, 1, 0)
    
    
        extract=os.path.join(scratch_dir, f'ditches_{n}_{thresh}.tif')
        with rasterio.open(extract, 'w', **meta) as dst:
            dst.write(np.array([new_data.astype(np.int32)]))
            
            
        wbt.raster_to_vector_polygons(
        extract, 
        os.path.join(scratch_dir, f'ditches_{n}_{thresh}.shp')
        
        )
        ensure_dir()
        

    #%%
watershed_path=os.path.join('Source_data', 'VT_Subwatershed_Boundaries_-_HUC12-shp', 'VT_Subwatershed_Boundaries_-_HUC12.shp')
aoi=gpd.read_file(watershed_path)
aoi=aoi[aoi['HUC12']=='041504080401']
crs=aoi.crs

stream_path=os.path.join('source_data', 'NHD_H_Vermont_State_Shape', 
                             "Shape", 'NHDFlowline.shp')
    
streams=gpd.read_file(stream_path)
streams.to_crs(crs, inplace=True)
streams=gpd.clip(streams, aoi)
streams['geometry']=streams.geometry.buffer(3)

#rivers:    
river_path=os.path.join( 'source_data', 'NHD_H_Vermont_State_Shape', 
                        "Shape", 'NHDArea.shp')
#ponds, lakes:
bodies_path=os.path.join('source_data', 'NHD_H_Vermont_State_Shape',
                         "Shape", 'NHDWaterbody.shp')
#other areas?


#add in all water elements
for path in [river_path, bodies_path]:
    new_gdf=gpd.read_file(path)
    new_gdf.to_crs(crs, inplace=True)
    new_gdf=gpd.clip(new_gdf, aoi)
    streams=gpd.overlay(streams, new_gdf, 'difference')
    #new_gdf.geometry=new_gdf.geometry.boundary
    
    streams=streams.append(new_gdf)
    
waterways=os.path.join(scratch_dir, 'waterways.shp')
#streams=streams.drop(columns=['index_right'])


streams.to_file(waterways)

    #%%
for file in [f for f in os.listdir(scratch_dir) if all([d in f for d in ['ditches', '.shp']])]:
    path=os.path.join(scratch_dir, file)
    gdf=gpd.read_file(path)
    gdf.crs=meta['crs']
    gdf.to_crs(aoi.crs)
    gdf.to_file(path)

def check_overlap(shp, compare_shp, directory):
    output=os.path.join(directory, 'scratch.shp')
    print(shp)
    print(compare_shp)
    print(output)
    wbt.clip(shp, compare_shp, output)
    
    overlap_area=get_area(output)
    shp_area=get_area(shp)
    compare_area=get_area(compare_shp)
    
    sensitivity=overlap_area/compare_area
    
    specificity=overlap_area/shp_area
    
    return {'sensitivity': sensitivity, 'specificity': specificity}

    
    
def get_area(shp_file):
    return gpd.read_file(shp_file).geometry.area.sum()



results={}
for shpfile in [f for f in os.listdir(scratch_dir) if '.shp' in f]:
    path=os.path.join(scratch_dir, shpfile)
    gpd.read_file(path).to_crs(crs).to_file(path)
    results[shpfile]=check_overlap(path, waterways, scratch_dir)


#%%
df=pd.DataFrame(results).T
df['weighted']=df['sensitivity']*df['specificity']
    