# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:49:15 2020

@author: benja


This script extracts all needed geospatial data from rasters.
Creates a crop-fields shapefile
"""

import fiona 
import geopandas as gpd
import os
import rasterio
from rasterstats import zonal_stats
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show as show_rast
from shapely.geometry.point import Point 
from rasterio.mask import mask

def get_cropFields():
    '''Load Crop Fields geodatabase'''
    gdb_file=os.path.join("P_Index_LandCoverCrops", "P_Index_LandCoverCrops","Crop_DomSoil.dbf")
    layers = fiona.listlayers(gdb_file)
    layer=layers[0]
    
    
    gdf = gpd.read_file(gdb_file,layer=layer)
    return gdf



    

#%%

def load_soils(soils_path):
    soils=gpd.read_file(soils_path)
    soils['null']=0
    aoi=soils.dissolve(by='null')[['geometry', 'AREASYMBOL']]
    return soils, aoi

def snip_to_aoi(gdf, aoi, buffer=0):
    '''Crop down a gdf to an area of interest.
    inputs: a gdf.
    an area of interest. '''
    crs=aoi.crs
    gdf.to_crs(crs, inplace=True)
    return gpd.overlay(aoi, gdf, how='intersection')



#%%

def point_from_raster_cell(y,x, raster):
    '''Return a point in the centroid of a given raster cell.
    Pass y and x: coordinates of that cell and the raster itself.
    Returns: a shapely Point obj'''
    top_left=(raster.bounds[3], raster.bounds[0])
    xform=raster.transform
    yloc=top_left[0]+(y+.5)*xform[4]
    xloc= (top_left[1]+(x+.5)*xform[0])
    try:
        return Point(xloc, yloc)
    except:
        print(xloc, yloc)
        assert False


def dist_to_water(field, raster, stream_line):
    
    '''Give the distance between the point w/ maximum flow accumulation and 
    the nearest stream.'''
    
    array, transform=mask(raster, [field['geometry']], crop=True)
    path=os.path.join('scratch', 'flow_acc.tif')
    
    out_meta = raster.meta
    out_meta.update({"driver": "GTiff",
                     "height": array.shape[1],
                     "width": array.shape[2],
                     "transform": transform, 
                     
                     })
        
        
    with rasterio.open(path, "w", **out_meta) as dest: #write the raster
        dest.write(array)
    with rasterio.open(path) as raster:   #read raster back in
        point=get_max_point(raster)
    point.crs=stream_line.crs
    dist= point.distance(stream_line.iloc[0]['geometry'])[0]
    if dist<0:
        return 0
    else:
        return dist, point
    
def dist_to_water_simple(field, stream_line):
    '''Shortest euclidean distance between field and any stream.'''
    
    distance= field['geometry'].distance(stream_line.iloc[0]['geometry'][0])
    if distance<4:
        return 4, None
    else:
        return distance, None
    
  #%%      

#%%
        
def all_dist_to_water(cfh2, streams, h2Osheds, usle_path):
    '''Find distance to water for all fields in the crop-fields gdf.'''
    
    results =[]
    points=[]
    for HUC12_code in cfh2['HUC12'].unique():
        raster_path=os.path.join(usle_path, HUC12_code, 'flow_acc.img')
        raster=rasterio.open(raster_path)
        
        #streams into a single geometric object for that watershed. 
        stream_line=streams[streams['HUC12']==HUC12_code].dissolve('HUC12')
        
        for i,field in cfh2[cfh2['HUC12']==HUC12_code].iterrows():
            dist, point=dist_to_water_simple(field, stream_line)
            
            #deleted complex method of calculating distance to H2O
            #dist, point=dist_to_water(field, raster, stream_line)
            results.append(dist)
            points.append(point)
    return results, points





def get_max_point(raster):
    '''Return a geoseries of a single point that is the centroid of the maximum value of a raster.'''
    
    a=raster.read(1)
    y, x=np.where(a==a.max())
    return gpd.GeoSeries(point_from_raster_cell(y[0],x[0], raster), crs=raster.crs)
    
    

def retrieve_rastervals(name, cf, usle_path, stat='mean'):
    '''Get zonal statistics for all shapes in a gdf. 
    Name: the name of the raster file. '''
    
    results=[]
    bad_HUC12s=[]
    for HUC12_code in cf['HUC12'].unique():
        raster_path=os.path.join(usle_path, HUC12_code, name)
        if not os.path.exists(raster_path):
            bad_HUC12s.append(HUC12_code)
            print(raster_path)
            print(cf[cf['HUC12']==HUC12_code])
            continue
        r=rasterio.open(raster_path)
        array=r.read(1)
        affine=r.transform
        stats=zonal_stats(cf[cf['HUC12']==HUC12_code], array, affine=affine, stats=[stat])
        results+=stats
        r.close()
    return [r[stat] for r in results]



def weighted_avg(df, avgcol, weight_col):
    '''Give weighted average of a column of a dataframe.
    df: a dataframe.
    avg_col: column to be averaged.
    weight_col: column to be weighted.
    Returns: float.'''
    df['weighted_totals']=df[avgcol]*df[weight_col]
    return df['weighted_totals'].sum()/df[weight_col].sum()

def weighted_majority(df, cat_col, weight_col):
    '''Return the category which is most prevalent in a series of a datatframe based on weights.
    df: a dataframe
    cat_col: the column with categories.
    weight_col: the column with weights.'''
    
    categories=df[cat_col].unique()
    cat_vals=[(cat, df[df[cat_col]==cat][weight_col].sum() )
              for cat in categories]
    return max(cat_vals, key=lambda x: x[1])[0]

def weighted_boolean_avg(df, avgcol, weight_col):
    '''Return the weighted avg of a column of the dataframe, where the column is a boolean.
    df: a dataframe.
    avgcol: a column of boolean values.
    weight_col: the column with weight values. '''
    return round(weighted_avg(df, avgcol, weight_col))


def set_globals():
    '''Setup globals for make these calculations.'''
    
    usle_path=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')
    soils_path=r"C:\Users\benja\VT_P_index\model\intermediate_data\Geologic_SO01_poly.shp"
    soils, aoi=load_soils(soils_path)
    
    crop_fields=get_cropFields()
    
    crs=soils.crs
    cf=snip_to_aoi(crop_fields, aoi)
    cf['IDNUM']=cf.index
    
    h2O_path=r"C:\Users\benja\VT_P_index\model\Source_data\VT_Subwatershed_Boundaries_-_HUC12-shp\VT_Subwatershed_Boundaries_-_HUC12.shp"
    
    h2Osheds=gpd.read_file(h2O_path)
    
    h2Osheds=snip_to_aoi(h2Osheds, aoi)

    return cf, soils, aoi, h2Osheds  , usle_path




def crop_fields_watersheds(cf, h2Osheds, streams):
    '''Create a geodatafraame whcih calculates all statistics for fields 
    which utilize watershed-level rasters.'''

    #prep the crop fields by watersheds gdf
    cfh2=gpd.overlay(h2Osheds, cf, how='intersection')
    cfh2['Area']=cf['geometry'].area
    cfh2.sort_values('HUC12', inplace=True)
    
    usle_path=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')
    
    
    
    #retreive raster stats for potential erosion, slope and elevation
    cfh2['RKLS']=retrieve_rastervals('RKLS.tif', cfh2, usle_path)
    cfh2['slope']=retrieve_rastervals('slope_raster.img', cfh2, usle_path)
    cfh2['elevation']=retrieve_rastervals('H2O_shed_DEM.img', cfh2, usle_path, stat='max')
    
    
    #calculate distance to water and erosion outflow points
    distances, outflow_points=all_dist_to_water(cfh2, streams, h2Osheds, usle_path)
    cfh2['distance_to_water']=distances
    cfh2['outflow_points']=outflow_points
    
    return  cfh2



    


def set_calculated_values(cf, cfh2):
    '''Extract values from overlays, and re-assign based on the appropriate measure of central tendency. 
    
    '''
    
    
    
    
    soils_overlay=gpd.overlay(soils, cf, how='intersection')
    soils_overlay['area']=soils_overlay['geometry'].area
    soils_overlay['hydro_group']=soils_overlay['HYDROGROUP'].apply(lambda x: x.split('/')[-1])
    cfh2['area']=cfh2['geometry'].area
    
    #set variables as empty lists
    rkls_values=[]
    clay_values=[]
    hydrogroup_values=[]
    elev_values=[]
    HUC12s=[]
    dist_to_water_values=[]
    
    
    for idnum in cf['IDNUM'].tolist():
        
        #extract values from the watershed-based gdf 
        subset_watershed=cfh2[cfh2['IDNUM']==idnum]
        rkls_values.append(weighted_avg(subset_watershed, 'RKLS', 'area'))
        HUC12s.append(subset_watershed['HUC12'].unique())
        dist_to_water_values.append(weighted_avg(subset_watershed, 'distance_to_water', 'area'))
        elev_values.append(weighted_majority(subset_watershed, 'elevation', 'area'))
        
        
        #soils overlay subset:
        subset_soils=soils_overlay[soils_overlay['IDNUM']==idnum]
        try:
            clay_values.append(weighted_boolean_avg(subset_soils,  'is_clay', 'area'))
        except KeyError:
            print(subset)
            print(idnum)
            assert False
        
        hydrogroup_values.append(weighted_majority(subset_soils, 'HYDROGROUP', 'area'))
        
        
        
    crop_codes_dict={2111: 'Corn', 2121: 'Hay', 2118: 'Small_Grain', 2124: 'Fallow'}    
        
    #assign values to original cropfields  gdf
    cf=cf[['IDNUM', 'CROP_COVER', 'geometry', ]]    
    cf['RKLS']=rkls_values
    cf['soil_is_clay']=clay_values
    cf['hydro_group']=hydrogroup_values
    cf['elevation']=np.array(elev_values)*3.28 #meters to feet
    cf['HUC12']=[str(tuple(H)) for H in HUC12s]
    cf['distance_to_water']=np.array(dist_to_water_values)*3.28 #meters to feet
    cf['crop_type']=cf['CROP_COVER'].apply(lambda x: crop_codes_dict[x])
    
    
    #assign the county, this will need a county dict later.
    cf['county']='Addison'
    
    
    save_path=os.path.join(os.getcwd(), 'intermediate_data', 'SO01_fields')
    cf=cf[[c for c in cf.columns if c!='geometry']+['geometry']] #reorder field names
    
    cf.to_file(save_path)
    
    #save column names to textfile to deal with shapefiel limitations. 
    with open(os.path.join('intermediate_data', 'SO01_fields', 'column_names.txt'), 'w') as file:
        for col_name in cf.columns.to_list():
            print(col_name, file=file)
    return cf


#to do vegetated buffer width 
#%%


def make_streams(h2Osheds, aoi):
    '''Make a shapefile of all bodies of water in the aoi.
    assumes streams are width= 0, while rivers/creeks are actually mapped in their boundaries. '''
    
    
    stream_path=os.path.join(os.getcwd(), 'source_data', 'NHD_H_Vermont_State_Shape', "Shape", 'NHDFlowline.shp')
    
    streams=gpd.read_file(stream_path)
    
    streams.to_crs(aoi.crs, inplace=True)
    streams=gpd.sjoin(streams, aoi)
        
    #rivers:    
    river_path=os.path.join(os.getcwd(), 'source_data', 'NHD_H_Vermont_State_Shape', "Shape", 'NHDArea.shp')
    #ponds, lakes:
    bodies_path=os.path.join(os.getcwd(), 'source_data', 'NHD_H_Vermont_State_Shape', "Shape", 'NHDWaterbody.shp')
    #areas?
    area_path=os.path.join(os.getcwd(), 'source_data', 'NHD_H_Vermont_State_Shape', "Shape", 'NHDArea.shp')    
    
    for path in [river_path, bodies_path, area_path]:
        new_gdf=gpd.read_file(path)
        new_gdf=snip_to_aoi(new_gdf, aoi)
        new_gdf['geometry']=new_gdf['geometry'].boundary
        streams=streams.append(new_gdf)
 
    save_path=os.path.join(os.getcwd(), 'intermediate_data', 'waterways.shp')
    streams=streams.drop(columns=['index_right'])
    
    
    streams=gpd.sjoin(streams, h2Osheds) #break up line segments by H2Oshed codes.
    streams.to_file(save_path)
    
    return streams
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
'''
#%%   

if __name__=='__main__':
    cf, soils, aoi, h2Osheds, usle_path  = set_globals()
    streams=make_streams(h2Osheds, aoi)    
    cfh2=crop_fields_watersheds(cf, h2Osheds, streams)
    #gpd.GeoDataFrame(geometry=[p[0] for p in cfh2['outflow_points'].tolist()], crs=cf.crs).to_file(os.path.join('intermediate_data', 'outflow_points'))
    cf=set_calculated_values(cf, cfh2)
    
    