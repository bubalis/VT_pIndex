# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:49:15 2020

@author: benja


This script extracts all needed geospatial data from rasters.
Creates a crop-fields shapefile.
"""

import fiona 
import geopandas as gpd
import os
import rasterio
import re
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt

from rasterio.plot import show as show_rast
from shapely.geometry.point import Point 
from rasterio.mask import mask
from rasterstats import zonal_stats
from raster_download import load_counties



def get_cropFields():
    '''Load Crop Fields geodatabase'''
    gdb_file=os.path.join("P_Index_LandCoverCrops", "P_Index_LandCoverCrops","Crop_DomSoil.dbf")
    layers = fiona.listlayers(gdb_file)
    layer=layers[0]
    
    
    gdf = gpd.read_file(gdb_file,layer=layer)
    return gdf


def save_shape_w_cols(gdf, dir_path):
    '''Save a gdf as a shapefile, with an additional .txt 
    file delineating column names (which otherwise would be truncated)'''
    
    gdf.to_file(dir_path)
    with open(os.path.join(dir_path, 'column_names.txt'), 'w') as file:
        for col_name in gdf.columns.to_list():
            print(col_name, file=file)

def load_shape_w_cols(dir_path):
    '''Load in a shapefile as a gdf, assigning the column names based on a text file. '''
    shape_file=[f for f in os.listdir(dir_path) if 'shp' in f][0]
    path=os.path.join(dir_path, shape_file)
    gdf=gpd.read_file(path)
    
    #rename columns to full length
    with open(os.path.join(dir_path, 'column_names.txt')) as f:
        gdf.columns=[line for line in f.read().split('\n') if line]
    return gdf

#%%
def load_crop_rotation_codes(crop_rot_path, rot_code_path):
    '''Load crop rotation code data.'''
    codes=gpd.read_file(rot_code_path)[['Code', 'Descriptio']]
    codes.rename(columns={'Code': 'Rotation'}, inplace=True)
    crop_rots=gpd.read_file(crop_rot_path)[['MUSYM', 'Rotation']]
    crop_rots=crop_rots.merge(codes, on='Rotation')
    return crop_rots



def load_soils(soils_path, aoi):
    '''Load the soils gdb and create the null aoi gdb.'''
    print('Making soils shapes and aoi.')
    soils=gpd.read_file(soils_path)
    soils=snip_to_aoi(soils, aoi, dissolve_feature='CNTY')
    #soils['is_clay']=(soils['MUID'].str.contains('clay')).astype(int)
    
    # extract likely rotation data
    crop_rot_path=os.path.join('P_Index_LandCoverCrops', 'P_Index_LandCoverCrops', 'CropRotation_SoilType.dbf' )
    rot_code_path=os.path.join('P_Index_LandCoverCrops', 'P_Index_LandCoverCrops', "CropRotationCodes_Domain.dbf")
    crop_rots=load_crop_rotation_codes(crop_rot_path, rot_code_path)
    
    #and add to soils
    soils=soils.merge(crop_rots, on='MUSYM')
    soils.drop(columns=['Rotation'], inplace=True)
    soils.rename(columns={'Descriptio': 'Rotation'}, inplace=True)
    
    return soils

def snip_to_aoi(gdf, aoi, dissolve_feature=None, buffer=0):
    '''Crop down a gdf to an area of interest.
    inputs: a gdf.
    an area of interest. '''
    crs=aoi.crs
    gdf.to_crs(crs, inplace=True)
    gdf['id']=gdf.index
    if dissolve_feature:
        aoi=aoi.dissolve(by=dissolve_feature)
    overlay= gpd.overlay(aoi, gdf, how='intersection')
    
    
    return gdf[gdf['id'].isin(overlay['id'].tolist())]
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
    '''Shortest euclidean distance between field and any water-body.'''
    
    distance= field['geometry'].distance(stream_line.iloc[0]['geometry'])
    if distance<4:
        return 4, None
    else:
        return distance, None
    
  #%%      

#%%
        
def all_dist_to_water(crop_wshed_ovlry, streams, h2Osheds, usle_path):
    '''Find distance to water for all fields in the crop-fields gdf.'''
    print('Calculating Distance to Water for fields.')
    results =[]
    points=[]
    for HUC12_code in crop_wshed_ovlry['HUC12'].unique():
        #raster_path=os.path.join(usle_path, HUC12_code, 'flow_acc.img')
        #raster=rasterio.open(raster_path)
        
        #streams into a single geometric object for that watershed. 
        stream_line=streams[streams['HUC12']==HUC12_code].dissolve('HUC12')
        
        for i,field in crop_wshed_ovlry[crop_wshed_ovlry['HUC12']==HUC12_code].iterrows():
            dist, point=dist_to_water_simple(field, stream_line)
            
            #deleted complex method of calculating distance to H2O
            #dist, point=dist_to_water(field, raster, stream_line)
            results.append(dist)
            points.append(point)
    return results, points





def get_max_point(raster):
    '''Return a geoseries of a single point that is the centroid of the maximum value of a raster.
    Used to calculate distance from outflow point to water. 
    '''
    
    a=raster.read(1)
    y, x=np.where(a==a.max())
    return gpd.GeoSeries(point_from_raster_cell(y[0],x[0], raster), crs=raster.crs)
    
    

def retrieve_rastervals(raster_file_name, crop_fields, usle_path, stat='mean'):
    '''Get zonal statistics for all shapes in a gdf. 
    '''
    
    results=[]
    bad_HUC12s=[]
    for HUC12_code in crop_fields['HUC12'].unique():
        raster_path=os.path.join(usle_path, HUC12_code, raster_file_name)
        if not os.path.exists(raster_path):
            bad_HUC12s.append(HUC12_code)
            print(raster_path)
            print(crop_fields[crop_fields['HUC12']==HUC12_code])
            continue
        results+=extract_zonal_stats(raster_path, crop_fields[crop_fields['HUC12']==HUC12_code], stat)
        
    
    if bad_HUC12s:
        print ('Bad HUC12 codes: ')
        print(bad_HUC12s)
    return [r[stat] for r in results]


def extract_zonal_stats(raster_path, gdf, stat):
    '''Extract zonal stats from a raster path.'''
    
    with rasterio.open(raster_path) as r:
         array=r.read(1)
         affine=r.transform
         return zonal_stats(gdf, array, affine=affine, stats=[stat])

def weighted_avg(df, avgcol, weight_col):
    '''Give weighted average of a column of a dataframe.
    df: a dataframe.
    avg_col: column to be averaged.
    weight_col: column to be weighted.
    Returns: float.'''
    df['weighted_totals']=df[avgcol]*df[weight_col]
    return df['weighted_totals'].sum()/df[weight_col].sum()

def weighted_mode(df, cat_col, weight_col):
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

def all_unique_values(df, valcol, *args):
    return df[valcol].unique()



def set_globals(county_codes):
    '''Setup globals for make geodata calculations.'''
    aoi=load_counties(county_codes)
    crs=aoi.crs
    usle_path=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')
    soils_path=os.path.join("Source_data", "GeologicSoils_SO", "GeologicSoils_SO.shp")
    
    soils=load_soils(soils_path, aoi)                        
    crop_fields=get_cropFields()
    
    
    crop_fields=snip_to_aoi(crop_fields, aoi, dissolve_feature='CNTY')
    crop_fields['IDNUM']=crop_fields.index
    
    h2O_path=r"C:\Users\benja\VT_P_index\model\Source_data\VT_Subwatershed_Boundaries_-_HUC12-shp\VT_Subwatershed_Boundaries_-_HUC12.shp"
    
    h2Osheds=gpd.read_file(h2O_path)
    
    h2Osheds=snip_to_aoi(h2Osheds, aoi, dissolve_feature='CNTY')

    return crop_fields, soils, aoi, h2Osheds  , usle_path




def crop_fields_watersheds(crop_fields, h2Osheds, streams):
    '''Create a geodatafraame which calculates all statistics for fields 
    which utilize watershed-level rasters.
    distance to water and potential erosion (RKLS) 
    are calculated on watershed level (for memory useage.)
    They have to be calculated and recombined for fields which straddle watersheds.
    '''
    print('Making calulations on Watershed Level')
    #prep the crop fields by watersheds gdf
    crop_wshed_ovlry=gpd.overlay(h2Osheds, crop_fields, how='intersection')
    crop_wshed_ovlry['Area']=crop_fields['geometry'].area
    crop_wshed_ovlry.sort_values('HUC12', inplace=True)
    
    usle_path=os.path.join(os.getcwd(), 'intermediate_data', 'USLE')
    
    
    
    #retreive raster stats for potential erosion, slope and elevation
    crop_wshed_ovlry['RKLS']=retrieve_rastervals('RKLS.tif', crop_wshed_ovlry, usle_path)
    crop_wshed_ovlry['slope']=retrieve_rastervals('slope_raster.img', crop_wshed_ovlry, usle_path)
    crop_wshed_ovlry['elevation']=retrieve_rastervals('H2O_shed_DEM.tif', crop_wshed_ovlry, usle_path, stat='max')
    
    
    #calculate distance to water and erosion outflow points
    distances, outflow_points=all_dist_to_water(crop_wshed_ovlry, streams, h2Osheds, usle_path)
    crop_wshed_ovlry['distance_to_water']=distances
    crop_wshed_ovlry['outflow_points']=outflow_points
    
    return  crop_wshed_ovlry
#%%

def parse_crop_rots(string):
    '''Parser for crop rotations.
    May need work. '''
    n_corn, n_hay, n_other, n_fallow=0,0,0,0
    if '/' in string:
        try:
            n_corn, n_hay=tuple([int(re.search('\d', part).group()) for part in string.split('/')])
        except:
            print(string.split('/'))
    elif 'Continuous' in string:
        if 'Corn' in string:
            n_corn=10 
        elif "Hay" in string:
            n_hay=10
    elif string=='Not Suited To Crops':
        n_fallow=10
    elif string=='HC':
        n_hay=10
    elif string=='None':
        n_fallow=10
    
    else:
        print(string)
        raise ValueError
        
    return Series([n_corn, n_hay, n_other, n_fallow])
        
   #%%     
    

class Column_Summary_by_Area():
    '''Object to aggregate values from a column. 
    Used for aggregating subsets while weighting by area and putting columns back together.'''
    
    def __init__(self, column_name, summary_function, null_value='None'):
        self.column_name=column_name
        self.summary_function=summary_function
        self.results=[]
        self.null_value=null_value
        
    def calculate(self, df):
        '''Run the function on a df, weighted by area.'''
        try:
            return self.summary_function(df, self.column_name, 'area')
        except ValueError:
            print(f'aggregator of {self.column_name} failed')
            return self.null_value
        
    def calc_and_append(self, df):
        self.results.append(self.calculate(df))

def set_calculated_values(crop_fields, crop_wshed_ovlry, soils, aoi):
    '''Extract values from overlays, and re-assign based on the appropriate measure of central tendency. 
    
    '''
    
    print('Assigning values to geo-dataframe')
    
    county_overlay=gpd.overlay(aoi, crop_fields, how='intersection')
    county_overlay['area']=county_overlay['geometry'].area
    
    soils_overlay=gpd.overlay(soils, crop_fields, how='intersection')
    soils_overlay['area']=soils_overlay['geometry'].area
    soils_overlay['hydro_group']=soils_overlay['HYDROGROUP'].apply(lambda x: x.split('/')[-1])
    
    crop_wshed_ovlry['area']=crop_wshed_ovlry['geometry'].area
    
    
    H2O_col_aggregators=[Column_Summary_by_Area(col_name, func_name) 
                         for col_name, func_name in 
                         [('RKLS', weighted_avg),
                          ('HUC12', all_unique_values),
                          ('distance_to_water', weighted_avg),
                          ('elevation', weighted_avg)]
                         ]

    soil_col_aggregators=[Column_Summary_by_Area(col_name, func_name) 
                          for col_name, func_name in 
                          [('is_clay', weighted_boolean_avg),
                           ('hydro_group', weighted_mode),
                           ('Rotation', weighted_mode)]
                          ]
    county_col_aggregator=Column_Summary_by_Area('CNTY', weighted_mode)
    
   
    for idnum in crop_fields['IDNUM'].tolist():
        #extract values from the watershed-based gdf 
        subset_watershed=crop_wshed_ovlry[crop_wshed_ovlry['IDNUM']==idnum]
        for agg in H2O_col_aggregators:
            agg.calc_and_append(subset_watershed)
        
        #soils overlay subset extractions:
        subset_soils=soils_overlay[soils_overlay['IDNUM']==idnum]
        for agg in soil_col_aggregators:
            agg.calc_and_append(subset_soils)
    for idnum in crop_fields['IDNUM'].tolist():     
        subset_county=county_overlay[county_overlay['IDNUM']==idnum]
        county_col_aggregator.calc_and_append(subset_county)
        
    #assign values to original cropfields  gdf
    crop_fields=crop_fields[['IDNUM', 'CROP_COVER', 'geometry', ]]
    aggregators=H2O_col_aggregators+ soil_col_aggregators+ [county_col_aggregator]
    for agg in aggregators:
        crop_fields[agg.column_name]=agg.results
    print(crop_fields.columns)
    
    #clean and reassign variables
    crop_fields.rename(columns={'is_clay':'soil_is_clay','HYDROGROUP':'hydro_group', 'CNTY': 'county'}, inplace=True)
    print(crop_fields.columns)
    for dist_col in ['elevation', 'distance_to_water']:
        crop_fields[dist_col]=crop_fields[dist_col]*3.28 #meters to feet
    
    
    crop_codes_dict={2111: 'Corn', 2121: 'Hay', 2118: 'Small_Grain', 2124: 'Fallow'}    
    crop_fields['crop_type']=crop_fields['CROP_COVER'].apply(lambda x: crop_codes_dict[x])
    crop_fields['HUC12']=crop_fields['HUC12'].apply(lambda x: str(tuple(x)))
    
    rotation_df=crop_fields['Rotation'].apply(parse_crop_rots)
    rotation_df.columns=['Years_Corn', "Years_Hay", 'Years_Other', "Years_Fallow"]
    crop_fields=crop_fields.merge(rotation_df, left_index=True, right_index=True)
    
    
    #assign the county, this will need a county dict later.
    
    county_num_dict={1:'Addison',
                 7: 'Chittenden',
                 21: 'Rutland', 
                 11: "Franklin",
                 15: 'Lamoille',
                 13: 'Grand Isle'}
    
    crop_fields['county']=crop_fields['county'].apply(lambda x: county_num_dict.get(x))
    
    
    
    crop_fields=crop_fields[[c for c in crop_fields.columns if c!='geometry']+['geometry']] #reorder field names
    
    
    save_path=os.path.join(os.getcwd(), 'intermediate_data', 'aoi_fields')
    save_shape_w_cols(crop_fields, save_path)
    
    return crop_fields


#to do: vegetated buffer width 
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
    #other areas?
    area_path=os.path.join(os.getcwd(), 'source_data', 'NHD_H_Vermont_State_Shape', "Shape", 'NHDArea.shp')    
    
    #add in all water elements
    for path in [river_path, bodies_path, area_path]:
        new_gdf=gpd.read_file(path)
        new_gdf=snip_to_aoi(new_gdf, aoi, dissolve_feature='CNTY')
        new_gdf['geometry']=new_gdf['geometry'].boundary
        streams=streams.append(new_gdf)
 
    save_path=os.path.join(os.getcwd(), 'intermediate_data', 'waterways.shp')
    streams=streams.drop(columns=['index_right'])
    
    
    streams=gpd.sjoin(streams, h2Osheds) #break up line segments by H2Oshed codes.
    streams.to_file(save_path)
    
    return streams


def plot_globals(crop_fields, soils, aoi, h2Osheds):
    print("Plotting shapes....")
    ax=aoi.plot()
    ax=crop_fields.plot(ax=ax, color='r', alpha=.9)
    ax=soils.plot(ax=ax, color='g', alpha=.5)
    h2Osheds.plot(ax=ax, color='grey', alpha=.5)
    plt.show()
#%%   
def main(county_codes):
    crop_fields, soils, aoi, h2Osheds, usle_path  = set_globals(county_codes)
    plot_globals(crop_fields, soils, aoi, h2Osheds)
    streams=make_streams(h2Osheds, aoi)    
    crop_wshed_ovlry=crop_fields_watersheds(crop_fields, h2Osheds, streams)
    #gpd.GeoDataFrame(geometry=[p[0] for p in crop_wshed_ovlry['outflow_points'].tolist()], crs=crop_fields.crs).to_file(os.path.join('intermediate_data', 'outflow_points'))
    crop_fields=set_calculated_values(crop_fields, crop_wshed_ovlry, soils, aoi)
    crop_fields['distance_to_water'].hist()
    plt.show()

if __name__=='__main__':
    county_codes=[1, 7, 11, 15]
    main(county_codes)
    
#%%

