# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:44:44 2020

@author: benja
"""
import pygeoprocessing
import natcap.invest.sdr.sdr as sdr
import os



out_dir=os.path.join('intermediate_data', 'sample', 'USLE')
data_dir=os.path.join('model', 'source_data', 'sample')
    

if __name__=='__main__':
    
    dem_path=os.path.join(data_dir, 'DEM.tif')
    erosivity_path=os.path.join(data_dir, 'R_factor.tif')
    erodibility_path=os.path.join(data_dir, 'K_factor.tif')
    lulc_path=os.path.join(data_dir, 'lulc.tif')
    biophysical_table=os.path.join(data_dir, 'biophysical.csv')
    
    stream_path=os.path.join(out_dir, 'stream.tif')
    pit_filled_dem_path=os.path.join(out_dir, 'pit_filled_dem.tif')
    slope_path=os.path.join(out_dir, 'slope_path.tif')
    flow_direction_path=os.path.join(out_dir, 'flow_direction.tif')
    avg_aspect_path=os.path.join(out_dir, 'avg_aspect.tif')
    flow_accumulation_path=os.path.join(out_dir, 'flow_acc.tif')
    ls_factor_path=os.path.join(out_dir, 'ls.tif')
    rkls_path=os.path.join(out_dir, 'rkls.tif')
    cp_factor_path=os.path.join(out_dir, 'cp_factor.tif')
    out_usle_path=os.path.join(out_dir, 'ulse.tif')
    threshold_flow_accumulation=1
    aligned_drainage_path=False
    stream_and_drainage_path=stream_path
    
    pygeoprocessing.routing.fill_pits((dem_path, 1), pit_filled_dem_path)
    
    pygeoprocessing.calculate_slope,((pit_filled_dem_path, 1),
            slope_path)
    
    pygeoprocessing.routing.flow_dir_mfd(
            (pit_filled_dem_path, 1),
            flow_direction_path)
    
    pygeoprocessing.routing.flow_accumulation_mfd((flow_direction_path, 1),
            flow_accumulation_path)
    pygeoprocessing.routing.extract_streams_mfd(
            (flow_accumulation_path, 1),
            (flow_direction_path, 1),
            float(threshold_flow_accumulation),
            stream_path,)
    
    if aligned_drainage_path:
        sdr._add_drainage(
                    stream_path,
                    aligned_drainage_path,
                    stream_and_drainage_path)
    else:
        pass
        
    sdr.sdr_core.calculate_average_aspect(flow_direction_path,
                                      avg_aspect_path)
    
    
    sdr._calculate_ls_factor(
        flow_accumulation_path, slope_path, avg_aspect_path,
        ls_factor_path)
    
    sdr._calculate_rkls(ls_factor_path, erosivity_path, erodibility_path, stream_path,
        rkls_path)
    
    sdr._calculate_cp(biophysical_table, lulc_path, cp_factor_path)
    
    sdr._calculate_usle(rkls_path, cp_factor_path, stream_and_drainage_path, out_usle_path)
