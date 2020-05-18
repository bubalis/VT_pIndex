# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:46:24 2020

@author: benja
"""

from helper_funcs import lookupBRV, lookup_RAF, get_soil_hyd_factor, manure_factor
import numpy as np

class CropField():
    def calcPindex(self):
        self.results['surface particulate loss']= self.surf_particulate_loss()
        self.results['surface dissolved loss']= self.surf_dissolved_loss()
        self.results['subsurface_loss']=self.subsurface_loss()
                
        
        self.results['total p loss']=sum(self.results[pathway] for pathway in 
                    ['surface particulate loss', 'surface dissolved loss', 'subsurface_loss'])
        
        print(self.results)
    
    
    def sim_parameters(self):
        dic={}
        dic['runoff_adj_factor']=lookup_RAF(self.hydro_group, self.veg_type,  self.cover_perc)
    
        return dic
    
    def sum_fractions(self, functions_to_sum, default_args={}):
        '''Return the sum of multiple functions, with all of the field's parameters passed to that function.'''
        kwargs={**self.params,**default_args}
        s=0
        for function in functions_to_sum:
            result=function(**kwargs)
            self.results[function.__name__]=result
            s+=result
        return s
    
    def surf_particulate_loss(self, scaling_factor=80):
        '''Surface Particulate P loss'''
        return self.sum_fractions([erodedSoilP, manure_partic_P]) *scaling_factor
   
    
    def surf_dissolved_loss(self, scaling_factor=80):
        '''Surface Dissolved P loss'''
        return self.sum_fractions([dis_soilP, dis_manureP, fertilizerP,])*scaling_factor
   
    
    def subsurface_loss(self, scaling_factor=80):
        '''Subsurface loss'''
        if self.params.get('tile_drain', None):
            return self.sum_fractions([erodedSoilP, 
                                       manure_partic_P,
                                       dis_soilP,
                                       dis_manureP, 
                                       fertilizerP], 
                                    {'SDR_factor':1, 'RDR_factor':1})*.2*scaling_factor
        else:
            return 0
        
    def link_to_apps(self):
        '''Set linkages between p applications and field object.'''
        for app in self.manure_applications+self.fertilizer_applications:
            app.link_to_field(self)

    
def CropFieldFirstRun(CropField):
    '''Initate class for a field for the first time.'''
    def __init__(self):
        self.get_parameters()
    
    def get_parameters(self):
        self.known_params={}
        self.known_params['baseROV']=lookupBRV(**self.known_params)
   
    
    
    
class CropFieldFromSave(CropField):
    '''Initiate a class from saved data.
    Pass a dictionary of known_parameters'''
    def __init__(self, key, data):
        self.known_params=data[key]
        
        
    def sim(self):
        self.sim_params=self.sim_parameters()
        self.params={**self.known_params, **self.sim_params}
        
    def calculate(self):
        
    

class CropFieldFromDic(CropField):
    def __init__(self, dic):
        self.params=dic
        self.params['soil_total_phos']=calc_soilTP(self.params['soil_test_phos'], self.params['soil_is_clay'])
        self.params['runoff_adj_factor']=lookup_RAF(self.params['hydro_group'], self.params['veg_type'], self.params['cover_perc'])
        self.params["baseROV"]=lookupBRV(self.params['county'], self.params['elevation'])
        self.manure_applications=self.params['manure_applications']
        self.fertilizer_applications=self.params['fertilizer_applications']
        self.params['hydro_factor']=get_soil_hyd_factor(self.params['hydro_group'], self.params['tile_drain'])
        self.params['RDR_factor']=calcRDR(**dic)
        self.results={}
        self.link_to_apps()
        
        
def calc_soilTP(test_phos, soil_is_clay):
    '''Calculated total phosphorus from test_phos. 
    From documentation page 6'''
    if soil_is_clay:
        return 6.56*test_phos+650
    else:
        return 10.87*test_phos+760
    
        
        
    
def erodedSoilP(erosion_rate, soil_test_phos, soil_total_phos,  **kwargs):
    '''Return the value for eroded soil Phosphorus. 
    Page 6 of technical docs:  
    1. Sediment (eroded soil) P loss = E * TP * TP Availability * SDR
The four terms are:
a. E = Annual soil loss. The RUSLE (ver. 1 or 2) or WEPP edge-of-field erosion rate in
tons/ac is divided by 500 to convert it to million lb/ac. Annual rather than rotation erosion
value should be used.

It is assumed that the soil test was made before any manure or fertilizer applications.
Therefore, the added P (after subtracting crop uptake, and dissolved and particulate P losses)
is used to adjust soil test P by the equation, ∆STP/added P = 1.277 * Al0.7639 (see
Aluminum factor above). This adjusted STP is then plugged into the appropriate regression
equation for Total P.

A limit on this TP estimate is made by adding the applied manure and fertilizer P (minus
losses, as above), which represents added TP, to the TP calculated by regression from the
original soil test P. The estimate cannot be greater than this limiting value.
c. TP availability factor. Research suggests that only a fraction of the total P in soils is
available for the growth of algae. This factor ranges from 0.2 (i.e., 20% of TP is algal
available) at a soil test P of 0 ppm, to a maximum of 0.4 at STP = 100 ppm (based on a
chemical extraction of Lake Champlain sediments that approximates algal uptake).
d. SDR = Sediment Delivery Ratio (see Sediment and Runoff Delivery Ratios, above). For
the case of sediment, if a sediment control structure is entered into the P Index, its factor 
May 24, 2017 Vermont-P-Index-User-Guide with logos and statements.doc 7
(with a range of 0 to 0.2) is used instead of the SDR (range 0.4 to 1.0). In addition, if the
total distance to the nearest water body is greater than the buffer width, the additional
distance beyond the buffer is considered to have some effect on sediment load. The Distance
Factor is calculated for this additional distance, and is multiplied by the Buffer Factor for the
final SDR    '''
    return (erosion_rate/500)*soil_total_phos*P_avail(soil_test_phos)*SDRsed(**kwargs)*.44





def P_avail(soil_test_phos):
    '''TP availability factor. From Page 6:
    This factor ranges from 0.2 (i.e., 20% of TP is algal available) 
at a soil test P of 0 ppm, to a maximum of 0.4 at STP = 100 ppm
        '''
    if soil_test_phos<100:
        return .2+soil_test_phos/100*.2
    else:
        return .4
    

def manure_partic_P(manure_applications, **kwargs):
    '''Phosphorus lost from manure particulate P
    Page 7 of Technical docs.'''
    s=0
    sdr=SDRm(**kwargs)
    for m_app in manure_applications:
        s+=m_app.calcParticPloss(sdr)
        m_app.sdr=sdr
    return s
        

def dis_soilP(soil_test_phos, baseROV, runoff_adj_factor, **kwargs):
    '''Calculate dissolved P loss.   '''
    return DRPexcel(soil_test_phos)*baseROV*runoff_adj_factor
    
def DRP(soil_test_phos):
    '''Calculate Dissolved Reactive Phosphorus:
        From VTPI Tech Docs:
           Dissolved reactive P (DRP) concentration in runoff, expressed in parts per million.
Research involving simulated rainfall applied to field plots on a wide variety of Vermont
agricultural soils has provided a good relationship between soil test P (STP) and DRP
concentration in runoff: DRP = 0.1275 + 0.0104 * STP (see Figure 4). Soil test P is first
adjusted for any increment due to manure or fertilizer P added since the soil test was made''' 
    return .1275+(.0104*soil_test_phos)

def DRPexcel(soil_test_phos):
    '''Dissolved Reactive Phosphorus Based on #s in the excel model'''
    return 2*(.00705*soil_test_phos+.03)
    
    
def dis_manureP(manure_applications, **kwargs):
    '''Calculate sum of dissolved Manure Applications'''
    s=0
    for m_app in manure_applications:
        s+=m_app.calcDisPloss()
    return s
    



    
def fertilizerP(fertilizer_applications, **kwargs):
    '''Calculate sum of dissolved P from fertilizer applications'''
    s=0
    for f_app in fertilizer_applications:
        s+=f_app.calcDissolved_P_loss()
    return s
    

def SDR(distance, Buffer=True):
    '''Sediment and Runoff Delivery Ratios Page 2 of the documentation.
    distance: width of buffer or distance to water (in feet).
    Buffer: Boolean: are we calculating Buffer factor?'''
    if Buffer:
        return 1.744*np.exp((-43-distance)/45)+.4
    else: 
        return 1.047*np.exp((-70-distance)/60)+.7
    
    
def SDRm(distance_to_water, buffer_width, **kwargs):
    if 'SDR_factor' in kwargs.keys():
        return kwargs['SDR_factor']
    else:
        return SDR(buffer_width, Buffer=False)

def calcRDR(buffer_width, **kwargs):
    if 'RDR_factor' in kwargs.keys():
        return kwargs['RDR_factor']
    else:
        return SDR(buffer_width, Buffer=False)

def SDRsed(sed_cntrl_structure_fact, buffer_width, distance_to_water, **kwargs):
    '''SDR for eroded soil. Page 6 of technical docs.'''
    if sed_cntrl_structure_fact:
        return sed_cntrl_structure_fact 
    elif buffer_width>distance_to_water:
        return SDR(buffer_width, True)
    else:
        return SDR(buffer_width, True)*SDR(distance_to_water, False)
    


'''Variables Needed:
    sed_cntrl_structure_fact (simulate)
    buffer_width (from GIS?)
    distance_to_water (from GIS)
    Hydrologic Group: (From GIS)
    Soil Type: (From GIS)
    Al_level (simulate)
    fertilizer_rate (simulate)
    manure_rate (simulate)
    vegetation levels at different time series: simulate/from GIS
    soil_test_phosphorus: simulate
    
    
    '''        



def aluminumFactor(Al_level, incorp_method):
    '''Calculate the al_factor for binding added Manure/fertilizer P.
    Description of page 5 of technical docs. '''
    if Al_level<20:
        factor= 1
    elif Al_level>80:
        factor= .4
    else:
        factor= 1.2-Al_level*.01
    if str(incorp_method.lower()) =='not incorporated' or not incorp_method:
        return factor + (1-factor)/2
    else: 
        return factor    
        

        
class pApplication(object):
    '''MetaClass for fertilizer and manure applications.'''
    def __init__(self, field, incorp_method):
        self.field=field
        self.dis_runoff_factor=.02
        self.avail_factor=1
        self.incorp_method=incorp_method
       
        
    def link_to_field(self, field):
        '''Link P_application to its field.'''
        self.field=field
        self.Al_factor=aluminumFactor(self.field.params['Al_level'], self.incorp_method)
            
class fertApplication(pApplication):
    '''A fertilzer application.
    Rate: fertilizer in lbs P2O5 per acre.
    Date: Date of fertilizer application. ''' 
    def __init__(self, field, rate, incorp_method, date):
        pApplication.__init__(self, field, incorp_method)
        self.date=date
        self.rate=rate
    
    def calcDissolved_P_loss(self):
        '''Calculate Fertilizer P. Page 10 of tech docs.
Dissolved P loss from applied fertilizer is calculated similarly to that from manure.
The Fertilizer Runoff Factor is the same as the Manure Runoff Factor, 0.02 or 2%. Availability is
assumed to be 1.0 for fertilizer P.
The Aluminum and Fertilizer Factors are explained above.
The Runoff Delivery Ratio uses vegetated buffer distance only.'''
        self.Dissolved_loss=np.product([self.rate,
                                        self.dis_runoff_factor,
                                        self.Al_factor,
                                        self.avail_factor,
                                        self.field.params['hydro_factor'],
                                        self.field.params['RDR_factor'],
                                        .44
                                            ])
        return self.Dissolved_loss
       
class manureApplication(pApplication):
    '''A manure application.
    Rate: Manure in lbs P2O5 per acre.
    Date: Date of fertilizer application. 
    Time_to incorp: int, in days. 
    Type: str (not relevant right now)
    incorp_method: str.''' 
    def __init__(self, field, rate, date, time_to_incorp, incorp_method, Type):
        pApplication.__init__(self, field, incorp_method)
        self.rate=rate
        self.date=date
        self.time_to_incorp=time_to_incorp
        self.mplf=.005
        self.type=Type
        self.manure_factor=manure_factor(self.date, self.incorp_method, self.time_to_incorp, )
        
    
    
        
    def calcParticPloss(self, sdr):
        ''''Estimate particulate P runoff from a manure application. 
        Page 7 of technical docs. 
        Pass a sediment deliver ratio : sdr.'''
        self.partic_loss= np.product([self.rate,
                       self.mplf,
                       sdr,
                       self.manure_factor,
                       .44]
                        )
        return self.partic_loss
    
    
    def calcDisPloss(self):
        ''''Estimate dissolved P runoff from a manure application. 
        Page 10 of technical docs. 
        Pass a sediment deliver ratio : sdr.'''
        self.dissolved_loss= np.product( [self.rate,
                            self.dis_runoff_factor,
                            self.manure_factor,
                            self.Al_factor,
                            self.avail_factor,
                            self.field.params['hydro_factor'],
                            self.field.params['RDR_factor'],        
                            .44,
                                    ])
        return self.dissolved_loss
        