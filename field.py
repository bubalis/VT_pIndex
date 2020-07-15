# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:46:24 2020

@author: benja

This script defines all of the base classes and functions to run the pIndex.'''

"""

from helper_funcs import lookupBRV, lookup_RAF, get_soil_hyd_factor, manure_factor, getFertFactor, uptake_dict
import numpy as np

class CropField():
    '''A single field to calculate the p Index for.'''
    
    def calcPindex(self):
        '''Calculate the p Index for this field.'''
        self.results['surface particulate loss']= self.surf_particulate_loss()
        self.results['surface dissolved loss']= self.surf_dissolved_loss()
        self.results['subsurface_loss']=self.subsurface_loss()
                
        
        self.results['total p index']=sum(self.results[pathway] for pathway in 
                    ['surface particulate loss', 'surface dissolved loss', 'subsurface_loss'])
        return self.results
        
    
    
    def setup_data(self):
        '''Calculate needed parameters from existing data.
        This function sets factors necessary for the pIndex that are
        deterministically calculated from loaded in data.
        '''
        
        self.params['soil_total_phos']=calc_soilTP(self.params['soil_test_phos'], self.params['soil_is_clay'])
        self.params['runoff_adj_factor']=lookup_RAF(self.params['hydro_group'], self.params['veg_type'], self.params['cover_perc'], self.params['tile_drain'])
        self.params["baseROV"]=lookupBRV(self.params['county'], self.params['elevation'])
        
        
        
        self.params['hydro_factor']=get_soil_hyd_factor(self.params['hydro_group'], self.params['tile_drain'])
        self.params['RDR_factor']=calcRDR(**self.params)
        self.params['crop_uptake']=uptake_dict[self.params['veg_type']]
        self.params['soil_al_factor']=al_factor_soilP(self.params['Al_level'])
        self.results={}
        self.link_to_apps()
        self.add_soil_P()
    
    
  
               
    
    def sum_fractions(self, functions_to_sum, default_args={}):
        '''Return the sum of multiple functions, with all of the field's parameters passed to that function.
        Default args can be used to overide parameters of the field object.'''
        
        kwargs={**self.params,**default_args}
        s=0
        
        for function in functions_to_sum:
            result=function(**kwargs)
            self.results[function.__name__]=result
            s+=result
        return s
    
    def surf_particulate_loss(self, scaling_factor=80):
        '''Surface Particulate P loss.'''
        return self.sum_fractions([erodedSoilP, manure_partic_P]) *scaling_factor
   
    
    def surf_dissolved_loss(self, scaling_factor=80):
        '''Surface Dissolved P loss.'''
        return self.sum_fractions([dis_soilP, dis_manureP, fertilizerP,])*scaling_factor
   
    
    def subsurface_loss(self, scaling_factor=80):
        '''Subsurface loss from tile drainage. Page 10-11 of technical docs. '''
        
        PF6=.2
        default_args={'SDR_factor':1, 
                      'RDR_factor':1, 
                      'buffer_width':0, 
                      'manure_setback':0, 
                      'sed_cntrl_structure_fact':None}
        
        if self.params.get('tile_drain', None):
            return self.sum_fractions([erodedSoilP, 
                                       manure_partic_P,
                                       dis_soilP,
                                       dis_manureP, 
                                       fertilizerP], 
                                        default_args)*PF6*scaling_factor
        else:
            return 0
        
        
        
    def link_to_apps(self):
        '''Set linkages between p applications and field object.'''
        self.manure_applications=self.params['manure_applications']
        self.fertilizer_applications=self.params['fertilizer_applications']
        self.params['total_p_added']=sum([app.rate for app in self.manure_applications+self.fertilizer_applications])
        for app in self.manure_applications+self.fertilizer_applications:
            app.link_to_field(self)
        
        
            
    def add_soil_P(self):
        '''Increase soil P levels to reflect fertilizer and manure applications.
        Cells 134 and 135 in Spreadsheet.
        from Technical docs, page 6:
            It is assumed that the soil test was made before any manure or fertilizer applications.
Therefore, the added P (after subtracting crop uptake, and dissolved and particulate P losses)
is used to adjust soil test P by the equation, ∆STP/added P = 1.277 * Al0.7639 (see
Aluminum factor above). This adjusted STP is then plugged into the appropriate regression
equation for Total P.

A limit on this TP estimate is made by adding the applied manure and fertilizer P (minus
losses, as above), which represents added TP, to the TP calculated by regression from the
original soil test P. The estimate cannot be greater than this limiting value.
        '''
        apps=self.fertilizer_applications+self.manure_applications
        added_TP=sum([app.TPincr() for app in apps])
        added_STP=sum([app.STPincr() for app in apps])
        adj_stp=self.params['soil_test_phos']+added_STP
        self.params['adj_total_phos']=min(
                                    [calc_soilTP(adj_stp, self.params['soil_is_clay']),
                                           self.params['soil_total_phos']+added_TP])
        self.params['adj_test_phos']=adj_stp
                   
                      

        
        

    

class CropFieldFromDic(CropField):
    '''Initialize crop field from a dictionary'''
    def __init__(self, dic):
        self.params=dic
        self.setup_data()
        
    



        
def calc_soilTP(test_phos, soil_is_clay):
    '''Calculated total phosphorus from test_phos. 
    From documentation page 6'''
    if soil_is_clay:
        return 6.56*test_phos+650
    else:
        return 10.87*test_phos+760
    
        

    
def erodedSoilP(erosion_rate, adj_test_phos, adj_total_phos,  **kwargs):
    '''Return the value for eroded soil Phosphorus. 
    Page 6 of technical docs:  
    1. Sediment (eroded soil) P loss = E * TP * TP Availability * SDR
The four terms are:
    
a. E = Annual soil loss. The RUSLE (ver. 1 or 2) or WEPP edge-of-field erosion rate in
tons/ac is divided by 500 to convert it to million lb/ac. Annual rather than rotation erosion
value should be used...

(b is Total Phosphorus)
...
c. TP availability factor. Research suggests that only a fraction of the total P in soils is
available for the growth of algae. This factor ranges from 0.2 (i.e., 20% of TP is algal
available) at a soil test P of 0 ppm, to a maximum of 0.4 at STP = 100 ppm (based on a
chemical extraction of Lake Champlain sediments that approximates algal uptake).
d. SDR = Sediment Delivery Ratio (see Sediment and Runoff Delivery Ratios, above). '''
    
    return np.product([erosion_rate,
                       .002,    #conversion factor to million lbs per acre
                       adj_total_phos,
                       P_avail_excel(adj_test_phos),
                       SDRsed(**kwargs)])



def P_avail(adj_test_phos):
    '''TP availability factor. From Page 6:
    This factor ranges from 0.2 (i.e., 20% of TP is algal available) 
at a soil test P of 0 ppm, to a maximum of 0.4 at STP = 100 ppm
        '''
    if adj_test_phos<100:
        return .2+adj_test_phos/100*.2
    else:
        return .4
    
    
    
def P_avail_excel(adj_test_phos):
    '''TP availability factor. This formula from the excel spreadsheet'''
    if adj_test_phos<100:
        return .1+adj_test_phos/1000
    else:
        return .2


def manure_partic_P(manure_applications, **kwargs):
    '''Phosphorus lost from manure particulate P
    Page 7 of Technical docs. Multiply by the SDR for manure and by .44 to convert to lbs P'''
    sdr=SDRm(buffer=False, **kwargs)
    return sum([m_app.partic_loss for m_app in manure_applications])*sdr*.44

  
      

def dis_soilP(adj_test_phos, baseROV, runoff_adj_factor, RDR_factor,  **kwargs):
    '''Calculate dissolved P loss.   '''
    return np.product([DRPexcel(adj_test_phos),
                       baseROV,
                       runoff_adj_factor,
                       RDR_factor])
    


def DRP(adj_test_phos):
    '''Calculate Dissolved Reactive Phosphorus:
        From VTPI Tech Docs:
           Dissolved reactive P (DRP) concentration in runoff, expressed in parts per million.
Research involving simulated rainfall applied to field plots on a wide variety of Vermont
agricultural soils has provided a good relationship between soil test P (STP) and DRP
concentration in runoff: DRP = 0.1275 + 0.0104 * STP (see Figure 4). Soil test P is first
adjusted for any increment due to manure or fertilizer P added since the soil test was made''' 
    return .1275+(.0104*adj_test_phos)



def DRPexcel(adj_test_phos):
    '''Dissolved Reactive Phosphorus Based on #s in the excel model'''
    return 2*(.00705*adj_test_phos+.03)
    



    
def dis_manureP(manure_applications, RDR_factor, **kwargs):
    '''Calculate sum of dissolved loss from Manure Applications.
    Multiply by sdr and .44 to convert to lbs P'''
    sdr=SDRm(buffer=False, **kwargs)
    return sum([m_app.dissolved_loss for m_app in manure_applications])*sdr*.44

    



    
def fertilizerP(fertilizer_applications, RDR_factor, **kwargs):
    '''Calculate sum of dissolved P from fertilizer applications. 
    Multiply by the RDR factor and .44 to convert to lbs P per acre.'''
    return sum([f_app.dissolved_loss for f_app in fertilizer_applications])*.44*RDR_factor
    

def SDR(distance, buffer=True, clay=False):
    '''Sediment and Runoff Delivery Ratios Page 2 of the documentation.
    distance: width of buffer or distance to water (in feet).
    Buffer: Boolean: are we calculating Buffer factor?
    clay: reflects the excel spreadsheet where if soil is clay, buffer factor is calculated as a distance factor.
    (See Row 63)
    '''
    if (buffer and (not clay)):
        fact= 1.744*np.exp((-43-distance)/45)+.4       
    else: 
        fact= 1.047*np.exp((-70-distance)/60)+.7
    if fact<1:
        return fact
    else:
        return 1
    
    
    
def SDRm(manure_setback,buffer_width, buffer, **kwargs):
    '''SDR for a manure application.'''
    if 'SDR_factor' in kwargs.keys():
        return kwargs['SDR_factor']
    else:
        return SDR(buffer_width+manure_setback, buffer=buffer)



def calcRDR(buffer_width, **kwargs):
    '''Calculate basic RDR.'''
    if 'RDR_factor' in kwargs.keys():
        return kwargs['RDR_factor']
    else:
        return SDR(buffer_width, buffer=False, clay=kwargs['soil_is_clay'])




def SDRsed(sed_cntrl_structure_fact, buffer_width, distance_to_water, **kwargs):
    '''SDR for eroded soil. Page 6 of technical docs.
   if a sediment control structure is entered into the P Index, its factor 
May 24, 2017 Vermont-P-Index-User-Guide with logos and statements.doc 7
(with a range of 0 to 0.2) is used instead of the SDR (range 0.4 to 1.0). In addition, if the
total distance to the nearest water body is greater than the buffer width, the additional
distance beyond the buffer is considered to have some effect on sediment load. The Distance
Factor is calculated for this additional distance, and is multiplied by the Buffer Factor for the
final SDR    '''
    if sed_cntrl_structure_fact:
        return sed_cntrl_structure_fact 
    elif buffer_width==distance_to_water:
        return SDR(buffer_width, True)
    else:
        return SDR(buffer_width, True)*SDR(distance_to_water, False)
    


def al_factor_soilP(al_level):
    '''Adjustment for Aluminum level for P applications.
    Page 5 of Technical Docs:
        Soil “reactive aluminum” binds a fraction of added P, making it less available to plants, and also
less extractable by buffer solutions used in soil testing laboratories to determine “soil test P” (STP), such
as the Modified Morgan test in Vermont. 
The relation between reactive Al and the change in STP per
unit of added fertilizer P is described in Jokela (1998) by a power curve, ∆STP per 1-ppm added P =
1.277 * Al-0.7639 (see diagram). 
This equation is used in the P Index to estimate the increases in soil test P
and total P with added fertilizer and manure.'''
    if al_level<=10:
        return .2
    else:
        factor=1.277*al_level**(-0.7639)
    return max([factor, .03])       


def aluminum_factor_runoff(Al_level, incorp_method):
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
        '''Link P_application to its field,
        calculate all relevant outputs.'''
        self.field=field
        self.Al_factor_runoff=aluminum_factor_runoff(self.field.params['Al_level'], self.incorp_method)
        self.Al_factor_soil=self.field.params['soil_al_factor']  
        self.runCalcs()
        
    def runCalcs(self):
        '''Caclulate all quantities for this application.'''
        self.calcDisPloss()
        self.calcParticLoss()
        self.P_incr(self.field.params['total_p_added'], self.field.params['crop_uptake'])
        
    def P_after_uptake(self,total_p_added, crop_uptake):
        '''Calculate P from P_application that is not taken up by crop.'''
        if total_p_added==0:
            total_p_added=1
        increment= self.rate-(crop_uptake/2*self.rate/total_p_added)
        if increment>0:
            return increment
        return 0
    
    def P_incr(self, total_p_added, crop_uptake):
        '''Calculate P not eroded or taken up by crop from manure.
        For use in calculating P adjusted soil P levels.'''
        self.post_uptake=self.P_after_uptake(total_p_added, crop_uptake)
        incr=self.post_uptake-self.dissolved_loss-self.partic_loss
        if incr>0:
            self.incr=incr
        else:   
            self.incr=0
      
    
      



 
class fertApplication(pApplication):
    '''A fertilzer application.
    Rate: fertilizer in lbs P2O5 per acre.
    Date: Date of fertilizer application. ''' 
    def __init__(self, field, rate, incorp_method, date):
        pApplication.__init__(self, field, incorp_method)
        self.date=date
        self.rate=rate
        self.method_factor=getFertFactor(self.incorp_method, self.date)
    


    
    def calcDisPloss(self):
        '''Calculate Fertilizer P loss at edge-of-field.
        Page 10 of tech docs.
Dissolved P loss from applied fertilizer is calculated similarly to that from manure.
The Fertilizer Runoff Factor is the same as the Manure Runoff Factor, 0.02 or 2%. Availability is
assumed to be 1.0 for fertilizer P.
The Aluminum and Fertilizer Factors are explained above.
'''
        self.dissolved_loss=np.product([self.rate,
                                        self.dis_runoff_factor,
                                        self.Al_factor_runoff,
                                        self.avail_factor,
                                        self.field.params['hydro_factor'],
                                        self.method_factor,
                                            ])
        
     
    def calcParticLoss(self):
        '''No P is lost from fertilizer in partic form.'''
        self.partic_loss=0
      
        
    def TPincr(self):
        '''Cell 131 in spreadsheet. Total Phosphorus added to soil from fertilizer application.'''
        if self.method_factor>=.4:
            mf=7.5
        else:
            mf=1
        return self.incr*.44/2*mf
    
    def STPincr(self):
        '''Cell 130 in spreadsheet. Soil Test Phosphorus added to soil from fertilizer application.'''
        if self.method_factor>=.4:
                mf=7.5
        else:
            mf=1
        return self.incr*self.Al_factor_soil *mf
    
    

    

        
    
class manureApplication(pApplication):
    '''A manure application.
    Rate: Manure in lbs P2O5 per acre.
    Date: Date of fertilizer application. 
    Time_to incorp: int, in days. 
    Type: str (not relevant right now, the type of manure.)
    incorp_method: str.''' 
    def __init__(self, field, rate, date, time_to_incorp, incorp_method, Type):
        pApplication.__init__(self, field, incorp_method)
        self.rate=rate
        self.date=date
        self.time_to_incorp=time_to_incorp
        self.mplf=.005
        self.type=Type
        self.manure_factor=manure_factor(self.date, self.incorp_method, self.time_to_incorp, )
        
    
        
    def calcParticLoss(self):
        ''''Estimate particulate P runoff from a manure application, to edge-of-field. 
        Page 7 of technical docs. 
        '''
        self.partic_loss= np.product([self.rate,
                       self.mplf,
                       self.manure_factor,
                       ]
                        )
    
    
    def calcDisPloss(self):
        ''''Estimate dissolved P runoff from a manure application to edge-of-field
        Page 10 of technical docs.
        '''
        self.dissolved_loss= np.product( [self.rate,
                            self.dis_runoff_factor,
                            self.manure_factor,
                            self.Al_factor_runoff,
                            self.avail_factor,
                            self.field.params['hydro_factor'],
                                    ])
 
    
    def STPincr(self):
        '''Cell 100 in spreadsheet.  Soil Test Phosphorus added to soil from manure application.'''
        if self.manure_factor==1:
            mf=3
        else:
            mf=1
        return  self.incr*0.44/2*mf*self.Al_factor_soil
    

    def TPincr(self):
        '''Cell 101 in spreadsheet. Total Phosphorus added to soil from manure application.'''
        if self.manure_factor==1:
            mf=7.5
        else:
            mf=1
        return self.incr*.44/2*mf



