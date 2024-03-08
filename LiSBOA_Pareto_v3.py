#01/17/2023: created
#01/20/2023 (v 2): allowed option for number of beams
#01/22/2023: finalized
#01/22/2023 (v 2.1): multiple U, finalized
#01/22/2023 (v 2.2): run option, finalized
#03/31/2023 (v 3): new halo simulator
#04/03/2023: finalized

import os
cd=os.path.dirname(__file__)

import sys
import utils as utl
import LiSBOA_functions as LiS
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import warnings
import xarray as xr
from datetime import datetime

warnings.filterwarnings('ignore')
plt.close('all')
mpl.rcParams.update({'font.size': 16})

#%% Inputs
source='data/240308_HELIX_scan_info.xlsx'
source_time='data/Halo_time_info.xlsx'

#farm info
St=0.4#Strouhal number in the turbine wake (Letizia et al. 2021)
D=127#[m] rotor diameter
Lu_IEC=340.2#[m] Kaimal streamwise legnthscale (IEC 61400-1 2015)

#graphics
cmap = cm.get_cmap('viridis')
markers=['o','v','s','p','*']

#%% Initialization
scan_info=pd.read_excel(source)

now=datetime.strftime(datetime.now(),'%Y%m%d_%H%M')
utl.mkdir('data/'+now+'_'+os.path.basename(source)[:-5])

#%% Main
for id_scan in range(len(scan_info.index)):

    if scan_info['Run'].iloc[id_scan]:
        scan_name=scan_info['Scan name'].iloc[id_scan]
        turbine=scan_info['turbine'].iloc[id_scan]
        yaw=scan_info['yaw'].iloc[id_scan]
        vec_dazi=[np.float64(d) for d in str(scan_info['vec_dazi'].iloc[id_scan]).split(',')]
        vec_N_azi=[np.float64(d) for d in str(scan_info['N_azi'].iloc[id_scan]).split(',')]
        vec_azi_max=[np.float64(d) for d in str(scan_info['vec_azi_max'].iloc[id_scan]).split(',')]
        vec_dele=[np.float64(d) for d in str(scan_info['vec_dele'].iloc[id_scan]).split(',')]
        vec_N_ele=[np.float64(d) for d in str(scan_info['N_ele'].iloc[id_scan]).split(',')]
        vec_ele_max=[np.float64(d) for d in str(scan_info['vec_ele_max'].iloc[id_scan]).split(',')]
        d_cartesian=scan_info['d_cartesian'].iloc[id_scan]
        max_cartesian=scan_info['max_cartesian'].iloc[id_scan]
        Dn0=[np.float64(d) for d in scan_info['Dn0'].iloc[id_scan].split(',')]
        dr=scan_info['dr'].iloc[id_scan]
        rmin=scan_info['rmin'].iloc[id_scan]
        rmax=scan_info['rmax'].iloc[id_scan]
        ppr=scan_info['PPR'].iloc[id_scan]
        scan_mode=scan_info['scan_mode'].iloc[id_scan]
        overlapping=scan_info['overlapping'].iloc[id_scan]
        lidar_model=scan_info['lidar_model'].iloc[id_scan]
        mins=[np.float64(d) for d in scan_info['mins'].iloc[id_scan].split(',')]
        maxs=[np.float64(d) for d in scan_info['maxs'].iloc[id_scan].split(',')]
        sigma=scan_info['sigma'].iloc[id_scan]
        T_tot=scan_info['T_tot'].iloc[id_scan]
        vec_U=[np.float64(u) for u in str(scan_info['vec_U'].iloc[id_scan]).split(',')]
        
        r=np.arange(rmin,rmax,dr)
        
        if turbine:
            Lu=St*D#[m] (Letizia et al. 2021)
        else:
            Lu=Lu_IEC#[m] from IEC 2005
        
        epsilon1=np.zeros((max(len(vec_dazi),len(vec_N_azi)),len(vec_azi_max),max(len(vec_dele),len(vec_N_ele)),len(vec_ele_max)))+np.nan
        epsilon2=np.zeros((max(len(vec_dazi),len(vec_N_azi)),len(vec_azi_max),max(len(vec_dele),len(vec_N_ele)),len(vec_ele_max),len(vec_U)))+np.nan
        epsilon3=epsilon2.copy()
        
        if np.isnan(vec_dazi[0]):
            mode_azi='N_azi'
            coord_azi=vec_N_azi
        if np.isnan(vec_N_azi[0]):
            mode_azi='dazi'
            coord_azi=vec_dazi
        if np.isnan(vec_dele[0]):
            mode_ele='N_ele'
            coord_ele=vec_N_ele
        if np.isnan(vec_N_ele[0]):
            mode_ele='dele'
            coord_ele=vec_dele
        for i_dazi in range(max(len(vec_dazi),len(vec_N_azi))):
            i_azi_max=0
            
            for azi_max in vec_azi_max:
                if mode_azi=='N_azi':
                    N=int(vec_N_azi[i_dazi])
                    if N==1:
                        theta=[yaw]
                    else:
                        theta=np.linspace(-azi_max,azi_max,N)+yaw
                if mode_azi=='dazi':
                    theta=np.arange(-azi_max,azi_max+vec_dazi[i_dazi]/2+10**-10,vec_dazi[i_dazi]+10**-10)+yaw
                
                i_dele=0
                for i_dele in range(len(vec_dele)):
                    i_ele_max=0
                    if (d_cartesian==False and i_dele==i_dazi) or d_cartesian==True:         
                        for ele_max in vec_ele_max:
                            if (max_cartesian==False and i_ele_max==i_azi_max) or  max_cartesian==True:     
                                if mode_ele=='N_ele':
                                    N=int(vec_N_ele[i_dele])
                                    if N==1:
                                        ele=[0]
                                    else:     
                                        beta=np.linspace(-ele_max,ele_max,N)
                                if mode_ele=='dele':
                                    beta=np.arange(-ele_max,ele_max+vec_dele[i_dele]/2+10**-10,vec_dele[i_dele]+10**-10)
                                
                                e1,e2,e3,X2,Dd,excl,T,x1,x2=LiS.Pareto_v2(r,theta,beta,lidar_model,ppr,scan_mode,overlapping,source_time,mins,maxs,Dn0,sigma,T_tot,vec_U,Lu,D)
                                epsilon1[i_dazi,i_azi_max,i_dele,i_ele_max]=np.round(e1,1)
                                epsilon2[i_dazi,i_azi_max,i_dele,i_ele_max,:]=np.round(e2,1)
                                epsilon3[i_dazi,i_azi_max,i_dele,i_ele_max,:]=np.round(e3,1)
                            i_ele_max+=1
                    i_dele+=1
                i_azi_max+=1
            i_dazi+=1
            
        #Output
        Output = xr.Dataset({
            'epsilon1': xr.DataArray(
                        data   = epsilon1,   # enter data here
                        dims   = [mode_azi,'azi_max',mode_ele,'ele_max'],
                        coords = {mode_azi: coord_azi,'azi_max': vec_azi_max,mode_ele: coord_ele,'ele_max': vec_ele_max},
                        attrs  = {
                            '_FillValue': 'Nan',
                            'units'     : '%'
                            }
                        ),
            'epsilon2': xr.DataArray(
                        data   = epsilon2,   # enter data here
                        dims   = [mode_azi,'azi_max',mode_ele,'ele_max','U'],
                        coords = {mode_azi: coord_azi,'azi_max': vec_azi_max,mode_ele: coord_ele,'ele_max': vec_ele_max,'U':vec_U},
                        attrs  = {
                            '_FillValue': 'Nan',
                            'units'     : '%'
                            }
                        ),
            'epsilon3': xr.DataArray(
                        data   = epsilon3,   # enter data here
                        dims   = [mode_azi,'azi_max',mode_ele,'ele_max','U'],
                        coords = {mode_azi: coord_azi,'azi_max': vec_azi_max,mode_ele: coord_ele,'ele_max': vec_ele_max,'U':vec_U},
                        attrs  = {
                            '_FillValue': 'Nan',
                            'units'     : '%'
                            }
                        ),
            'Dn0': xr.DataArray(
                        data   = Dn0,   # enter data here
                        attrs  = {'units':'D'}
                        ),
            'rmin': xr.DataArray(
                        data   = rmin,   # enter data here
                        attrs  = {'units':'m'}
                        ),
            'rmax': xr.DataArray(
                        data   = rmax,   # enter data here
                        attrs  = {'units':'m'}
                        ),
            'dr': xr.DataArray(
                        data   = dr,   # enter data here
                        attrs  = {'units':'m'}
                        ),
            'PPR': xr.DataArray(
                        data   = ppr,   # enter data here
                        attrs  = {'units':'none'}
                        ),
            'scan_mode': xr.DataArray(
                        data   = scan_mode,   # enter data here
                        attrs  = {'units':'none'}
                        ),
            'overlapping': xr.DataArray(
                        data   = overlapping,   # enter data here
                        attrs  = {'units':'boolean'}
                        ),
            'mins': xr.DataArray(
                        data   = mins,   # enter data here
                        attrs  = {'units':'D'}
                        ),
            'maxs': xr.DataArray(
                        data   = maxs,   # enter data here
                        attrs  = {'units':'D'}
                        ),
            'sigma': xr.DataArray(
                        data   = sigma,   # enter data here
                        attrs  = {'units':'none'}
                        ),
            'T_tot': xr.DataArray(
                        data   = T_tot,   # enter data here
                        attrs  = {'units':'s'}
                        ),
            'yaw': xr.DataArray(
                        data   = yaw,   # enter data here
                        attrs  = {'units':'deg'}
                        ),
            'turbine': xr.DataArray(
                        data   = turbine,   # enter data here
                        attrs  = {'units':'boolean'}
                        ),
            'd_cartesian': xr.DataArray(
                        data   = d_cartesian,   # enter data here
                        attrs  = {'units':'boolean'}
                        ),
            'max_cartesian': xr.DataArray(
                        data   = max_cartesian,   # enter data here
                        attrs  = {'units':'boolean'}
                        ),
            'lidar_model': xr.DataArray(
                        data   = lidar_model,   # enter data here
                        attrs  = {'units':'none'}
                        ),
            'source_time': xr.DataArray(
                        data   = source_time,   # enter data here
                        attrs  = {'units':'none'}
                        ),
            'D': xr.DataArray(
                        data   = D,   # enter data here
                        attrs  = {'units':'m'}
                        ),
            'St': xr.DataArray(
                        data   = St,   # enter data here
                        attrs  = {'units':'none'}
                        ),
            'Lu_IEC': xr.DataArray(
                        data   = Lu_IEC,   # enter data here
                        attrs  = {'units':'m'}
                        ),
            'Lu': xr.DataArray(
                        data   = Lu,   # enter data here
                        attrs  = {'units':'m'}
                        )},
            attrs = {'Contact': 'stefano.letizia@nrel.gov',
                      'Description':'Pareto front result from LiSBOA algorithm',
                      'Source':source,
                      'Scan name':scan_name})
        
        Output.to_netcdf('data/'+now+'_'+os.path.basename(source)[:-5]+'/'+scan_name+'.nc')
        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write(scan_name+' completed')
        sys.stdout.write('\n')
        sys.stdout.flush()
