#01/20/2023: created
#01/22/2023 (v 2): only one velocity case, finalized
#01/22/2023 (v 3): sensitivity to U reintroduced
#01/23/2023 (LiSBOA_postpro_v2): made general
#01/23/2023 : finalized
#03/31/2023 (v 3): new halo simulator

import os
cd=os.path.dirname(__file__)

import utils as utl
import LiSBOA_functions as LiS
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import warnings
import xarray as xr
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
root='data/20240308_1202_240308_HELIX_scan_info/'
source='Wake statistics.nc'
sel=[2,15,2,15]#optimal azimuth step, maximum azimuth, elevation step, maximum elevation

save_fig=True

#graphics
cmap = cm.get_cmap('viridis')
markers=['o','v','s','p','*']

#%% Initialization

#%% Main

Data=xr.open_dataset(root+source)

epsilon1=Data['epsilon1'].values
epsilon2=Data['epsilon2'].values
epsilon3=Data['epsilon3'].values

yaw=np.float64(Data['yaw'].values)
try:
    vec_dazi=Data['dazi'].values
    mode_azi='dazi'
except:
    vec_N_azi=Data['N_azi'].values
    mode_azi='N_azi'
vec_azi_max=Data['azi_max'].values
try:
    vec_dele=Data['dele'].values
    mode_ele='dele'
except:
    vec_N_ele=Data['N_ele'].values
    mode_ele='N_ele'
vec_ele_max=Data['ele_max'].values
vec_U=np.float64(Data['U'].values)
if utl.len2(vec_U)==1:
    vec_U=np.array([vec_U])
Dn0=Data['Dn0'].values
dr=np.float64(Data['dr'].values)
rmin=np.float64(Data['rmin'].values)
rmax=np.float64(Data['rmax'].values)
ppr=np.float64(Data['PPR'].values)
scan_mode=str(Data['scan_mode'].values)
overlapping=Data['overlapping'].item()
source_time=str(Data['source_time'].values)
lidar_model=str(Data['lidar_model'].values)
mins=Data['mins'].values
maxs=Data['maxs'].values
sigma=np.float64(Data['sigma'].values)
T_tot=np.float64(Data['T_tot'].values)
Lu=np.float64(Data['Lu'].values)
D=np.float64(Data['D'].values)
d_cartesian=Data['d_cartesian'].values
max_cartesian=Data['max_cartesian'].values

r=np.arange(rmin,rmax,dr)

#%% Main
if len(sel)==4:
    assert np.sum(vec_dazi==sel[0])==1, 'azimuth resolution not available'
    assert np.sum(vec_azi_max==sel[1])==1, 'maximum azimuth not available'
    assert np.sum(vec_dele==sel[2])==1, 'elevation resolution not available'
    assert np.sum(vec_ele_max==sel[3])==1, 'maximum elevation not available'
    
    i_azi_max_sel=np.where(vec_azi_max==sel[1])[0][0]
    azi_max=sel[1]
    ele_max=sel[3]
    if mode_azi=='N_azi':
        N=int(sel[0])
        if N==1:
            theta=[yaw]
        else:
            theta=np.linspace(-azi_max,azi_max,N)+yaw
    if mode_azi=='dazi':
        theta=np.arange(-azi_max,azi_max+sel[0]/2+10**-10,sel[0]+10**-10)+yaw
        
    if mode_ele=='N_ele':
        N=int(sel[0])
        if N==1:
            beta=[0]
        else:
            beta=np.linspace(-ele_max,ele_max,N)
    if mode_ele=='dele':
        beta=np.arange(-ele_max,ele_max+sel[2]/2+10**-10,sel[2]+10**-10)
    
    e1,e2,e3,X2,Dd,excl,T,x1,x2=LiS.Pareto_v2(r,theta,beta,lidar_model,ppr,scan_mode,overlapping,source_time,mins,maxs,Dn0,sigma,T_tot,vec_U,Lu,D)
    
    e1=np.round(e1,1)
    e2=np.round(e2,1)
    e3=np.round(e3,1)
    T=np.round(T,1)
        
    if len(Dn0)==3:
        X_plot=np.transpose(X2[0],(1,2,0))
        Y_plot=np.transpose(X2[1],(1,2,0))
        Z_plot=np.transpose(X2[2],(1,2,0))
        excl_plot=np.transpose(excl,(1,2,0))
        Dd_plot=np.transpose(Dd,(1,2,0))
        Dd_plot[excl_plot]=np.nan
    else:
        X_plot=X2[0]
        Y_plot=X2[1]
        Z_plot=X2[0]*0
        excl_plot=excl
        Dd_plot=Dd
        Dd_plot[excl_plot]=np.nan
else:
    X_plot=np.nan
    Y_plot=np.nan
    Z_plot=np.nan
    Dd_plot=np.nan
    excl_plot=np.nan
    e1=np.nan
    e2=np.nan
    e3=np.nan
    i_azi_max_sel=0

#%% Plots
if len(mins)==2:
    mins=np.append(mins,-1)
    maxs=np.append(maxs,1)
for i_U in range(len(vec_U)):
    U=vec_U[i_U]
    
    #Pareto 1
    if d_cartesian==True and max_cartesian==True:
        plt.figure(figsize=(18,10))
        i_dele=0
        for dele in vec_dele:
            i_ele_max=0
            for ele_max in vec_ele_max:
                
                ax=plt.subplot(len(vec_ele_max),len(vec_dele),i_ele_max*(len(vec_dele))+i_dele+1)
                plt.plot(epsilon1.ravel(),epsilon2[:,:,:,:,i_U].ravel(),'.k',markersize=15,alpha=0.2)
                
                for i_dazi in range(len(vec_dazi)):
                    for i_azi_max in range(len(vec_azi_max)):
                        p=plt.plot(epsilon1[i_dazi,i_azi_max,i_dele,i_ele_max],epsilon2[i_dazi,i_azi_max,i_dele,i_ele_max,i_U],'.',markersize=10,alpha=1,markeredgecolor='k',marker=markers[i_azi_max],color=cmap(i_dazi/(len(vec_dazi)-1)))
                
                if len(sel)==4:
                    if dele==sel[2] and ele_max==sel[3]:
                        plt.plot(e1,e2[i_U],'.',markersize=12,alpha=1,markeredgecolor='r',marker=markers[i_azi_max_sel],markeredgewidth=2,fillstyle='none')
                    
                for i_azi_max in range(len(vec_azi_max)):
                    plt.plot(1000,1000,'.',markeredgecolor='k',marker=markers[i_azi_max],label=r'$\theta_{max} ='+utl.str_dec(vec_azi_max[i_azi_max])+'^\circ$',color='k')
                
                for i_dazi in range(len(vec_dazi)):
                    plt.plot(1000,1000,'.',marker='o',markeredgecolor='k',color=cmap(i_dazi/(len(vec_dazi)-1)),label=r'$\Delta \theta ='+utl.str_dec(vec_dazi[i_dazi])+'^\circ$')
                i_ele_max+=1
            
                plt.xlim([0,101])
                plt.ylim([0,101])
                
                ax.grid(visible=True) 
                plt.title(r'$\Delta \beta ='+utl.str_dec(dele)+'^\circ$, '+r'$\beta_{max}='+utl.str_dec(ele_max)+'^\circ$, $U_\infty = '+utl.str_dec(U)+'$ m/s')
                plt.xlabel(r'$\epsilon_I$ [%]')
                plt.ylabel(r'$\epsilon_{II}$ [%]')
            i_dele+=1
    
    if d_cartesian==False and max_cartesian==True:
        plt.figure(figsize=(18,6))
        i_ele_max=0
        for ele_max in vec_ele_max:
            ax=plt.subplot(1,len(vec_ele_max),i_ele_max+1)
            plt.plot(epsilon1.ravel(),epsilon2[:,:,:,:,i_U].ravel(),'.k',markersize=15,alpha=0.2)
            
            for i_dazi in range(len(vec_dazi)):
                for i_azi_max in range(len(vec_azi_max)):
                    p=plt.plot(epsilon1[i_dazi,i_azi_max,i_dazi,i_ele_max],epsilon2[i_dazi,i_azi_max,i_dazi,i_ele_max,i_U],'.',markersize=10,alpha=1,markeredgecolor='k',marker=markers[i_azi_max],color=cmap(i_dazi/(len(vec_dazi)-1)))
            
            if len(sel)==4:         
                if ele_max==sel[3]:
                    plt.plot(e1,e2[i_U],'.',markersize=12,alpha=1,markeredgecolor='r',marker=markers[i_azi_max_sel],markeredgewidth=2,fillstyle='none')
                
            for i_azi_max in range(len(vec_azi_max)):
                plt.plot(1000,1000,'.',markeredgecolor='k',marker=markers[i_azi_max],label=r'$\theta_{max} ='+utl.str_dec(vec_azi_max[i_azi_max])+'^\circ$',color='k')
            
            for i_dazi in range(len(vec_dazi)):
                plt.plot(1000,1000,'.',marker='o',markeredgecolor='k',color=cmap(i_dazi/(len(vec_dazi)-1)),label=r'$\Delta \theta ='+utl.str_dec(vec_dazi[i_dazi])+r'^\circ$, $\Delta \beta ='+utl.str_dec(vec_dele[i_dazi])+'^\circ$ ')
            i_ele_max+=1
            
            plt.xlim([0,101])
            plt.ylim([0,101])
            
            ax.grid(visible=True) 
            plt.title(r'$\beta_{max}='+utl.str_dec(ele_max)+'^\circ$, $U_\infty = '+utl.str_dec(U)+'$ m/s')
            plt.xlabel(r'$\epsilon_I$ [%]')
            plt.ylabel(r'$\epsilon_{II}$ [%]')
    
    if d_cartesian==True and max_cartesian==False:
        plt.figure(figsize=(18,6))
        i_dele=0
        for dele in vec_dele:       
            ax=plt.subplot(1,len(vec_dele),i_dele+1)
            plt.plot(epsilon1.ravel(),epsilon2[:,:,:,:,i_U].ravel(),'.k',markersize=15,alpha=0.2)
            
            for i_dazi in range(len(vec_dazi)):
                for i_azi_max in range(len(vec_azi_max)):
                    p=plt.plot(epsilon1[i_dazi,i_azi_max,i_dele,i_azi_max],epsilon2[i_dazi,i_azi_max,i_dele,i_azi_max,i_U],'.',markersize=10,alpha=1,markeredgecolor='k',marker=markers[i_azi_max],color=cmap(i_dazi/(len(vec_dazi)-1)))
            
            if len(sel)==4:        
                if dele==sel[2]:
                    plt.plot(e1,e2[i_U],'.',markersize=12,alpha=1,markeredgecolor='r',marker=markers[i_azi_max_sel],markeredgewidth=2,fillstyle='none')
                
            for i_azi_max in range(len(vec_azi_max)):
                plt.plot(1000,1000,'.',markeredgecolor='k',marker=markers[i_azi_max],label=r'$\theta_{max} ='+utl.str_dec(vec_azi_max[i_azi_max])+'^\circ$, $\beta_{max} ='+utl.str_dec(vec_ele_max[i_azi_max])+'^\circ$',color='k')
            
            for i_dazi in range(len(vec_dazi)):
                plt.plot(1000,1000,'.',marker='o',markeredgecolor='k',color=cmap(i_dazi/(len(vec_dazi)-1)),label=r'$\Delta \theta ='+utl.str_dec(vec_dazi[i_dazi])+'^\circ$')
            i_ele_max+=1
            
            plt.xlim([0,101])
            plt.ylim([0,101])
            
            ax.grid(visible=True) 
            plt.title(r'$\Delta \beta ='+utl.str_dec(dele)+'^\circ$, $U_\infty = '+utl.str_dec(U)+'$ m/s')
            plt.xlabel(r'$\epsilon_I$ [%]')
            plt.ylabel(r'$\epsilon_{II}$ [%]')
    
    if d_cartesian==False and max_cartesian==False:
        plt.figure(figsize=(12,10))
        for i_dazi in range(len(vec_dazi)):
            for i_azi_max in range(len(vec_azi_max)):
                p=plt.plot(epsilon1[i_dazi,i_azi_max,i_dazi,i_azi_max],epsilon2[i_dazi,i_azi_max,i_dazi,i_azi_max,i_U],'.',markersize=10,alpha=1,markeredgecolor='k',marker=markers[i_azi_max],color=cmap(i_dazi/(len(vec_dazi)-1)))
        
        plt.plot(e1,e2[i_U],'.',markersize=12,alpha=1,markeredgecolor='r',marker=markers[i_azi_max_sel],markeredgewidth=2,fillstyle='none')
        
        for i_azi_max in range(len(vec_azi_max)):
            plt.plot(1000,1000,'.',markeredgecolor='k',marker=markers[i_azi_max],label=r'$\theta_{max} =\beta_{max}='+utl.str_dec(vec_azi_max[i_azi_max])+'^\circ$',color='k')
        
        for i_dazi in range(len(vec_dazi)):
            plt.plot(1000,1000,'.',marker='o',markeredgecolor='k',color=cmap(i_dazi/(len(vec_dazi)-1)),label=r'$\Delta \theta =\Delta \beta='+utl.str_dec(vec_dazi[i_dazi])+'^\circ$')
        
        plt.xlim([0,101])
        plt.ylim([0,101])
        
        plt.gca().grid(visible=True) 
        plt.title(r'$U_\infty = '+utl.str_dec(U)+'$ m/s')
        plt.xlabel(r'$\epsilon_I$ [%]')
        plt.ylabel(r'$\epsilon_{II}$ [%]')
        
    plt.legend().set_draggable(state=True)
    plt.tight_layout()
    
    #Pareto 2
    if d_cartesian==True and max_cartesian==True:
        plt.figure(figsize=(18,10))
        i_dele=0
        for dele in vec_dele:
            i_ele_max=0
            for ele_max in vec_ele_max:
                
                ax=plt.subplot(len(vec_ele_max),len(vec_dele),i_ele_max*(len(vec_dele))+i_dele+1)
                plt.plot(epsilon1.ravel(),epsilon3[:,:,:,:,i_U].ravel(),'.k',markersize=15,alpha=0.2)
                
                for i_dazi in range(len(vec_dazi)):
                    for i_azi_max in range(len(vec_azi_max)):
                        p=plt.plot(epsilon1[i_dazi,i_azi_max,i_dele,i_ele_max],epsilon3[i_dazi,i_azi_max,i_dele,i_ele_max,i_U],'.',markersize=10,alpha=1,markeredgecolor='k',marker=markers[i_azi_max],color=cmap(i_dazi/(len(vec_dazi)-1)))
                
                if len(sel)==4:
                    if dele==sel[2] and ele_max==sel[3]:
                        plt.plot(e1,e3[i_U],'.',markersize=12,alpha=1,markeredgecolor='r',marker=markers[i_azi_max_sel],markeredgewidth=2,fillstyle='none')
                    
                for i_azi_max in range(len(vec_azi_max)):
                    plt.plot(1000,1000,'.',markeredgecolor='k',marker=markers[i_azi_max],label=r'$\theta_{max} ='+utl.str_dec(vec_azi_max[i_azi_max])+'^\circ$',color='k')
                
                for i_dazi in range(len(vec_dazi)):
                    plt.plot(1000,1000,'.',marker='o',markeredgecolor='k',color=cmap(i_dazi/(len(vec_dazi)-1)),label=r'$\Delta \theta ='+utl.str_dec(vec_dazi[i_dazi])+'^\circ$')
                i_ele_max+=1
            
                plt.xlim([0,101])
                plt.ylim([0,101])
                
                ax.grid(visible=True) 
                plt.title(r'$\Delta \beta ='+utl.str_dec(dele)+'^\circ$, '+r'$\beta_{max}='+utl.str_dec(ele_max)+'^\circ$, $U_\infty = '+utl.str_dec(U)+'$ m/s')
                plt.xlabel(r'$\epsilon_I$ [%]')
                plt.ylabel(r'$\epsilon_{III}$ [%]')
            i_dele+=1
    
    if d_cartesian==False and max_cartesian==True:
        plt.figure(figsize=(18,6))
        i_ele_max=0
        for ele_max in vec_ele_max:
            ax=plt.subplot(1,len(vec_ele_max),i_ele_max+1)
            plt.plot(epsilon1.ravel(),epsilon3[:,:,:,:,i_U].ravel(),'.k',markersize=15,alpha=0.2)
            
            for i_dazi in range(len(vec_dazi)):
                for i_azi_max in range(len(vec_azi_max)):
                    p=plt.plot(epsilon1[i_dazi,i_azi_max,i_dazi,i_ele_max],epsilon3[i_dazi,i_azi_max,i_dazi,i_ele_max,i_U],'.',markersize=10,alpha=1,markeredgecolor='k',marker=markers[i_azi_max],color=cmap(i_dazi/(len(vec_dazi)-1)))
            
            if len(sel)==4:        
                if ele_max==sel[3]:
                    plt.plot(e1,e3[i_U],'.',markersize=12,alpha=1,markeredgecolor='r',marker=markers[i_azi_max_sel],markeredgewidth=2,fillstyle='none')
                
            for i_azi_max in range(len(vec_azi_max)):
                plt.plot(1000,1000,'.',markeredgecolor='k',marker=markers[i_azi_max],label=r'$\theta_{max} ='+utl.str_dec(vec_azi_max[i_azi_max])+'^\circ$',color='k')
            
            for i_dazi in range(len(vec_dazi)):
                plt.plot(1000,1000,'.',marker='o',markeredgecolor='k',color=cmap(i_dazi/(len(vec_dazi)-1)),label=r'$\Delta \theta ='+utl.str_dec(vec_dazi[i_dazi])+r'^\circ$, $\Delta \beta ='+utl.str_dec(vec_dele[i_dazi])+'^\circ$ ')
            i_ele_max+=1
            
            plt.xlim([0,101])
            plt.ylim([0,101])
            
            ax.grid(visible=True) 
            plt.title(r'$\beta_{max}='+utl.str_dec(ele_max)+'^\circ$, $U_\infty = '+utl.str_dec(U)+'$ m/s')
            plt.xlabel(r'$\epsilon_I$ [%]')
            plt.ylabel(r'$\epsilon_{III}$ [%]')
    
    if d_cartesian==True and max_cartesian==False:
        plt.figure(figsize=(18,6))
        i_dele=0
        for dele in vec_dele:       
            ax=plt.subplot(1,len(vec_dele),i_dele+1)
            plt.plot(epsilon1.ravel(),epsilon3[:,:,:,:,i_U].ravel(),'.k',markersize=15,alpha=0.2)
            
            for i_dazi in range(len(vec_dazi)):
                for i_azi_max in range(len(vec_azi_max)):
                    p=plt.plot(epsilon1[i_dazi,i_azi_max,i_dele,i_azi_max],epsilon3[i_dazi,i_azi_max,i_dele,i_azi_max,i_U],'.',markersize=10,alpha=1,markeredgecolor='k',marker=markers[i_azi_max],color=cmap(i_dazi/(len(vec_dazi)-1)))
            
            if len(sel)==4:        
                if dele==sel[2]:
                    plt.plot(e1,e3[i_U],'.',markersize=12,alpha=1,markeredgecolor='r',marker=markers[i_azi_max_sel],markeredgewidth=2,fillstyle='none')
                
            for i_azi_max in range(len(vec_azi_max)):
                plt.plot(1000,1000,'.',markeredgecolor='k',marker=markers[i_azi_max],label=r'$\theta_{max} ='+utl.str_dec(vec_azi_max[i_azi_max])+'^\circ$, $\beta_{max} ='+utl.str_dec(vec_ele_max[i_azi_max])+'^\circ$',color='k')
            
            for i_dazi in range(len(vec_dazi)):
                plt.plot(1000,1000,'.',marker='o',markeredgecolor='k',color=cmap(i_dazi/(len(vec_dazi)-1)),label=r'$\Delta \theta ='+utl.str_dec(vec_dazi[i_dazi])+'^\circ$')
            i_ele_max+=1
            
            plt.xlim([0,101])
            plt.ylim([0,101])
            
            ax.grid(visible=True) 
            plt.title(r'$\Delta \beta ='+utl.str_dec(dele)+'^\circ$, $U_\infty = '+utl.str_dec(U)+'$ m/s')
            plt.xlabel(r'$\epsilon_I$ [%]')
            plt.ylabel(r'$\epsilon_{III}$ [%]')
    
    if d_cartesian==False and max_cartesian==False:
        plt.figure(figsize=(12,10))
        for i_dazi in range(len(vec_dazi)):
            for i_azi_max in range(len(vec_azi_max)):
                p=plt.plot(epsilon1[i_dazi,i_azi_max,i_dazi,i_azi_max],epsilon3[i_dazi,i_azi_max,i_dazi,i_azi_max,i_U],'.',markersize=10,alpha=1,markeredgecolor='k',marker=markers[i_azi_max],color=cmap(i_dazi/(len(vec_dazi)-1)))
        
        plt.plot(e1,e3[i_U],'.',markersize=12,alpha=1,markeredgecolor='r',marker=markers[i_azi_max_sel],markeredgewidth=2,fillstyle='none')
        
        for i_azi_max in range(len(vec_azi_max)):
            plt.plot(1000,1000,'.',markeredgecolor='k',marker=markers[i_azi_max],label=r'$\theta_{max} =\beta_{max}='+utl.str_dec(vec_azi_max[i_azi_max])+'^\circ$',color='k')
        
        for i_dazi in range(len(vec_dazi)):
            plt.plot(1000,1000,'.',marker='o',markeredgecolor='k',color=cmap(i_dazi/(len(vec_dazi)-1)),label=r'$\Delta \theta =\Delta \beta='+utl.str_dec(vec_dazi[i_dazi])+'^\circ$')
        
        plt.xlim([0,101])
        plt.ylim([0,101])
        
        plt.gca().grid(visible=True) 
        plt.title('$U_\infty = '+utl.str_dec(U)+'$ m/s')
        plt.xlabel(r'$\epsilon_I$ [%]')
        plt.ylabel(r'$\epsilon_{III}$ [%]')
        
    plt.legend().set_draggable(state=True)
    plt.tight_layout()


fig = plt.figure(figsize=(18,9))
ax = fig.add_subplot(121,projection='3d')
th_plot=np.arange(360)
plt.plot(0*th_plot,0.5*utl.cosd(th_plot),0.5*utl.sind(th_plot),'r')
sel_x=(x1[0]>mins[0])*(x1[0]<maxs[0])
sel_y=(x1[1]>mins[1])*(x1[1]<maxs[1])
try:
    sel_z=(x1[2]>mins[2])*(x1[2]<maxs[2])
    sel_plot=sel_x*sel_y*sel_z
except:
    sel_plot=sel_x*sel_y

sc=ax.scatter(x1[0][sel_plot],x1[1][sel_plot],x1[2][sel_plot],s=3, c='r', alpha=1,vmin=0,vmax=1,marker='x')
sc=ax.scatter(x2[0],x2[1],x2[2],s=7, c='k', alpha=0.1,vmin=0,vmax=1)

ax.set_xlim([mins[0],maxs[0]])
ax.set_ylim([mins[1],maxs[1]])
ax.set_zlim(mins[2],maxs[2])
ax.set_box_aspect([maxs[0]-mins[0],maxs[1]-mins[1],maxs[2]-mins[2]])

ax.set_xlabel(r'$x/D$')
ax.set_ylabel(r'$y/D$')
ax.set_zlabel(r'$z/D$')
if maxs[1]-mins[1]<10:
    ax.set_yticks(np.arange(mins[1],maxs[1]+0.1))
else:
    ax.set_yticks(np.arange(mins[1],maxs[1]+0.1,5))
ax.set_zticks(np.arange(mins[2],maxs[2]+0.1))

ax.xaxis.labelpad=30

ax = fig.add_subplot(122,projection='3d')
th_plot=np.arange(360)
plt.plot(0*th_plot,0.5*utl.cosd(th_plot),0.5*utl.sind(th_plot),'r')
sc=ax.scatter(X_plot[~excl_plot],Y_plot[~excl_plot],Z_plot[~excl_plot],s=2, c=Dd_plot[~excl_plot], alpha=0.5,vmin=0,vmax=1)
ax.set_xlim([mins[0],maxs[0]])
ax.set_ylim([mins[1],maxs[1]])
ax.set_zlim(mins[2],maxs[2])
ax.set_box_aspect([maxs[0]-mins[0],maxs[1]-mins[1],maxs[2]-mins[2]])
 
plt.colorbar(sc,label='$\Delta d$',location='top')

ax.set_xlabel(r'$x/D$')
ax.set_ylabel(r'$y/D$')
ax.set_zlabel(r'$z/D$')
if maxs[1]-mins[1]<10:
    ax.set_yticks(np.arange(mins[1],maxs[1]+0.1))
else:
    ax.set_yticks(np.arange(mins[1],maxs[1]+0.1,5))
ax.set_zticks(np.arange(mins[2],maxs[2]+0.1))

ax.xaxis.labelpad=30

ax=fig.add_axes([0.05,0.4,0.1,0.1])
plt.text(0,0,r'$\Delta\theta = '+utl.str_dec(sel[0])+' ^\circ$\n'+
             r'$\theta_{max}='+utl.str_dec(sel[1])+' ^\circ$ \n'+
             r'$\Delta\beta = '+utl.str_dec(sel[-2])+' ^\circ$\n'+
             r'$\beta_{max}='+utl.str_dec(sel[-1])+' ^\circ$ \n'+
             r'$\gamma = '+utl.str_dec(yaw)+'^\circ$ \n'
             r'$\Delta r='+utl.str_dec(dr)+'$ m \n'+
             r'ppr $='+utl.str_dec(ppr)+'$ \n'+
             'scan mode: '+scan_mode+'\n'+
             'gate overlapping: '+str(overlapping)+'\n'+
             r'$\Delta n_0 = '+utl.str_dec(Dn0)+'D$ \n'+
             r'$T='+utl.str_dec(T)+'$ s \n'+
             r'$U_\infty='+utl.str_dec(vec_U)+'$ m/s \n'+
             r'$\epsilon_I='+utl.str_dec(e1)+'\%$ \n'+
             r'$\epsilon_{II} ='+utl.str_dec(e2)+'\%$ \n'+
             r'$\epsilon_{III} ='+utl.str_dec(e3)+'\%$',fontsize='small') 
plt.axis('off')

if save_fig:
    utl.mkdir('figures/'+root[5:]+Data.attrs['Scan name'])
    utl.save_all_fig(root[5:]+'/'+Data.attrs['Scan name']+'/'+Data.attrs['Scan name']+'_LiSBOA')