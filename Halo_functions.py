# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:35:46 2023

@author: sletizia
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time as tt
import sys

def Halo_sim(azi,ele,ppr,CSM,overlapping,model,source_time):
    
    if CSM:
        
        #inputs
        if model=='XR+':
            if overlapping:
                tp=0.163
                tp_idle=0.3
            else:
                tp=0.066
                tp_idle=0.19
            ta=0.967*10**(-4)
        elif model=='XR':
            if overlapping:
                tp=0.136
                tp_idle=0.22
            else:
                tp=0.023
                tp_idle=0.068
            ta=1*10**(-4)
        
            
        ppd1=500000/360#points per degree
        ppd2=250000/360#points per degree
        dt=0.01#[s] time step of scanner's model
        tau_s=tp+ta*ppr
        tau_idle=tp_idle+ta*ppr
      
        time=[0]
       
        dazi=np.diff(azi)
        dele=np.diff(ele)
        
        stop=np.concatenate(([0],np.where((np.diff(dazi)+np.diff(dele))!=0)[0]+1,[-1]))
        
        th_range=azi[stop]
        P1=-th_range*ppd1
        azi2=[th_range[0]]
        s1=np.abs(dazi[stop[:-1]]/tau_s)
        S1=np.append(5000,s1*ppd1/10)
        S1[S1>5000]=5000
        A1=50+np.zeros(len(P1))
        a1=A1*1000/ppd1
        
        b_range=ele[stop]
        P2=-b_range*ppd2
        ele2=[b_range[0]]
        s2=np.abs(dele[stop[:-1]])/tau_s
        S2=np.append(5000,s2*ppd2/10)
        S2[S2>5000]=5000
        A2=50+np.zeros(len(P2))
        a2=A2*1000/ppd2
        
        for i in range(len(th_range)-1):
            th0=th_range[i]
            th3=th_range[i+1]
            b0=b_range[i]
            b3=b_range[i+1]
            
            t=[tau_idle]
            
            dth_dt=[0]
            th=[th0]
            dth_acc=np.min([s1[i]**2/(2*a1[i]+10**-10),np.abs(th3-th0)/2])
            sign1=(th3-th0)/np.abs(th3-th0+10**-10)
            
            db_dt=[0]
            b=[b0]
            db_acc=np.min([s2[i]**2/(2*a2[i]+10**-10),np.abs(b3-b0)/2])
            sign2=(b3-b0)/np.abs(b3-b0+10**-10)
            
            for i_t in range(100000):
                t=np.append(t,t[-1]+dt)
                
                if np.abs(th3-th[-1])>dth_acc:
                    dth_dt=np.append(dth_dt,np.min([dth_dt[-1]+a1[i]*dt,s1[i]]))
                else:
                    dth_dt=np.append(dth_dt,np.max([dth_dt[-1]-a1[i]*dt,0]))
                th=np.append(th,th[-1]+dth_dt[-1]*dt*sign1)
                
                if np.abs(b3-b[-1])>db_acc:
                    db_dt=np.append(db_dt,np.min([db_dt[-1]+a2[i]*dt,s2[i]]))
                else:
                    db_dt=np.append(db_dt,np.max([db_dt[-1]-a2[i]*dt,0]))
                b=np.append(b,b[-1]+db_dt[-1]*dt*sign2)
    
                if np.max([dth_dt[-1],db_dt[-1]])==0:
                    break
            
            th[-1]=th3
            b[-1]=b3
            
            time=np.append(time,time[-1]+np.arange(t[0],t[-1],tau_s))
            azi2=np.append(azi2,np.interp(np.arange(t[0],t[-1],tau_s),t,th))
            ele2=np.append(ele2,np.interp(np.arange(t[0],t[-1],tau_s),t,b))
            
            time=np.append(time,time[-1]+tau_s)
            azi2=np.append(azi2,th[-1])
            ele2=np.append(ele2,b[-1])
        
        return time,azi2,ele2,P1,P2,S1,S2,A1,A2
    
    
def Halo_sim_v3(azi,ele,ppr,CSM,overlapping,model,source_time):
    #03/29/2023 (v 2): added SSM
    #03/30/2023 (v 3): added new calibrated accelerations and speed
    
    import pandas as pd
    time_info=pd.read_excel(source_time,sheet_name=model)
    time_info=time_info.set_index('Scan settings')
        
    if CSM:
        
        #inputs
        if model=='XR+':
            if overlapping:
                tp=0.163
            else:
                tp=0.066
            ta=0.967*10**(-4)
        elif model=='XR':
            if overlapping:
                tp=0.136
            else:
                tp=0.023
            ta=1*10**(-4)
        
        ppd1=500000/360#points per degree
        ppd2=250000/360#points per degree
        dt=0.01#[s] time step of scanner's model
        tau_s=tp+ta*ppr
      
        time=[0]
       
        dazi=np.diff(azi)
        dele=np.diff(ele)
        
        stop=np.concatenate(([0],np.where((np.diff(dazi)+np.diff(dele))!=0)[0]+1,[-1]))
        
        th_range=azi[stop]
        P1=-th_range*ppd1
        azi2=[th_range[0]]
        s1=np.abs(dazi[stop[:-1]]/tau_s)
        S1=np.append(5000,s1*ppd1/10)
        S1[S1>5000]=5000
        A1=50+np.zeros(len(P1))
        a1=A1*1000/ppd1
        
        b_range=ele[stop]
        P2=-b_range*ppd2
        ele2=[b_range[0]]
        s2=np.abs(dele[stop[:-1]])/tau_s
        S2=np.append(5000,s2*ppd2/10)
        S2[S2>5000]=5000
        A2=50+np.zeros(len(P2))
        a2=A2*1000/ppd2
        
    else:
        #inputs
        if overlapping:
            tp=time_info['Processing time']['SSM - overlapping']
            ta=time_info['Acquisition time']['SSM - overlapping']
            s1=time_info['Speed azimuth']['SSM - overlapping']
            s2=time_info['Speed elevation']['SSM - overlapping']
            a1=time_info['Acceleration azimuth']['SSM - overlapping']
            a2=time_info['Acceleration elevation']['SSM - overlapping']
        else:
            tp=time_info['Processing time']['SSM - no-overlapping']
            ta=time_info['Acquisition time']['SSM - no-overlapping']
            s1=time_info['Speed azimuth']['SSM - no-overlapping']
            s2=time_info['Speed elevation']['SSM - no-overlapping']
            a1=time_info['Acceleration azimuth']['SSM - no-overlapping']
            a2=time_info['Acceleration elevation']['SSM - no-overlapping']
            
        dt=0.01#[s] time step of scanner's model
        
        tau_s=tp+ta*ppr
        time=[0]
        
        th_range=azi
        azi2=azi.copy()
        s1=s1+np.zeros(len(azi))
        a1=a1+np.zeros(len(azi))
        
        b_range=ele.copy()
        ele2=ele.copy()
        s2=s2+np.zeros(len(azi))
        a2=a2+np.zeros(len(azi))
        
        P1=[]
        P2=[]
        S1=[]
        S2=[]
        A1=[]
        A2=[]
        
    #scanning head movement        
    for i in range(len(th_range)-1):
        th0=th_range[i]
        th3=th_range[i+1]
        if th3-th0>180:
            th3=th3-360
        if th3-th0<-180:
            th3=th3+360
        b0=b_range[i]
        b3=b_range[i+1]
        
        t=[tau_s]
        
        dth_dt=[0]
        th=[th0]
        dth_acc=np.min([s1[i]**2/(2*a1[i]+10**-10),np.abs(th3-th0)/2])
        sign1=(th3-th0)/np.abs(th3-th0+10**-10)
        
        db_dt=[0]
        b=[b0]
        db_acc=np.min([s2[i]**2/(2*a2[i]+10**-10),np.abs(b3-b0)/2])
        sign2=(b3-b0)/np.abs(b3-b0+10**-10)
        
        for i_t in range(100000):
            t=np.append(t,t[-1]+dt)
            
            if np.abs(th3-th[-1])>dth_acc:
                dth_dt=np.append(dth_dt,np.min([dth_dt[-1]+a1[i]*dt,s1[i]]))
            else:
                dth_dt=np.append(dth_dt,np.max([dth_dt[-1]-a1[i]*dt,0]))
            th=np.append(th,th[-1]+dth_dt[-1]*dt*sign1)
            
            if np.abs(b3-b[-1])>db_acc:
                db_dt=np.append(db_dt,np.min([db_dt[-1]+a2[i]*dt,s2[i]]))
            else:
                db_dt=np.append(db_dt,np.max([db_dt[-1]-a2[i]*dt,0]))
            b=np.append(b,b[-1]+db_dt[-1]*dt*sign2)

            if np.max([dth_dt[-1],db_dt[-1]])==0:
                break
        
        th[-1]=th3
        b[-1]=b3
        
        if CSM:
            time=np.append(time,time[-1]+np.arange(t[0],t[-1],tau_s))
            time=np.append(time,time[-1]+tau_s)
        
            azi2=np.append(azi2,np.interp(np.arange(t[0],t[-1],tau_s),t,th))
            ele2=np.append(ele2,np.interp(np.arange(t[0],t[-1],tau_s),t,b))
            azi2=np.append(azi2,th[-1])
            ele2=np.append(ele2,b[-1])
        else:
            time=np.append(time,time[-1]+t[-1])
    
    return time,azi2,ele2,P1,P2,S1,S2,A1,A2
            
def Halo_sim_v4(azi,ele,ppr,scan_mode,overlapping,model,source_time):
    #03/29/2023 (v 2): added SSM
    #03/30/2023 (v 3): added new calibrated accelerations and speed
    #03/31/2023 (v 4): calibration from spreadsheet, finalized
    #04/03/2023: added progress monitor, added tolerance on angular difference change
    
    t0=tt.time()
    dt=0.01#[s] time step of scanner's model
    
    time_info=pd.read_excel(source_time,sheet_name=model)
    time_info=time_info.set_index('Scan settings')
    
    if overlapping:
        overlapping_flag='overlapping'
    else:
        overlapping_flag='no-overlapping'
    
    tp=time_info['Processing time'][scan_mode+ ' - '+overlapping_flag]
    ta=time_info['Acquisition time'][scan_mode+ ' - '+overlapping_flag]
    tau_s=tp+ta*ppr
    time=[0]
    
    if scan_mode=='CSM':
        
        ppd1=500000/360#points per degree
        ppd2=250000/360#points per degree
        
        dazi=np.diff(azi)
        dele=np.diff(ele)
        stop=np.concatenate(([0],np.where(np.abs((np.diff(dazi)+np.diff(dele)))>10**-10)[0]+1,[-1]))
        
        th_range=azi[stop]
        P1=-th_range*ppd1
        azi2=[th_range[0]]
        s1=np.abs(dazi[stop[:-1]]/tau_s)
        s1[s1>50000/ppd1]=50000/ppd1
        S1=np.append(5000,s1*ppd1/10)
        S1[S1>5000]=5000
        A1=50+np.zeros(len(P1))
        a1=A1*1000/ppd1
        
        b_range=ele[stop]
        P2=-b_range*ppd2
        ele2=[b_range[0]]
        s2=np.abs(dele[stop[:-1]])/tau_s
        s2[s2>50000/ppd2]=50000/ppd2
        S2=np.append(5000,s2*ppd2/10)
        S2[S2>5000]=5000
        A2=50+np.zeros(len(P2))
        a2=A2*1000/ppd2
        
    elif scan_mode=='SSM':
        #inputs
        s1=time_info['Speed azimuth'][scan_mode+ ' - '+overlapping_flag]
        s2=time_info['Speed elevation'][scan_mode+ ' - '+overlapping_flag]
        a1=time_info['Acceleration azimuth'][scan_mode+ ' - '+overlapping_flag]
        a2=time_info['Acceleration elevation'][scan_mode+ ' - '+overlapping_flag]

        th_range=azi
        azi2=azi.copy()
        s1=s1+np.zeros(len(azi))
        a1=a1+np.zeros(len(azi))
        
        b_range=ele.copy()
        ele2=ele.copy()
        s2=s2+np.zeros(len(azi))
        a2=a2+np.zeros(len(azi))
        
        P1=[]
        P2=[]
        S1=[]
        S2=[]
        A1=[]
        A2=[]
        
    #scanning head movement        
    for i in range(len(th_range)-1):
        th0=th_range[i]
        th3=th_range[i+1]
        
        if scan_mode=='SSM':#the azimuth in SSM fins the shortest path to the next location
            if th3-th0>180:
                th3=th3-360
            if th3-th0<-180:
                th3=th3+360
                
        b0=b_range[i]
        b3=b_range[i+1]
        
        t=[tau_s]
        
        dth_dt=[0]
        th=[th0]
        dth_acc=np.min([s1[i]**2/(2*a1[i]+10**-10),np.abs(th3-th0)/2])
        sign1=(th3-th0)/np.abs(th3-th0+10**-10)
        
        db_dt=[0]
        b=[b0]
        db_acc=np.min([s2[i]**2/(2*a2[i]+10**-10),np.abs(b3-b0)/2])
        sign2=(b3-b0)/np.abs(b3-b0+10**-10)
        
        for i_t in range(100000):
            t=np.append(t,t[-1]+dt)
            
            if (th3-th[-1])*sign1>dth_acc:
                dth_dt=np.append(dth_dt,np.min([dth_dt[-1]+a1[i]*dt,s1[i]]))
            else:
                dth_dt=np.append(dth_dt,np.max([dth_dt[-1]-a1[i]*dt,0]))
            th=np.append(th,th[-1]+dth_dt[-1]*dt*sign1)
            
            if (b3-b[-1])*sign2>db_acc:
                db_dt=np.append(db_dt,np.min([db_dt[-1]+a2[i]*dt,s2[i]]))
            else:
                db_dt=np.append(db_dt,np.max([db_dt[-1]-a2[i]*dt,0]))
            b=np.append(b,b[-1]+db_dt[-1]*dt*sign2)

            if np.max([dth_dt[-1],db_dt[-1]])==0:
                break
        
        th[-1]=th3
        b[-1]=b3
        
        if scan_mode=='CSM':
            time=np.append(time,time[-1]+np.arange(t[0],t[-1],tau_s))
            time=np.append(time,time[-1]+tau_s)
            azi2=np.append(azi2,np.interp(np.arange(t[0],t[-1],tau_s),t,th))
            ele2=np.append(ele2,np.interp(np.arange(t[0],t[-1],tau_s),t,b))
            azi2=np.append(azi2,th[-1])
            ele2=np.append(ele2,b[-1])
        elif scan_mode=='SSM':
            time=np.append(time,time[-1]+t[-1])
            
        ctr=i+1
        if np.floor(ctr/len(th_range)*100)>np.floor((ctr-1)/len(th_range)*100):
            est_time=(tt.time()-t0)/ctr*(len(th_range)-ctr)
            sys.stdout.write('\r Scanning head simulator: '+str(np.floor(i/len(th_range)*100).astype(int))+'% done, '+str(round(est_time))+' s left.') 
    sys.stdout.write('\r                                                                         ')
    sys.stdout.flush()
    
    return time,azi2,ele2,P1,P2,S1,S2,A1,A2
            
            
              
    