import os
cd=os.path.dirname(__file__)

import utils as utl
import numpy as np
import Halo_functions as HF
from matplotlib import pyplot as plt

def sampling_time(dazi,ppr,lidar_model):
    
    if lidar_model=='Halo XR':
        t_d=0.333#[s]
        t_a=0.096/1000#[s/ppr]
        domega_dt=12#[deg/s**2]
        omega_max=36#[deg/s]
    elif lidar_model=='Halo XR+':
        t_d=0.346#[s]
        t_a=0.098/1000#[s/ppr]
        domega_dt=17#[deg/s**2]
        omega_max=36#[deg/s]
    else:
        raise BaseException('Invalid lidar model')
    dazi=np.array(dazi)
    ppr=np.array(ppr)
    T=np.zeros(np.shape(dazi))

    sel=dazi<omega_max**2/(2*domega_dt)
    T[sel]=(t_d+t_a*ppr[sel]+(2*dazi[sel]/domega_dt)**0.5)
    
    sel=dazi>=omega_max**2/(2*domega_dt)
    T[sel]=omega_max/(2*domega_dt)+dazi[sel]/omega_max
    
    return T

def Pareto_v2(r,theta,beta,lidar_model,ppr,scan_mode,overlapping,source_time,mins,maxs,Dn0,sigma,T_tot,vec_U,Lu,D):
    #03/31/2023 (v 2): new halo simulator
    #04/03/2023: LiSBOA with real scanning points, finalized
    
    f=np.float64(10)**np.arange(-5,3,0.01)#[Hz] range of frequencies
    
    if utl.len2(vec_U)==1:
        vec_U=np.array([vec_U])
        
    theta1,beta1=np.meshgrid(theta,beta)
    theta1=theta1.ravel() 
    beta1=beta1.ravel() 
    #%virtual lidar (with ideal geometry)
    
    [R1,TH1]=np.meshgrid(r,theta1)
    [R1,B1]=np.meshgrid(r,beta1)
    X=R1*utl.cosd(B1)*utl.cosd(TH1)
    Y=R1*utl.cosd(B1)*utl.sind(TH1)
    Z=R1*utl.sind(B1)
    x1=[np.ravel(X)/D,np.ravel(Y)/D,np.ravel(Z)/D]
   
    time,theta2,beta2,P1,P2,S1,S2,A1,A2=HF.Halo_sim_v4(theta1.ravel(),beta1.ravel(),ppr,scan_mode,overlapping,lidar_model,source_time)
    
    #%virtual lidar (with actual points)
    [R2,TH2]=np.meshgrid(r,theta2)
    [R2,B2]=np.meshgrid(r,beta2)
    X=R2*utl.cosd(B2)*utl.cosd(TH2)
    Y=R2*utl.cosd(B2)*utl.sind(TH2)
    Z=R2*utl.sind(B2)
    x2=[np.ravel(X)/D,np.ravel(Y)/D,np.ravel(Z)/D]
    
    T=time[-1]
    L=np.floor(T_tot/T)
    
    if L>0:
        #LiSBOA
        X2,Dd,excl,avg,HOM=LiSBOA_v7_2(x2,mins,maxs,Dn0,sigma)

        #Pareto
        epsilon1=np.sum(excl)/np.size(excl)*100
            
        p=np.arange(1,L)
        f_nqs=1/(2*T)
        epsilon2=[]
        epsilon3=[]
        for U in vec_U:
            tau=Lu/U#[s]
            epsilon2=np.append(epsilon2,(1/L+2/L**2*np.sum((L-p)*np.exp(-T/tau*p)))**0.5*100)
        
            ER=np.trapz(Kaimal(f[f<f_nqs],U,Lu),x=f[f<f_nqs])
            epsilon3=np.append(epsilon3,(1-ER)*100)
    else:
        epsilon1=epsilon2=epsilon3=X2=Dd=excl=np.nan
        
    return epsilon1,epsilon2,epsilon3,X2,Dd,excl,T,x1,x2

def Kaimal(f,U,Lu=340.2):
   
    return (4*Lu/U)/(1+6*f*Lu/U)**(5/3)  

def LiSBOA_v7_2(x_exp,mins,maxs,Dn0,sigma,max_iter=None,calculate_stats=False,f=None,order=2,R_max=3,grid_factor=0.25,tol_dist=0.1,max_Dd=1,verbose=True):
    #03/01/2021 (v 5): undersampling checkes in hypercubes instead of hyperspheres (faster)
    #03/02/2022: finalized
    #08/28/2023 (v 7): added HOM,handled 0 dimensions, finalized
    #09/20/2023 (v 7.1): fixed bug on dimension number, handling grid in infinite dimension, finalized
    #03/05/2024 (v 7.2): spatial bins centered
    #03/06/2024: added verbosity, finalized
    
    from scipy.special import gamma
    from scipy.interpolate import interpn
    import itertools
    import sys
    import time
         
    n=3  
    
    #outliers rejection
    if calculate_stats:
        real=~np.isnan(np.sum(np.array(x_exp),axis=0)+f)
        for j in range(n):
            x_exp[j]=x_exp[j][real]
        f=f[real]
    else:
        real=~np.isnan(np.sum(np.array(x_exp),axis=0))
        for j in range(n):
            x_exp[j]=x_exp[j][real]
    
    #Initialization
    t0=time.time()
    Dn0=np.array(Dn0) 
    n_eff=np.sum(Dn0>0)
    Dn0[Dn0==0]=10**99
    N=len(x_exp[0])
    x=np.zeros((n,N))
    xc=np.zeros(n)
    X_bin=[];
    X_vec=[];
    X2=[]
    avg=None
    HOM=None
    
    #LiSBOA setup
    dx=grid_factor*Dn0
    R_max=R_max*sigma
    V=np.pi**(n_eff/2)/gamma(n_eff/2+1)*R_max**n_eff
    
    for j in range(n):
        xc[j]=np.nanmean(x_exp[j])
        x[j]=(x_exp[j]-xc[j])/Dn0[j]       
        X_bin.append((np.arange(mins[j]-dx[j]/2,maxs[j]+dx[j]/2+dx[j]*10**-10,dx[j])-xc[j])/Dn0[j])
        X_vec.append(utl.mid(X_bin[j]))

    NoD=np.ceil(np.log10(np.max(np.abs(np.array(x)/tol_dist))))+1
    X=np.meshgrid(*[X_vec[j] for j in range(n)], indexing='ij')
    
    for j in range(n):
        X2.append(X[j]*Dn0[j]+xc[j])
      
    w=np.zeros(np.shape(X[0]),dtype=object)
    sel=np.zeros(np.shape(X[0]),dtype=object)
    val=np.zeros(np.shape(X[0]),dtype=object)
    Dd=np.zeros(np.shape(X[0]))
    N_grid=X[0].size
    dist_inf=np.zeros(n)
    
    #weights
    for j in range(n):
        dist_inf[j]=np.ceil(R_max/(dx[j]/Dn0[j]))
        
    nodes=np.where(X[0])
    counter=0
    for i in zip(*[xx for xx in nodes]):
        distSq=0
        for j in range(n):
            distSq+=(x[j]-X[j][i])**2
        s=np.where(distSq<R_max**2)   
        if len(s)>0:
            w[i]=np.exp(-distSq[s]/(2*sigma**2))
       
        #local spacing
        if Dd[i]!=10:   
            if len(s[0])>1:                
                pos_uni=np.around(x[0][s]/tol_dist)*tol_dist
                for j in range(1,n):                
                    pos_uni+=np.around(x[j][s]/tol_dist)*tol_dist*(10**NoD)**j          
                N_uni= len(np.unique(np.array(pos_uni)))
                
                if N_uni>1:
                    Dd[i]=V**(1/n_eff)/(N_uni**(1/n_eff)-1)
                else:
                    Dd[i]=np.inf
            else:
                Dd[i]=np.inf
                
            ind_inf=[]
            if Dd[i]>max_Dd:
                for j in range(n):
                    i1=max(i[j]-dist_inf[j],0)
                    i2=min(i[j]+dist_inf[j],np.shape(X[0])[j])
                    ind_inf.append(np.arange(i1,i2).astype(int))                
                for i_inf in itertools.product(*[ii for ii in ind_inf]):
                    Dd[i_inf]=10

        #store
        sel[i]=s
        
        counter+=1
        if np.floor(counter/N_grid*100)>np.floor((counter-1)/N_grid*100) and verbose:
            est_time=(time.time()-t0)/counter*(N_grid-counter)
            sys.stdout.write('\r LiSBOA:'+str(np.floor(counter/N_grid*100).astype(int))+'% done, '+str(round(est_time))+' s left.') 
    sys.stdout.write('\r                                                                         ')
    sys.stdout.flush()
    excl=Dd>max_Dd
                
    #stats
    if calculate_stats:
        avg=[]
        HOM=[]
        df=f
        for m in range(max_iter+1):
            WM=np.zeros(np.shape(X[0]))+np.nan
            WM_HOM=np.zeros(np.shape(X[0]))+np.nan
            if verbose:
                sys.stdout.write('\r Iteration #'+str(m))
                sys.stdout.flush()
            for i in zip(*[xx for xx in nodes]):
                val[i]=f[s]
                if not excl[i]:
                    fs=np.array(df[sel[i]])
                    ws=np.array(w[i])
                    reals=~np.isnan(fs+ws)
                    if sum(reals)>0:
                        fs=fs[reals]
                        ws=ws[reals]                     
                        WM[i]=sum(np.multiply(fs,ws))/sum(ws)              
                        if m>0:
                            WM_HOM[i]=sum(np.multiply(fs**order,ws))/sum(ws)  
            if m==0:
                avg.append(WM+0)
            else:
                avg.append(avg[m-1]+WM)
                HOM.append(WM_HOM)
            
            df=f-interpn(tuple(X_vec),avg[m],np.transpose(x),bounds_error=False,fill_value=np.nan)
        if verbose:
            sys.stdout.flush()
    return X2,Dd,excl,avg,HOM 

  